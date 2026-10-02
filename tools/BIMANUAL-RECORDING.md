# Recording and exporting a two-arm take

Written 2026-08-08 alongside the `--arms` support in `tools/export_lerobot.py`.
It documents one modelling decision and the procedure that decision implies.
**Nothing here has been run against real bimanual data — no such recording
exists yet.** Everything is exercised against synthetic two-arm episodes in
`tests/labeling/test_export_bimanual.py` and against the real single-arm corpus.

---

## The constraint that shapes everything

The operator cannot teleop both arms at once. There are two GELLO leaders and
one pair of hands. So a bimanual take is **one continuous episode with a handoff
in the middle**:

```
t ──────────────────────────────────────────────────────────────────▶
       right arm: source box → mat          left arm: mat → kit box
       ██████████████████████████           ███████████████████████████
       left arm parked                      right arm parked
```

At every instant exactly one arm is being driven and the other is parked.

---

## How the idle arm is represented

**By its own recorded values, unchanged.** `observation.state` is where it
actually was; `action` is what its parked leader was actually commanding. It is
not masked, not zeroed, and not moved to a separate action space.

The full argument lives in `export_lerobot.window_rows.__doc__`; the short form:

- **Not masked.** ACT emits a whole action chunk per step and the runtime
  executes it. A masked dimension has no value at inference time, so masking
  forces a second hand-written "hold" controller to invent one — and the moment
  that controller and the policy disagree about which arm is idle, an
  un-modelled arm moves. Keeping the hold *in* the action space makes the
  policy's output directly executable and makes "stay still" something the
  policy is explicitly supervised to emit.
- **Not a separate action space per arm.** Two policies cannot learn the
  handoff, and the handoff — *when is the mat ready for the left arm?* — is the
  only genuinely bimanual thing in this task. Splitting it deletes the reason to
  record bimanually at all.
- **Not a synthesised constant.** "Hold at the last commanded value" is already
  what the data contains, because a teleop follower servos to its leader. The
  moment you synthesise it instead of recording it, you hide the one way this
  representation fails.

### The one way it fails, and the gate for it

If the operator lets go of the right leader **somewhere other than where the
right follower is**, that leader keeps publishing its parked pose as the
commanded action. Train on that and the policy learns to snap the idle arm
across the workspace the instant the other arm starts working. No loss curve
shows it.

So `export_lerobot` measures, per arm per window:

| | |
|---|---|
| `ptp_rad` | max over the 6 arm joints of (max − min) of the **measured** pose. Small ⇒ parked. |
| `divergence_rad` | max \|action − state\| over the 6 arm joints. On a driven arm this is tracking error; on a parked arm it is leader-follower disagreement. |

A window where an arm did **not** move and whose divergence exceeds
`--max-idle-divergence` (default 0.10 rad) is dropped with a named reason. A
moving arm is never vetoed on divergence — tracking lag on a driven arm is
normal and large.

**Operator consequence: park each leader where its follower is before you let
go of it.** If the export starts dropping windows for `idle arm …: leader parked
… from the follower`, that is what happened.

### What is deliberately NOT a feature

There is no "which arm is active" channel in the observation. It is knowable at
export time and **not** knowable at inference time — at run time nothing knows
whose turn it is; that is exactly what the policy has to infer from the images.
Feeding it in would train a policy that cannot run. Activity is reported by the
exporter and pinned by `test_no_active_arm_feature_is_exposed`.

---

## The data contract

| | one arm (default) | two arms |
|---|---|---|
| `--arms` | `left` | `both` |
| state/action width | 7 | 14 |
| joint names | `joint_1..joint_6, gripper` | `left_joint_1..left_gripper, right_joint_1..right_gripper` |
| wrist camera key | `observation.images.wrist` | `observation.images.wrist_left`, `observation.images.wrist_right` |
| `robot_type` | `yam_left` | `yam_left_right` |
| annotations file | `annotations.json` | `annotations.json` (left) + `annotations_right.json` |

**Left is always first** in the concatenation. That order is part of the dataset
contract: swapping it trains a policy that drives the wrong arm and nothing
raises.

The bimanual wrist keys are deliberately **not** `observation.images.wrist`. A
single-arm checkpoint must refuse a bimanual dataset rather than quietly load
the right wrist into weights trained on the left.

`--arms left` reproduces every dataset exported before 2026-08-08 byte for byte
(verified: the default dry run over `recordings/` still yields 15 episodes / 77
windows, the exact composition of `ETHRC/yam_grasp_v1`).

---

## Window mode

```
--window-mode grasp   one training episode per successful grasp (default)
--window-mode full    one training episode per recorded take
```

`grasp` is what a *grasp* policy wants and is what built v1/v2. For two arms it
pools both arms' grasps, so a handoff take yields a window at the right arm's
pick and another at the left arm's.

**`full` is the mode for a handoff take.** The thing to be learned is the
sequence; cutting it into 5-second grasp windows deletes exactly that.

---

## Procedure

1. **Bring `can_follow_r` up.** It exists as an interface and is currently DOWN.
   Nothing in this file can be recorded until it is up. *(Not verified by this
   stream — no hardware was touched.)*
2. Record one continuous episode. Drive the right arm box → mat. **Park the
   right leader at the right follower's pose before letting go.** Drive the left
   arm mat → kit box.
3. Press `x` on a bad take. It now actually excludes it — before 2026-08-08 the
   filter looked for a tag that could not exist (`AUDIT.md` S7.1).
4. Label both arms:
   ```bash
   python -m robots_realtime.labeling.label_episode <ep> --arm left  \
          --open-ref 1.0 --closed-ref 0.0 --min-transport 0.10
   python -m robots_realtime.labeling.label_episode <ep> --arm right \
          --open-ref 1.0 --closed-ref 0.0 --min-transport 0.10
   ```
   Pass the refs. Without them the labeller guesses the polarity of the whole
   episode from its first sample, and on a channel with no range it now refuses
   outright and flags `gripper_range_unknown`. (`--auto-label` forwards these
   automatically as of 2026-08-08.)
5. Look at the wrist streams before exporting anything:
   ```bash
   python tools/wrist_view.py summary --episode <ep>
   python tools/wrist_view.py grasps  --episode <ep> --arm right
   ```
6. Dry-run the export and read what it drops:
   ```bash
   python tools/export_lerobot.py --root <ep> --arms both --cameras wrists \
          --window-mode full --dry-run
   ```
   The **zone gate is calibrated to where the mat was in July**
   (`GRASP_WORKSPACE_X_MIN = 0.25`) and your source box is not the mat. If good
   right-arm grasps are rejected as "outside the zone", override `--zone-x-min`
   rather than concluding the recording failed. (`DATA-PIPELINE.md` 2.2.)
7. Record which parameters you exported with. Nothing on disk does this for you
   — `AUDIT.md` S2 is the story of a deployed policy whose training set cannot
   be reproduced from the repo because the export arguments survive only in
   someone's shell history.

---

## Known gap

**There is no `camera_right` configured on this rig.** `--cameras wrists` and
`wrists_top` name it, and the exporter rejects an episode that does not have it
with `missing camera_right`. Until a right wrist camera exists, a bimanual export
has to run with `--cameras top` or `--cameras wrist` (left wrist only), which
gives the right arm no eye-in-hand view at all.

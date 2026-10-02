# rl-teleop — the rig

Recording, teleop, the message bus, and policy training for the YAM arms.
Part of the Bühler kitting pipeline: see `../README.md` for how the four repos fit together.

`yam-pick-pipeline/` imports **this repo** as a library (`sys.path.insert`), so if you
move or rename this directory you break that repo too.

## Hardware safety — read before touching a session

- **The arm has no brakes.** It sags whenever nothing is actively commanding it. Joint 2
  has been measured sagging +48.65° during a power gap while parked at home. "Parked at
  home" is not a safe state on its own, it only means the pose was correct at the time.
- **Never kill a running session on an unparked arm.** Check the pose, ask, then act.
  There is a `check_home` gate in `../yam-pick-pipeline/check_home.py` for this.
- A commanded session with no publisher does not hold (kp=0 grav-comp). This is fixed via
  `startup_hold`, but a boot with no command still leaves the arm limp.

## The bus

**`rr-bus` does NOT own the cameras.** This section used to say it did, and that was
never true of the code — `robots_realtime/rr_bus_cli.py` starts *only* the XPUB/XSUB
broker (`MessageBus`). The camera daemon it refers to was never written. **Sessions open
`/dev/video*` and the RealSense devices directly**, which is why two sessions collide and
why a "camera busy" error usually IS device contention.

Corrected 2026-08-12. It mattered: it is the same class of failure as the cameras
themselves — a signal confidently reporting something that is not so — and anyone
debugging a busy camera was being sent to look in the wrong place.

```bash
# rr-bus starts a BROKER whose lifetime is independent of any session. Useful so
# subscribers survive a session restart; it does not hold any camera.
tmux new-session -d -s bus 'cd ~/Desktop/kitting-v2/rl-teleop && uv run rr-bus'
uv run rr-session <config>.yaml --attach-bus --no-tui
```

A "camera busy" error means **another process has the device open**. Find it:

```bash
for d in /dev/video*; do h=$(lsof -t "$d" 2>/dev/null); [ -n "$h" ] && echo "$d -> $h"; done
pgrep -fa rr-session
```

Recording is session-owned, and the writer is called in-process by `Publisher.publish()`
*before* the ZMQ send — so **a perfect mp4 is not evidence that anything was published**.

## Camera health — read this before trusting any camera indicator

As of 2026-08-12 every camera runs behind `SupervisedCamera`: bounded reads,
reopen-with-backoff, and a `<node>/health` topic carrying
`{state, last_frame_age_s, consecutive_failures, device_path, reopens, ...}` with
`state ∈ {ok, degraded, reopening, failed}`.

```bash
# is a camera physically present? (opens nothing — safe with a session running)
./.venv/bin/python3 tools/preflight_cameras.py configs/yam/yam_right_kitting_teleop.yaml

# is it actually DELIVERING? the independent auditor — run before every session
./.venv/bin/python3 tools/check_streams.py

# attack the camera stack (safe: fakes and synthetic cameras, no hardware)
bash tools/red_rounds.sh
```

Three rules that came out of the hardening run:

- **`check_streams.py` must never read the health topic.** Its whole value is that it can
  catch the health topic lying. There is a test that fails if anyone changes this.
- **An episode recorded while a camera was unhealthy is stamped `"degraded": true` in
  `session_meta.json`**, naming the cameras, covering mid-episode dropouts. Filter with
  `jq -r 'select(.degraded) | .episode_dir' recordings/*/*/session_meta.json`.
- **Never fault-inject a config containing a `RobotNode`.** `robots_realtime/runtime/
  safety_guard.py` refuses, and `tools/camera_soak.py` calls it before anything starts.

## RealSense

- **Never call `hardware_reset()` on a RealSense here.** It leaves colour and IR working
  and depth dead until someone physically replugs the camera. The scan cam was broken
  this way once already.
- Depth failure signature: 0 depth frames while colour is fine. `usb1`/`usb2` are two
  halves of one physical port, so this is not fixed by re-cabling.
- Depth rides on the `/rgb` topic as float32 **metres**, not on a separate depth topic.
  Zeros are not NaN, they are real "no return" readings and must be masked explicitly.

## Python env

- `.venv` is a `uv` venv on **torch 2.7.1+cu128**. The 5090 (sm_120) needs the `+cu128`
  build; PyPI's cu126 wheel has no kernels for it and dies on the first matmul with "no
  kernel image is available". Verified 2026-08-06: the installed venv is `torch-2.7.1+cu128`.
- **The `uv run` downgrade is fixed at the root, but keep using `.venv/bin/python` anyway.**
  `uv sync` / `uv run` *used to* swap the cu128 build for cu126 — 14 packages, silently, in
  under a second. That is no longer possible from this lockfile: `pyproject.toml:48-52` pins
  torch and torchvision to an explicit `[[tool.uv.index]]`, and `uv.lock` resolves both from
  `https://download.pytorch.org/whl/cu128`. Confirmed 2026-08-06, and consistent with the
  rig's own sessions being started via `uv run rr-session`.
  Still prefer `.venv/bin/python`: the failure was silent and surfaced as "the policy is
  bad" rather than "the install changed", and the explicit interpreter costs nothing.
  Note `sort_kitting.sh` and `PLAN-ACT-V2-ZONE-FILTERED.md` still say "NEVER `uv run`" —
  over-cautious now rather than wrong. See `../AUDIT.md` §S6.
- 589 tests collect, 9 collection errors are pre-existing and not caused by your change.
  **Unverified as of 2026-08-06** — checking it means importing the test modules, which this
  cleanup pass was not permitted to do while the rig was live.

#!/usr/bin/env python3
"""Export recorded teleop episodes to a LeRobot dataset for ACT training.

Scope: GRASPING ONLY. Each recorded episode contains a whole kitting run (reach,
grasp, carry, place, repeat). A grasp policy should not be trained on the carry
and place phases, so this cuts one short LeRobot episode per successful grasp.

    recorded episode (114s, 4 grasps)
    |---------------------------------------------------------------|
        [==]        [==]              [==]        [==]
         ^grasp      ^grasp            ^grasp      ^grasp
        4 LeRobot episodes, ~5s each

WHY A WINDOW AND NOT A SEGMENT: annotations.json does carry `segments` with
phase="grasp", but those are INSTANTS — t_start == t_end, the moment the gripper
closed. Only "transport" has real duration. So the trainable window has to be
constructed: PRE_S before the close (the descent/approach) through POST_S after
it (the lift). Defaults are deliberately conservative; tune with --pre-s/--post-s.

CLOCKS. Four streams, four rates, and the cameras do not even agree with each
other on frame count within one episode (measured: 3372 / 3426 / 3422 over the
same 114.3s). Nothing can be zipped by index. Everything is resampled onto a
uniform grid by TIMESTAMP:

    grid (30Hz) .....|.........|.........|.........|.....
    yam    200Hz ||||||||||||||||||||||||||||||||||||||||   nearest
    gello 62.5Hz  |   |   |   |   |   |   |   |   |   |     nearest
    cam_top  30Hz .   |    .   |    .   |    .   |    .     nearest, staleness-capped
    cam_left 31Hz .  |   .   |   .    |   .   |   .   .     nearest, staleness-capped

ACTION vs STATE. action = gello (leader, what the operator commanded),
observation.state = yam (follower, where the arm actually was). Swapping these
trains a policy to predict where the arm already is: the loss curve looks fine
and the policy is useless on hardware. Pinned by test.

GRIPPER. Raw motor position, and the limits are auto-calibrated on every boot
(observed 5.2218 / -0.0235 on one boot). Raw values are therefore NOT comparable
across sessions. Normalised per-episode to [0,1] (0=closed, 1=open) by
labeling.segmentation.normalize_width — the SAME function the labeller uses, not
a second copy of it. The scale is taken over the WHOLE episode, not the window,
so every window of one episode shares one scale. An episode whose gripper
channel has no usable range is REJECTED with a reason, not exported with an
invented constant; see normalize_gripper below.

ARMS. --arms left (default) reproduces every dataset exported before 2026-08-08.
--arms both concatenates left then right into a 14-DoF state and action for a
take in which the operator drives one arm at a time (right: box → mat, then
left: mat → kit box). How the idle arm is represented, and why, is argued in
window_rows() and in tools/BIMANUAL-RECORDING.md.

Usage:
    uv run python tools/export_lerobot.py --repo-id ETHRC/yam_grasp_v1
    uv run python tools/export_lerobot.py --root recordings/20260728 --dry-run
    uv run python tools/export_lerobot.py --arms both --cameras wrists \\
        --window-mode full --repo-id ETHRC/yam_kitting_bimanual_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_windows as _target_windows                        # noqa: E402

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.label_episode import annotations_path
from robots_realtime.labeling.mcap_io import read_positions
from robots_realtime.labeling.segmentation import GripperRangeUnknown, normalize_width

# ── tunables ────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_PRE_S = 3.0      # descent/approach before the gripper closes
DEFAULT_POST_S = 2.0     # lift after it closes
# A grid point whose nearest camera frame is further away than this in time has
# no honest image. Two frames at 30Hz — beyond that we are inventing data.
MAX_CAM_STALENESS_S = 2.0 / DEFAULT_FPS
# A window needs at least this many frames to be worth training on.
MIN_WINDOW_FRAMES = 10

# ── W2: pose-predicate windows and the close-index gate ─────────────────────
# OPT-IN (--window-mode grasp-pose). Nothing below is reachable from the default
# "grasp" mode; existing exports are byte-identical.
#
# THE PROPERTY THAT PREDICTED THE ONE WORKING POLICY (check_handover_pose.py:
# 17-31): the FRAME INDEX of the first jaw close inside the training window,
# measured against the policy's chunk. A correctly-trained checkpoint scored 0/5
# on hardware because 46% of its windows closed after frame 100 — one committed
# chunk physically could not reach the close. Loss cannot see this.
#
# The fixed [t_close-3s, t_close+2s] window puts the close at frame 90 BY
# CONSTRUCTION, whatever the operator did. That is 90% of the chunk, so the
# close sits in the thin tail of the horizon where the policy has the least
# supervision left. The pose predicate replaces the fixed 3 s lead with the
# moment the end-effector actually began its final descent, so the window opens
# on the approach the policy has to reproduce rather than on whatever the arm
# happened to be doing three seconds earlier.
#
# The plane. Measured on the 2026-09-02 left-arm grasp demos, FK z at the close
# is ~0.10 m and the arm approaches from ~0.20-0.30 m, so a plane 5 cm above the
# grasp is inside every descent and above every close. It is a PRE-GRASP plane,
# not a table height: it is defined relative to THIS grasp's own z, so it
# survives grasps at different heights and a re-levelled table.
PREGRASP_PLANE_M = 0.05
# Frames of lead kept before the crossing, so the window contains the decision to
# descend and not only the descent. The plan's range is 10-20 frames; 15 at 30 Hz
# = 0.5 s.
PREGRASP_MARGIN_FRAMES = 15
# Set by --pregrasp-margin-s / --pregrasp-plane-m (None = the constants above
# apply, byte-identical to every export before 2026-09-05). Global-set-by-main,
# same pattern as KEEP. Raising the margin opens every window earlier (pair it
# with a larger --chunk-frames so the close stays inside the chunk); raising the
# plane starts windows nearer the top of the approach. Window-length experiment
# lineage: night of 2026-09-05.
PREGRASP_MARGIN_S_CLI: float | None = None
PREGRASP_PLANE_M_CLI: float | None = None
# The LEFT policy's TRAINING chunk, in frames of the exporter's 30 Hz grid.
# act_runner.py:129-131 executes n_action_steps=16 ("16 of 100"), so the runtime
# re-observes every 0.53 s and the executed horizon is NOT the binding
# constraint — chunk_size is: the model only ever plans chunk_size frames ahead,
# so a close beyond it was never demonstrated inside a single plan.
LEFT_CHUNK_FRAMES = 100
# The gate: the close must land within this fraction of the chunk.
#
# TUNABLE ON PURPOSE — the plan (PLAN-ACT-READINESS.md, UNRESOLVED DECISIONS)
# marks 0.8 as a starting point, not a measured threshold. Measured against the
# known-good corpus at implementation time: in the default fixed-window mode
# every window closes at frame 90 (0.90 of the chunk) by construction, so 0.8
# would reject the entire corpus — which is exactly why the gate ships only with
# the pose-predicate mode that makes the index a measurement instead of a
# constant. Override with --close-idx-frac; the distribution is always reported.
CLOSE_IDX_CHUNK_FRAC = 0.8
# FK for the pose predicate. Explicit, repo-relative: fk.py's own default is
# "urdf/yam.urdf", which is cwd-relative and silently wrong from anywhere but
# the repo root.
DEFAULT_URDF = str(Path(__file__).resolve().parents[1] / "urdf" / "yam.urdf")

# recorded camera name -> LeRobot feature suffix. The scan camera is deliberately
# absent: it looks at the packet mat, not the workspace, contributes nothing to a
# grasp policy, and is ~49% of every episode's bytes.
CAMERAS = {"camera_top": "top", "camera_left": "wrist"}

# How far an arm's joints must travel inside a window before we call it "moving".
# Radians, max over the 6 arm joints of (max - min). A parked teleop follower
# holds within ~1e-3 rad of encoder noise; a real reach is >0.1 rad on several
# joints, so anything in [0.005, 0.05] separates them. Only used to REPORT which
# arm was active — nothing is dropped on the strength of it.
ARM_MOVING_PTP_RAD = 0.02
# The safety gate that IS load-bearing (see the bimanual note above export()):
# if an arm did not move but its recorded action sits this far from its recorded
# state, the parked leader is commanding a pose the follower is not in, and
# training on it teaches the policy to jump the idle arm. Radians, per joint.
IDLE_ARM_DIVERGENCE_MAX_RAD = 0.10

# camera_left is the GRIPPER camera (640x480 USB webcam mounted at the wrist);
# camera_top is the fixed overhead view. A wrist-only dataset is a legitimate
# configuration, not a degraded one: run_pick.py detects, aims and moves with
# classical IK, so by the time ACT takes over the arm is already positioned and
# the policy only has to do the final descent and close. That is a local
# servoing job, and the wrist is the view that shows fingers and packet.
#
# ACT does not care how many cameras it gets -- the ResNet18 backbone and
# encoder_img_feat_input_proj are SHARED across cameras and the camera position
# embedding is sinusoidal, so no weight is camera-keyed. Camera count only
# changes the number of vision tokens entering the transformer encoder.
CAMERA_SETS = {
    "both": CAMERAS,
    "wrist": {"camera_left": "wrist"},
    "top": {"camera_top": "top"},
    # Bimanual sets. camera_left is the LEFT wrist and keeps the historical
    # "wrist" suffix nowhere here -- once there are two wrists, "wrist" is
    # ambiguous and a checkpoint trained on one must not silently load the
    # other. Explicit names, and a single-arm checkpoint will refuse the
    # bimanual dataset at load time rather than mis-wire itself.
    "wrists": {"camera_left": "wrist_left", "camera_right": "wrist_right"},
    "wrists_top": {"camera_top": "top",
                   "camera_left": "wrist_left", "camera_right": "wrist_right"},
    # RIGHT-arm single-wrist sets, added 2026-08-12 for the first right-arm grasp
    # dataset. They map camera_RIGHT onto the historical "wrist" suffix, so the
    # feature layout is identical to yam_grasp_v2_wrist and the two datasets are
    # directly comparable (and a v2 checkpoint can warm-start this one).
    #
    # THE HAZARD THAT BUYS: "wrist" no longer says WHICH wrist, so a left-wrist
    # checkpoint will load a right-wrist dataset without complaint. That is
    # acceptable here only because these sets are single-arm and the arm is named
    # in the repo-id. Never add camera_left to one of these.
    "wrist_right": {"camera_right": "wrist"},
    "wrist_right_top": {"camera_top": "top", "camera_right": "wrist"},
}


def resolve_cameras(cameras: dict | None) -> dict:
    """None means "the default set". Not an empty dict -- a dataset with no
    images at all would train a state-only policy, which is never what someone
    meant by omitting a flag."""
    return CAMERAS if cameras is None else cameras


JOINT_NAMES = [f"joint_{i + 1}" for i in range(C.N_ARM_JOINTS)] + ["gripper"]
N_DOF = C.N_ARM_JOINTS + 1

# Which physical arms an export covers. "left" is the default and is what every
# dataset before 2026-08-08 contains.
ARM_SETS = {"left": ("left",), "right": ("right",), "both": ("left", "right")}


def resolve_arms(arms: tuple[str, ...] | None) -> tuple[str, ...]:
    return ("left",) if arms is None else tuple(arms)


def joint_names(arms: tuple[str, ...] | None = None) -> list[str]:
    """Feature names for the concatenated state/action vector.

    ONE arm keeps the bare names (`joint_1..gripper`) so yam_grasp_v1/v2 still
    reproduce byte-for-byte and existing checkpoints still load. TWO arms get
    prefixed names, in the fixed order of ARM_SETS["both"] -- left first. The
    order is part of the dataset contract: swapping it trains a policy that
    drives the wrong arm, and nothing raises.
    """
    arms = resolve_arms(arms)
    if len(arms) == 1:
        return list(JOINT_NAMES)
    return [f"{a}_{n}" for a in arms for n in JOINT_NAMES]


def n_dof(arms: tuple[str, ...] | None = None) -> int:
    return N_DOF * len(resolve_arms(arms))


DEFAULT_TASK = "grasp the bag and lift it"
DEFAULT_BIMANUAL_TASK = "move the bag from the source box to the mat, then into the kit box"


# ── episode selection ───────────────────────────────────────────────────────
@dataclass
class Rejection:
    episode: str
    reason: str


@dataclass
class Report:
    kept: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    windows: int = 0
    frames: int = 0
    # episode -> arm -> {"moving", "ptp_rad", "divergence_rad"}, one entry per
    # written window. Bimanual only in practice; harmless for one arm.
    activity: list[tuple[str, dict]] = field(default_factory=list)
    # W2 (grasp-pose mode only): one row per candidate window —
    # (episode, close frame index, start mode, kept). REPORTED, and the gate
    # decision is already in `kept`, so a corpus can be inspected before the
    # threshold is trusted.
    close_idx: list[tuple[str, int, str, bool]] = field(default_factory=list)
    # episodes exported 10-D with target_valid=-1 everywhere because they predate the
    # recording loop (--targets-absent mask). Named so a conditioned dataset can never
    # quietly contain an unconditioned day.
    masked_episodes: list[str] = field(default_factory=list)

    def note_masked_episode(self, ep: str) -> None:
        self.masked_episodes.append(ep)

    def note_close_idx(self, ep: str, idx: int, mode: str, kept: bool) -> None:
        self.close_idx.append((ep, int(idx), mode, bool(kept)))

    def reject(self, ep: str, reason: str) -> None:
        self.rejected.append(Rejection(ep, reason))

    def note_activity(self, ep: str, activity: dict) -> None:
        self.activity.append((ep, activity))


def episode_dirs(root: Path, arms: tuple[str, ...] | None = None) -> list[Path]:
    if any((root / f"yam_{a}.mcap").exists() for a in resolve_arms(arms)):
        return [root]
    # .trash holds episodes the operator threw away in the cockpit. Delete is a
    # move so it stays undoable — but it must never be training data.
    return sorted(p for p in root.rglob("episode_*")
                  if p.is_dir() and ".trash" not in p.parts)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def in_zone(g: dict, x_min: float | None = None, y_max: float | None = None) -> bool:
    """Is this grasp inside the trainable zone?

        y_max  ─────────────────────────────  drop above (corner of the mat)
               │                           │
               │      T R A I N A B L E    │
               │                           │
               └───────────────────────────┘
             x_min
             drop left (near/high, mid-air or mislabelled)

    Two independent bounds, defaulting to constants.GRASP_WORKSPACE_X_MIN and
    constants.GRASP_ZONE_Y_MAX. ``y_max=None`` means no lateral gate, which is
    the default and reproduces yam_grasp_v1 exactly.

    THE BOUNDS DO DIFFERENT JOBS, do not collapse them into one "workspace":
    x_min removes grasps that are near AND high (z 0.162-0.226 vs a corpus mean
    of 0.120) -- mid-air or mislabelled, not table grasps. y_max removes grasps
    that are perfectly good table grasps in a part of the mat you may not want
    to train on. The first is a data-quality cut, the second is a task-scope
    choice, and only the first has a 176 mm empty band justifying it.

    Grasp-level, NOT episode-level: episodes holding out-of-zone grasps also
    hold good ones, so rejecting the whole episode would throw away 23 usable
    windows to remove 4 bad ones.

    Fails OPEN: a grasp with no ee_pose is unknown, not out-of-bounds, and this
    gate only removes grasps it can positively measure as outside. Silently
    dropping unlabelled data here would hide a labeller bug as a zone result.
    (All 81 grasps in the corpus carry a pose; this is the guard, not the common
    path.) Fails open on BOTH bounds together -- a pose is present or it is not.
    """
    x_min = C.GRASP_WORKSPACE_X_MIN if x_min is None else x_min
    if y_max is None:
        y_max = C.GRASP_ZONE_Y_MAX

    pose = g.get("ee_pose")
    if not pose:
        return True
    if float(pose[0]) < x_min:
        return False
    if y_max is not None and float(pose[1]) > y_max:
        return False
    return True


def zone_label(g: dict, x_min: float | None = None,
               y_max: float | None = None) -> str:
    """Why a grasp is in or out: 'in' | 'near' | 'corner' | 'nopose'.

    The review tool badges with this so a dropped grasp says WHICH bound
    dropped it. Reads the same bounds as in_zone and must stay consistent with
    it -- pinned by test_zone_label_agrees_with_in_zone.
    """
    x_min = C.GRASP_WORKSPACE_X_MIN if x_min is None else x_min
    if y_max is None:
        y_max = C.GRASP_ZONE_Y_MAX

    pose = g.get("ee_pose")
    if not pose:
        return "nopose"
    if float(pose[0]) < x_min:
        return "near"
    if y_max is not None and float(pose[1]) > y_max:
        return "corner"
    return "in"


def operator_rejected(ep: Path) -> str | None:
    """The tag the operator used to reject this take live, or None.

    Reads the SAME constant review_corpus.py does. The bug this replaces:
    the filter looked for the tag "x", but "x" is the KEYBOARD KEY -- tui.py
    maps it to the tag "bad", session.py writes "bad", and control_server.py
    rejects a literal "x" with a 400. So the tag "x" could not exist in any
    operator_flags.json and the filter never once fired. (AUDIT.md S7.1)
    """
    flags = load_json(ep / "operator_flags.json") or {}
    tags = {f.get("tag") for f in flags.get("flags", []) if isinstance(f, dict)}
    hit = sorted(t for t in tags if t in C.OPERATOR_BAD_TAGS)
    return hit[0] if hit else None


def usable_grasps(ep: Path, workspace_gate: bool = True,
                  x_min: float | None = None,
                  y_max: float | None = None,
                  arm: str = "left") -> tuple[list[dict], str | None]:
    """Successful grasp attempts in this episode, or (,reason-it-was-rejected).

    Every filter here corresponds to a real failure seen in the recorded corpus,
    not a hypothetical one.

    ``workspace_gate=False`` returns the out-of-zone grasps too, so a review
    tool can show what was dropped and why. Nothing that writes training data
    should pass False.

    ``x_min``/``y_max`` override the zone; None on both reproduces v1.
    """
    ann_file = annotations_path(ep, arm)
    ann = load_json(ann_file)
    if ann is None:
        return [], f"no {ann_file.name}"           # 2 of 30 episodes

    meta = ann.get("episode_meta") or {}
    attempts = ann.get("grasp_attempts") or []

    # `outcome` DEFAULTS to "success" and is not derived from whether anything
    # was actually grasped: 3 of 18 "success" episodes hold zero grasp_attempts.
    # So the attempt list is the authority and outcome is only a veto.
    if meta.get("outcome") == "aborted":
        return [], "episode outcome=aborted"       # 10 of 28
    if not attempts:
        return [], "zero grasp_attempts (label says success, nothing was grasped)"

    # The operator pressed 'x' during the take = "that one was bad". Their
    # judgement outranks the labeller's; that is the whole point of the flag.
    bad = operator_rejected(ep)
    if bad is not None:
        return [], f"operator flagged {bad!r} (bad take)"

    good = [a for a in attempts if a.get("outcome") == "success"]
    if not good:
        return [], f"{len(attempts)} grasp attempts, none with outcome=success"

    if workspace_gate:
        n_before = len(good)
        good = [a for a in good if in_zone(a, x_min, y_max)]
        if not good:
            xm = C.GRASP_WORKSPACE_X_MIN if x_min is None else x_min
            ym = C.GRASP_ZONE_Y_MAX if y_max is None else y_max
            bound = f"x < {xm}" if ym is None else f"x < {xm} or y > {ym}"
            return [], f"all {n_before} grasps outside the zone ({bound})"
    return good, None


# ── operator keep-list ──────────────────────────────────────────────────────
# Set by --keep. None means "no keep-list", which is NOT the same as an empty
# one: None exports everything usable_grasps returns, {} exports nothing.
KEEP: dict[str, list[float]] | None = None

# --windows: {episode_name: [[t_abs_start, t_abs_end], ...]} — explicit training
# windows for "full" mode, written by tools/segment_home_episodes.py (one window
# per home→home stretch). Episodes absent from the list are REJECTED loudly, so
# a windows file that was built for a different night cannot silently export a
# whole take as one episode.
WINDOWS: dict[str, list[tuple[float, float]]] | None = None


WINDOW_TASKS: dict[str, list[str | None]] | None = None

# --targets: the SAM2 target the recording loop logged per home-to-home cycle, appended
# to observation.state as [target_u, target_v, target_valid]. None = plain export, and
# every existing invocation stays byte-identical because build_features() and the frame
# dict both branch on this being None. See tools/target_windows.py for the
# representation and why it is normalised pixels rather than metres.
TARGETS: dict[str, list] | None = None
TARGET_MANIFEST: dict | None = None
TARGET_DIM_NAMES = ("target_u", "target_v", "target_valid")
# "reject" (default) or "mask" -- see the note at the per-episode check in the export loop
TARGETS_ABSENT = "reject"


def load_windows(path) -> dict[str, list[tuple[float, float]]]:
    """Read a segment file → {episode: [(t0, t1), ...]} (absolute seconds).

    2026-09-16: the file may also carry ``"tasks": {episode: [str, ...]}`` aligned
    one-to-one with ``windows[episode]`` (before sorting). When present, each window
    is exported with ITS OWN task string instead of the episode-wide one, which is
    how a phase-conditioned policy ("isolate" vs "pick and place") gets its labels
    without touching the recordings. Filled into the module global WINDOW_TASKS
    (sorted in step with the windows); absent or null entries fall back to the
    episode task. Written by training_experiment/tools/phase_split.py.
    """
    global WINDOW_TASKS
    data = load_json(Path(path))
    if data is None:
        raise SystemExit(f"--windows: cannot read {path}")
    wins = data.get("windows") if isinstance(data, dict) else None
    if not isinstance(wins, dict) or not wins:
        raise SystemExit(f"--windows: {path} has no non-empty 'windows' dict")
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if tasks is not None and not isinstance(tasks, dict):
        raise SystemExit(f"--windows: 'tasks' must be a dict {{episode: [str, ...]}}")
    out: dict[str, list[tuple[float, float]]] = {}
    task_out: dict[str, list[str | None]] = {}
    for ep, lst in wins.items():
        tl = (tasks or {}).get(ep)
        if tl is not None and len(tl) != len(lst):
            raise SystemExit(f"--windows: {ep} has {len(lst)} windows but {len(tl)} tasks")
        rows = []
        for k, w in enumerate(lst):
            if not (isinstance(w, (list, tuple)) and len(w) == 2 and w[1] > w[0]):
                raise SystemExit(f"--windows: bad window for {ep}: {w!r}")
            rows.append((float(w[0]), float(w[1]), (tl[k] if tl is not None else None)))
        rows.sort(key=lambda r: (r[0], r[1]))
        out[str(ep)] = [(r[0], r[1]) for r in rows]
        task_out[str(ep)] = [r[2] for r in rows]
    for ep, lst in out.items():
        for (a0, a1), (b0, b1) in zip(lst, lst[1:]):
            if b0 < a1:
                raise SystemExit(f"--windows: {ep} has overlapping windows "
                                 f"[{a0:.2f},{a1:.2f}] and [{b0:.2f},{b1:.2f}] — "
                                 "re-run tools/segment_home_episodes.py (pads are clamped there)")
    WINDOW_TASKS = task_out if tasks else None
    return out

# t_close comes back through JSON, so allow for float round-trip only — not for
# "near enough". Two grasps in these takes are never closer than ~1.5 s, so a
# tolerance this tight cannot match the wrong grasp.
KEEP_T_TOL = 1e-3


def load_keep_list(path) -> dict[str, list[float]]:
    """Read tools/review_grasps.py's keep-list JSON → {episode: [t_close, ...]}."""
    data = load_json(Path(path))
    if data is None:
        raise SystemExit(f"--keep: cannot read {path}")
    entries = data.get("keep") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise SystemExit(f"--keep: {path} has no 'keep' list")
    out: dict[str, list[float]] = {}
    for e in entries:
        ep = e.get("episode")
        t = e.get("t_close")
        if ep is None or t is None:
            raise SystemExit(f"--keep: entry missing episode/t_close: {e!r}")
        out.setdefault(str(ep), []).append(float(t))
    return out


def filter_by_keep_list(ep, grasps: list[dict]) -> tuple[list[dict], list[float]]:
    """Keep only grasps this episode's keep-list names. Returns (kept, unmatched).

    ``unmatched`` is the keep-list times that found no grasp — the caller must
    treat a non-empty list as an error rather than exporting the remainder,
    because a partial match means the reviewed set and the exported set differ.
    """
    wanted = list((KEEP or {}).get(ep.name, []))
    kept, used = [], set()
    for g in grasps:
        t = g.get("t")
        if t is None:
            continue
        for i, w in enumerate(wanted):
            if i not in used and abs(float(t) - w) <= KEEP_T_TOL:
                kept.append(g)
                used.add(i)
                break
    missing = [w for i, w in enumerate(wanted) if i not in used]
    return kept, missing


def grasp_windows_indexed(grasps: list[dict], t0: float, t1: float,
                          pre_s: float, post_s: float) -> list[tuple[int, float, float]]:
    """``grasp_windows`` but each window keeps the index of the grasp it came from.

    THE RULE LIVES HERE, ONCE. A window that clips to nothing is dropped, so the
    output can be SHORTER than the input — and without the index the caller has
    no way to say which grasps survived. review_grasps.py used to treat that
    length mismatch as "reject the whole episode", which threw away 30 good
    grasps because a wrist camera died in the last 79 s of a 392 s take, while
    the exporter happily wrote those same 30. A review tool that shows less than
    the exporter writes is the same class of lie as one that shows more.
    """
    ts = sorted(float(g["t"]) for g in grasps if g.get("t") is not None)
    out = []
    for i, t in enumerate(ts):
        lo = max(t0, t - pre_s)
        hi = min(t1, t + post_s)
        if i > 0:                       # never reach back past the previous grasp
            lo = max(lo, (ts[i - 1] + t) / 2.0)
        if i + 1 < len(ts):             # never reach forward past the next one
            hi = min(hi, (t + ts[i + 1]) / 2.0)
        if hi > lo:
            out.append((i, lo, hi))
    return out


def pregrasp_start(t_close: float, times: np.ndarray, z: np.ndarray, *,
                   plane_m: float, margin_s: float,
                   floor_t: float) -> tuple[float, str]:
    """When did the final descent to THIS grasp begin? → (start_time, mode).

    The predicate: the LAST time before the close at which the end-effector was
    still above a plane ``plane_m`` above its own height at the close, minus a
    margin. "Last" and not "first" on purpose — an operator who hovers, lifts
    away and comes back down crosses the plane several times, and only the final
    crossing belongs to the grasp being windowed.

    ``mode`` is how the start was decided, and it is reported, never inferred:
        "pose"    — a crossing was found and used
        "clamp"   — no crossing inside the search span (the arm was already below
                    the plane, e.g. a re-grasp from a low hover) → the historical
                    ``t_close - pre_s`` clamp, i.e. exactly the old behaviour
        "clamped" — a crossing was found but sits earlier than the clamp → clamp

    FAILS TO THE OLD BEHAVIOUR, never to something new: every branch that cannot
    measure a descent returns ``floor_t``. A pose predicate that misfires
    therefore produces the window this exporter has always produced, and the
    close-index gate then catches a late close rather than exporting it silently.
    """
    t = np.asarray(times, float)
    zz = np.asarray(z, float)
    m = (t >= floor_t) & (t <= t_close)
    if not m.any() or zz.size != t.size:
        return floor_t, "clamp"
    tt, zs = t[m], zz[m]
    plane = float(zs[-1]) + plane_m          # relative to the grasp's own height
    above = np.nonzero(zs >= plane)[0]
    if above.size == 0:
        return floor_t, "clamp"
    lo = float(tt[above[-1]]) - margin_s
    if lo <= floor_t:
        return floor_t, "clamped"
    return lo, "pose"


def grasp_windows_pose_indexed(
    grasps: list[dict], t0: float, t1: float, pre_s: float, post_s: float,
    ee_z, *, plane_m: float = PREGRASP_PLANE_M,
    margin_s: float = PREGRASP_MARGIN_FRAMES / DEFAULT_FPS,
) -> list[tuple[int, float, float, float, str]]:
    """``grasp_windows_indexed`` with a pose-predicate start.

    → (index into the time-sorted grasps, lo, hi, t_close, start mode).

    ``ee_z(grasp)`` returns ``(times, z)`` for the arm that made that grasp, or
    None when no FK is available — in which case that window falls back to the
    fixed clamp. The non-overlap clipping is IDENTICAL to
    ``grasp_windows_indexed`` (same rule, applied after the start is chosen):
    two windows sharing frames teach contradictory actions for one image.
    """
    order = sorted(range(len(grasps)), key=lambda i: float(grasps[i]["t"]))
    ts = [float(grasps[i]["t"]) for i in order]
    out: list[tuple[int, float, float, float, str]] = []
    for k, gi in enumerate(order):
        t = ts[k]
        floor_t = max(t0, t - pre_s)
        tz = ee_z(grasps[gi])
        if tz is None:
            lo, mode = floor_t, "clamp"
        else:
            lo, mode = pregrasp_start(t, tz[0], tz[1], plane_m=plane_m,
                                      margin_s=margin_s, floor_t=floor_t)
        hi = min(t1, t + post_s)
        if k > 0:                       # never reach back past the previous grasp
            lo = max(lo, (ts[k - 1] + t) / 2.0)
        if k + 1 < len(ts):             # never reach forward past the next one
            hi = min(hi, (t + ts[k + 1]) / 2.0)
        if hi > lo:
            out.append((k, lo, hi, t, mode))
    return out


def close_frame_index(t_close: float, lo: float, fps: int) -> int:
    """Frame index of the jaw close inside a window that starts at ``lo``.

    The same quantity check_handover_pose.py measures on an exported corpus —
    computed here BEFORE the export instead of after the training run.
    """
    return int(round((t_close - lo) * fps))


def close_idx_gate(chunk_frames: int, frac: float) -> int:
    """Highest close frame index a window may have. See CLOSE_IDX_CHUNK_FRAC."""
    return int(chunk_frames * frac)


def grasp_windows(grasps: list[dict], t0: float, t1: float,
                  pre_s: float, post_s: float) -> list[tuple[float, float]]:
    """[close - pre, close + post] per grasp, clipped so windows never overlap.

    Overlap matters: two grasps 2s apart with pre=3/post=2 would otherwise share
    frames, and the same frames appearing in two training episodes silently
    inflates the dataset while teaching contradictory actions for one image.

    A thin wrapper over ``grasp_windows_indexed`` — do not fork the rule.
    """
    return [(lo, hi) for _, lo, hi in
            grasp_windows_indexed(grasps, t0, t1, pre_s, post_s)]


# ── signals ─────────────────────────────────────────────────────────────────
def normalize_gripper(col: np.ndarray, open_ref: float | None = None,
                      closed_ref: float | None = None) -> np.ndarray:
    """Raw gripper motor position -> [0,1], 0=closed 1=open, per episode.

    A THIN WRAPPER over labeling.segmentation.normalize_width, which is the one
    implementation. It used to be a second, independent one, and the two
    disagreed on the only case where either had no information: this file
    returned all-ONES ("jaws wide open"), the labeller returned all-ZEROS ("jaws
    fully shut"), over the same recordings. So the grasp corpus and the ACT
    tensors built from one session contradicted each other and the ACT gripper
    action channel was trained on a constant. (AUDIT.md S1.3.)

    Neither constant was right. There is no defensible value, so the shared
    function raises GripperRangeUnknown and plan_episode drops the episode with
    a reason -- a named rejection in the report instead of a silent constant in
    the dataset.

    Raw units differ per rig AND per boot (limits are auto-detected at startup),
    so an absolute value means nothing across sessions -- which is why refs are
    optional here even though passing them is always better.
    """
    return normalize_width(col, open_ref=open_ref,
                           closed_ref=closed_ref).astype(np.float32)


def nearest_index(sorted_t: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the nearest sample in sorted_t for each target time."""
    idx = np.searchsorted(sorted_t, targets)
    idx = np.clip(idx, 1, len(sorted_t) - 1)
    left, right = sorted_t[idx - 1], sorted_t[idx]
    return np.where(targets - left <= right - targets, idx - 1, idx)


class CameraStream:
    """One recorded mp4 + its per-frame timestamps, read strictly forward.

    Sequential decode, never seek. cv2 seeking on h264 lands on the nearest
    keyframe and silently returns the wrong frame; sequential reads are also an
    order of magnitude faster across a whole episode.
    """

    def __init__(self, mp4: Path, stamps: Path):
        import cv2
        self.t = np.load(stamps).astype(float)
        self.cap = cv2.VideoCapture(str(mp4))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {mp4}")
        self._pos = -1
        self._frame = None
        self._cv2 = cv2

    def frame_at(self, index: int):
        """RGB frame at the given index, decoding forward as needed."""
        if index < self._pos:
            raise RuntimeError("backward seek requested; windows must be sorted")
        while self._pos < index:
            ok, bgr = self.cap.read()
            if not ok:
                return None
            self._pos += 1
            self._frame = bgr
        if self._frame is None:
            return None
        # cv2 decodes BGR; LeRobot stores RGB. Getting this wrong costs a full
        # training run to notice, and only if someone looks at the images.
        return self._cv2.cvtColor(self._frame, self._cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


# ── export ──────────────────────────────────────────────────────────────────
def build_features(shapes: dict[str, tuple[int, int]], cameras: dict | None = None,
                   arms: tuple[str, ...] | None = None) -> dict:
    """shapes: recorded camera name -> (height, width).

    Per-camera, NOT one shared resolution: the rig records camera_top at
    1280x720 and camera_left at 640x480. Assuming they match makes
    LeRobotDataset reject every wrist frame at add_frame() time.
    """
    names = joint_names(arms)
    dof = (len(names),)
    # The target rides on observation.state and NOWHERE else. Both VLAs on this rig
    # turn observation.state into text inside the prompt (MolmoAct2 discretises it to
    # 256 levels, processor_molmoact2.py:341-350; pi0.5 writes "Task: {task}, State:
    # {state}", processor_pi05.py:67-74) and NEITHER reads
    # observation.environment_state, so extra state dims are the only numeric channel
    # that reaches either model without touching model code.
    #
    # ACTION IS DELIBERATELY NOT WIDENED. The deploy path truncates the policy's output
    # to 7 (training_experiment/tools/vla_policy_server.py:214, `out = out[:, :7]`), so
    # a wider action would train dimensions that are thrown away at inference.
    state_names = list(names) + (list(TARGET_DIM_NAMES) if TARGETS is not None else [])
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),),
                              "names": state_names},
        "action": {"dtype": "float32", "shape": dof, "names": names},
    }
    cams = resolve_cameras(cameras)
    missing = [c for c in cams if c not in shapes]
    if missing:
        # Almost always a call site that forgot to pass `cameras` through, so the
        # shapes were probed for one set and the features built for another. The
        # bare KeyError this replaces pointed at the dict, not at the mismatch.
        raise KeyError(
            f"no probed shape for {missing}; features requested {sorted(cams)} "
            f"but shapes cover {sorted(shapes)} -- pass the same `cameras` to "
            f"probe_shapes() and build_features()")
    for cam, suffix in cams.items():
        h, w = shapes[cam]
        feats[f"observation.images.{suffix}"] = {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def probe_shapes(plan: dict, cameras: dict | None = None) -> dict[str, tuple[int, int]] | None:
    """Decode one frame per camera to learn each one's real resolution."""
    shapes = {}
    for cam in resolve_cameras(cameras):
        s = CameraStream(*plan["cams"][cam])
        try:
            img = s.frame_at(0)
            if img is None:
                return None
            shapes[cam] = img.shape[:2]
        finally:
            s.close()
    return shapes


def arm_activity(state: np.ndarray, action: np.ndarray) -> dict:
    """Did this arm move over these rows, and does its command match its pose?

    ``state`` and ``action`` are (n, N_DOF) slices already resampled onto the
    same grid. Returns the two numbers the bimanual export has to reason about:

        ptp_rad        max over the 6 arm joints of (max - min) of the MEASURED
                       pose. Small = the arm was parked for this whole window.
        divergence_rad max |action - state| over the 6 arm joints. On a teleop
                       follower this is the tracking error, normally ~1e-2.

    The pair matters because "parked" is only safe if the parked LEADER agrees
    with the parked FOLLOWER. A leader let go of at a different pose keeps
    publishing that pose as the commanded action; training on it teaches the
    policy to snap the idle arm across the workspace the moment the other arm
    starts working. That failure is invisible in a loss curve.
    """
    j = slice(0, C.N_ARM_JOINTS)
    if state.size == 0 or action.size == 0:
        return {"ptp_rad": 0.0, "divergence_rad": 0.0, "moving": False}
    ptp = float(np.max(np.ptp(state[:, j], axis=0))) if state.shape[0] > 1 else 0.0
    div = float(np.max(np.abs(action[:, j] - state[:, j])))
    return {"ptp_rad": ptp, "divergence_rad": div, "moving": ptp > ARM_MOVING_PTP_RAD}


def read_arm(ep: Path, arm: str,
             open_ref: float | None = None,
             closed_ref: float | None = None,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t_state, state, t_action, action) for one arm, gripper channel normalised.

    action = gello (leader, what the operator commanded), observation.state =
    yam (follower, where the arm actually was). Swapping them trains a policy to
    predict where the arm already is: the loss curve looks fine and the policy
    is useless on hardware. Pinned by test.

    ``open_ref``/``closed_ref`` are the gripper's PHYSICAL limits. Passing them
    makes the channel absolute; leaving them None keeps the historical
    per-episode percentile scale, which is only valid when the episode happens
    to span the full open→closed range. That distinction bites hardest on SHORT
    episodes — one grasp-and-place cycle per take — where the same physical jaw
    opening otherwise lands on a different number in every episode.
    """
    t_yam, p_yam = read_positions(ep / f"yam_{arm}.mcap", f"yam_{arm}")
    t_gel, p_gel = read_positions(ep / f"gello_{arm}.mcap", f"gello_{arm}")
    if t_yam.size == 0 or t_gel.size == 0:
        raise RuntimeError(f"empty joint stream for {arm}")
    # Gripper scale over the WHOLE episode so all its windows share one scale.
    state = p_yam.astype(np.float32).copy()
    action = p_gel.astype(np.float32).copy()
    state[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(
        state[:, C.GRIPPER_JOINT_INDEX], open_ref, closed_ref)
    action[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(
        action[:, C.GRIPPER_JOINT_INDEX], open_ref, closed_ref)
    return t_yam, state, t_gel, action


def ee_z_lookup(streams: dict, arms: tuple[str, ...], urdf_path: str,
                fps: int, pre_s: float):
    """Build ``ee_z(grasp) -> (times, z)`` over the approach to each grasp.

    FK is run ONLY on the ``pre_s`` seconds before each close, resampled to the
    export grid (≈90 samples per grasp at 30 Hz) — not on the whole 200 Hz joint
    stream, which would be ~20k FK calls per episode for a number that is only
    needed near the close. No video is touched.

    The URDF path is passed in explicitly: ``fk.ForwardKinematics``'s default is
    the cwd-relative "urdf/yam.urdf", which resolves to nothing from anywhere but
    the repo root and would silently produce no windows.
    """
    from robots_realtime.labeling.fk import ForwardKinematics

    fk = ForwardKinematics(urdf_path)

    def lookup(g: dict):
        arm = g.get("arm") or arms[0]
        if arm not in streams:
            return None
        t_s, state, _, _ = streams[arm]
        t = float(g["t"])
        n = max(2, int(pre_s * fps) + 1)
        grid = t - np.arange(n - 1, -1, -1) / fps          # [t-pre_s .. t]
        grid = grid[(grid >= float(t_s[0])) & (grid <= float(t_s[-1]))]
        if grid.size < 2:
            return None
        idx = nearest_index(t_s, grid)
        z = fk.ee_positions(state[idx, : C.N_ARM_JOINTS])[:, 2]
        return grid, z

    return lookup


def plan_episode(ep: Path, pre_s: float, post_s: float, fps: int, report: Report,
                 x_min: float | None = None, y_max: float | None = None,
                 cameras: dict | None = None,
                 arms: tuple[str, ...] | None = None,
                 window_mode: str = "grasp",
                 open_ref: float | None = None,
                 closed_ref: float | None = None,
                 urdf_path: str = DEFAULT_URDF,
                 chunk_frames: int = LEFT_CHUNK_FRAMES,
                 close_idx_frac: float = CLOSE_IDX_CHUNK_FRAC):
    """Everything needed to write this episode's windows, or None if unusable.

    ``arms`` is one or more physical arms; with more than one the per-arm state
    and action vectors are concatenated in that order.

    ``window_mode``:
      "grasp" — one training window per successful grasp (the historical
                behaviour, and what a grasp policy wants). With two arms the
                grasps of BOTH arms go into one pool, so a handoff take yields
                a window at the right arm's pick and another at the left arm's.
      "full"  — one window spanning the whole recorded episode. This is the mode
                for the bimanual handoff take: the thing to be learned is the
                SEQUENCE (right arm box→mat, then left arm mat→kit box) and
                cutting it into grasp windows deletes exactly that.
      "grasp-pose" — W2. Same one-window-per-grasp rule as "grasp", but the
                window OPENS when the end-effector actually began its final
                descent (pregrasp_start) instead of a fixed pre_s earlier, and
                every window must then close within
                ``close_idx_frac × chunk_frames`` of the LEFT policy's chunk or
                it is rejected with its measured index. OPT-IN: "grasp" is
                untouched, and the same episode exports the same windows there
                as it did before this mode existed.
    """
    arms = resolve_arms(arms)

    grasps: list[dict] = []
    for arm in arms:
        g, why = usable_grasps(ep, x_min=x_min, y_max=y_max, arm=arm)
        if why:
            if window_mode in ("grasp", "grasp-pose"):
                report.reject(ep.name, why if len(arms) == 1 else f"[{arm}] {why}")
                return None
            continue
        grasps.extend(g)

    # Operator keep-list from tools/review_grasps.py. The reviewer looked at every
    # window and said which ones go in; nothing else in this file can know that a
    # bag was placed upside-down or that the grip slipped after the lift, because
    # neither leaves a trace in the gripper width or the joint stream.
    if KEEP is not None:
        grasps, missing = filter_by_keep_list(ep, grasps)
        if missing:
            # LOUD, not silent. A keep-list entry that matches nothing means the
            # annotations were re-generated after the review, so the reviewed
            # windows and the exported ones are not the same windows.
            report.reject(ep.name,
                          f"keep-list names {len(missing)} grasp(s) with no matching "
                          f"t_close in annotations (re-run review after re-labelling): "
                          f"{[round(t, 3) for t in missing[:3]]}")
            return None
        if not grasps:
            report.reject(ep.name, "no grasps in the keep-list")
            return None
    # A bad take is a bad take for every arm in it -- checked even in "full"
    # mode, where the per-arm annotations may legitimately be missing.
    bad = operator_rejected(ep)
    if bad is not None:
        report.reject(ep.name, f"operator flagged {bad!r} (bad take)")
        return None

    streams = {}
    try:
        for arm in arms:
            streams[arm] = read_arm(ep, arm, open_ref, closed_ref)
    except GripperRangeUnknown as e:
        # Used to be silently exported as an all-open (or all-shut) gripper channel.
        report.reject(ep.name, f"gripper channel unusable: {e}")
        return None
    except Exception as e:
        report.reject(ep.name, f"mcap read failed: {e}")
        return None

    cams = {}
    for cam in resolve_cameras(cameras):
        mp4 = ep / f"{cam}-images-rgb.mp4"
        stamps = ep / f"{cam}-rgb-timestamp.npy"
        if not mp4.exists() or not stamps.exists():
            report.reject(ep.name, f"missing {cam}")
            return None
        cams[cam] = (mp4, stamps)

    # The common span of every stream from every arm.
    t0 = max(float(s[0][0]) for s in streams.values())
    t0 = max(t0, max(float(s[2][0]) for s in streams.values()))
    t1 = min(float(s[0][-1]) for s in streams.values())
    t1 = min(t1, min(float(s[2][-1]) for s in streams.values()))

    window_meta: list[dict] = []
    if window_mode == "full":
        if t1 <= t0:
            report.reject(ep.name, "arms' recorded spans do not overlap")
            return None
        if WINDOWS is None:
            windows = [(t0, t1)]
        else:
            wanted = WINDOWS.get(ep.name)
            if not wanted:
                report.reject(ep.name, "not in --windows list")
                return None
            windows = []
            for k, (lo, hi) in enumerate(wanted):
                clo, chi = max(lo, t0), min(hi, t1)
                if chi - clo < (hi - lo) * 0.9:
                    # A window that mostly falls outside the recorded span was
                    # cut against a different take; say so instead of trimming.
                    report.reject(ep.name, f"--windows #{k + 1} [{lo:.1f},{hi:.1f}] "
                                  f"lies outside the recorded span [{t0:.1f},{t1:.1f}]")
                    continue
                windows.append((clo, chi))
                wtask = ((WINDOW_TASKS or {}).get(ep.name) or [None] * len(wanted))[k]
                window_meta.append({"window": k + 1, "t0": clo, "t1": chi, "task": wtask})
            if not windows:
                report.reject(ep.name, "no --windows entry inside the recorded span")
                return None
    elif window_mode == "grasp-pose":
        max_idx = close_idx_gate(chunk_frames, close_idx_frac)
        cand = grasp_windows_pose_indexed(
            grasps, t0, t1, pre_s, post_s,
            ee_z_lookup(streams, arms, urdf_path, fps, pre_s),
            plane_m=(PREGRASP_PLANE_M if PREGRASP_PLANE_M_CLI is None
                     else PREGRASP_PLANE_M_CLI),
            margin_s=(PREGRASP_MARGIN_FRAMES / DEFAULT_FPS
                      if PREGRASP_MARGIN_S_CLI is None
                      else PREGRASP_MARGIN_S_CLI))
        windows = []
        for _k, lo, hi, t_close, mode in cand:
            idx = close_frame_index(t_close, lo, fps)
            keep = idx <= max_idx
            report.note_close_idx(ep.name, idx, mode, keep)
            if not keep:
                # A LOUD per-window rejection with the measured number, not a
                # silent skip: this is the property that decided whether the one
                # working checkpoint worked, so a dropped window has to say why.
                report.reject(ep.name,
                              f"grasp @ {t_close:.3f} closes at frame {idx} > "
                              f"{max_idx} ({close_idx_frac:g} x {chunk_frames}-frame "
                              f"chunk, start={mode}) — one chunk cannot reach it")
                continue
            windows.append((lo, hi))
            window_meta.append({"t_close": t_close, "close_idx": idx, "start": mode})
        if not windows:
            report.reject(ep.name, "no grasp window passed the close-index gate")
            return None
    else:
        windows = grasp_windows(grasps, t0, t1, pre_s, post_s)
        if not windows:
            report.reject(ep.name, "no grasp window inside the recorded span")
            return None

    ann = load_json(annotations_path(ep, arms[0])) or {}
    default_task = DEFAULT_TASK if len(arms) == 1 else DEFAULT_BIMANUAL_TASK
    task = (ann.get("episode_meta") or {}).get("instruction") or default_task

    plan = {"ep": ep, "windows": windows, "task": task, "cams": cams,
            "arms": arms, "streams": streams, "window_meta": window_meta}
    # Back-compat keys for the single-arm callers and tests that read the plan.
    t_yam, state, t_gel, action = streams[arms[0]]
    plan.update({"t_yam": t_yam, "state": state, "t_gel": t_gel, "action": action})
    return plan


def window_rows(plan: dict, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample every arm onto ``grid`` and concatenate → (state, action, activity).

    Each arm is resampled on ITS OWN timeline. The two arms are separate nodes in
    separate subprocesses publishing at their own rates; zipping them by index
    would drift silently.

    HOW THE IDLE ARM IS REPRESENTED -- the modelling decision, stated once here.

    The operator cannot teleop both arms at once, so a bimanual take is one
    continuous episode in which exactly one arm is being driven at a time. The
    idle arm is represented by its OWN RECORDED VALUES, unchanged: state = where
    it actually was, action = what its (parked) leader was actually commanding.
    It is not masked, not zeroed, not given a separate action space.

    Why not masked: ACT emits a whole action chunk per step and the runtime
    executes it. A masked dimension has no value at inference time, so a masked
    export forces a second, hand-written "hold" controller to invent one -- and
    the moment that controller and the policy disagree about which arm is idle,
    an un-modelled arm moves. Keeping the hold IN the action space makes the
    policy's output directly executable and makes "stay still" a thing the
    policy is explicitly supervised to emit.

    Why not a separate action space per arm: two policies cannot learn the
    handoff, and the handoff (when is the mat ready for the left arm?) is the
    only genuinely bimanual thing in this task. Splitting it deletes the reason
    to record bimanually at all.

    Why not "hold constant at the last commanded value": that is what the data
    already contains, and fabricating it would hide the one failure this
    representation has -- a leader parked away from its follower. See
    arm_activity() and IDLE_ARM_DIVERGENCE_MAX_RAD.

    NOT included: any "which arm is active" flag. It would be a free lunch at
    training time and undefined at inference -- at run time nothing knows whose
    turn it is; that is precisely what the policy has to infer from the images.
    Activity is returned here as REPORT metadata, never as a feature.
    """
    states, actions, activity = [], [], {}
    for arm in plan["arms"]:
        t_s, s, t_a, a = plan["streams"][arm]
        s_rows = s[nearest_index(t_s, grid)]
        a_rows = a[nearest_index(t_a, grid)]
        states.append(s_rows)
        actions.append(a_rows)
        activity[arm] = arm_activity(s_rows, a_rows)
    return np.hstack(states), np.hstack(actions), activity


def idle_arm_veto(activity: dict, max_divergence: float) -> str | None:
    """Reason to drop this window because a parked arm was commanded elsewhere."""
    if max_divergence <= 0:
        return None
    for arm, act in activity.items():
        if not act["moving"] and act["divergence_rad"] > max_divergence:
            return (f"idle arm {arm}: leader parked {act['divergence_rad']:.3f} rad "
                    f"from the follower (> {max_divergence}) -- training on this "
                    "teaches the policy to jump it")
    return None


def export(root: Path, repo_id: str, out: Path | None, fps: int,
           pre_s: float, post_s: float, dry_run: bool,
           x_min: float | None = None, y_max: float | None = None,
           cameras: dict | None = None,
           arms: tuple[str, ...] | None = None,
           window_mode: str = "grasp",
           max_idle_divergence: float = IDLE_ARM_DIVERGENCE_MAX_RAD,
           open_ref: float | None = None,
           closed_ref: float | None = None,
           urdf_path: str = DEFAULT_URDF,
           chunk_frames: int = LEFT_CHUNK_FRAMES,
           close_idx_frac: float = CLOSE_IDX_CHUNK_FRAC) -> Report:
    cameras = resolve_cameras(cameras)
    arms = resolve_arms(arms)
    report = Report()
    eps = episode_dirs(root, arms)
    if not eps:
        print(f"no episodes under {root}", file=sys.stderr)
        return report

    plans = []
    for ep in eps:
        plan = plan_episode(ep, pre_s, post_s, fps, report, x_min, y_max, cameras,
                            arms, window_mode, open_ref, closed_ref,
                            urdf_path, chunk_frames, close_idx_frac)
        if plan:
            plans.append(plan)
            report.kept.append(ep.name)
            report.windows += len(plan["windows"])

    if dry_run or not plans:
        return report

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shapes = probe_shapes(plans[0], cameras)
    if shapes is None:
        print("could not decode a probe frame", file=sys.stderr)
        return report

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps,
                               features=build_features(shapes, cameras, arms),
                               root=str(out) if out else None,
                               robot_type="_".join(("yam",) + arms), use_videos=True)

    for plan in plans:
        # A camera re-plugged at a different resolution mid-corpus would other-
        # wise blow up add_frame() partway through a long export. Check once per
        # episode and drop that episode with a reason instead.
        got = probe_shapes(plan, cameras)
        if got != shapes:
            report.reject(plan["ep"].name,
                          f"camera resolution {got} != dataset schema {shapes}")
            continue

        # An episode with no target_cycles.json inside a --targets export would be
        # written with target_valid = -1 on every frame -- indistinguishable, to the
        # model and to every metric downstream, from a deliberately masked window. That
        # is the one failure mode that silently costs a whole training run, so it is a
        # rejection with a reason, mirroring load_windows' "episodes absent from the
        # list are REJECTED loudly" rule rather than a warning nobody reads.
        if TARGETS is not None and plan["ep"].name not in TARGETS:
            if TARGETS_ABSENT != "mask":
                report.reject(plan["ep"].name,
                              "no target_cycles.json -- refusing to export it "
                              "unconditioned into a conditioned dataset")
                continue
            # --targets-absent mask: DELIBERATELY export this episode 10-D with
            # target_valid = -1 everywhere. The only legitimate use is widening an OLD
            # corpus (recorded before the recording loop existed) so it can be merged
            # with a conditioned one -- lerobot-edit-dataset merge requires identical
            # features, so a 7-D and a 10-D dataset cannot be combined at all.
            #
            # This is exactly the state the loud rejection exists to prevent, so it is
            # opt-in, named, and printed per episode. If it is ever used on a day that
            # WAS recorded with the loop, that day's targets are silently discarded.
            report.note_masked_episode(plan["ep"].name)

        streams = {c: CameraStream(*plan["cams"][c]) for c in cameras}
        try:
            for wi, (lo, hi) in enumerate(plan["windows"]):
                n = int((hi - lo) * fps)
                if n < MIN_WINDOW_FRAMES:
                    report.reject(plan["ep"].name, f"window only {n} frames")
                    continue
                grid = lo + np.arange(n) / fps
                # per-window task (phase-conditioned exports), else the episode task
                _wm = plan.get("window_meta") or []
                window_task = (_wm[wi].get("task") if wi < len(_wm) and isinstance(_wm[wi], dict) else None) or plan["task"]

                w_state, w_action, activity = window_rows(plan, grid)
                if TARGETS is not None:
                    # JOIN BY TIME CONTAINMENT, never by index: target_cycles.json's `k`
                    # and the window list's ordering come from different processes and
                    # are not guaranteed to agree, so a positional join would mislabel
                    # whole windows silently. grid is absolute epoch seconds
                    # (load_windows returns absolute seconds), the same clock
                    # target_cycles.json stamps t_start/t_end in.
                    trows = TARGETS.get(plan["ep"].name) or []
                    tcol = np.array([_target_windows.lookup(trows, float(t))
                                     for t in grid], dtype=np.float32)
                    w_state = np.concatenate(
                        [np.asarray(w_state, dtype=np.float32), tcol], axis=1)
                veto = idle_arm_veto(activity, max_idle_divergence)
                if veto:
                    report.reject(plan["ep"].name, veto)
                    continue
                report.note_activity(plan["ep"].name, activity)

                ci = {c: nearest_index(streams[c].t, grid) for c in cameras}
                # The reader is forward-only. Windows are sorted, but nearest-
                # index rounding at a shared boundary can still ask for the
                # frame before the last one served; reopen rather than abort
                # a 40-minute export on a one-frame step back.
                for c in cameras:
                    if int(ci[c][0]) < streams[c]._pos:
                        streams[c].close()
                        streams[c] = CameraStream(*plan["cams"][c])
                stale = {c: np.abs(streams[c].t[ci[c]] - grid) > MAX_CAM_STALENESS_S
                         for c in cameras}

                wrote = 0
                for k in range(n):
                    if any(stale[c][k] for c in cameras):
                        continue                      # no honest image for this instant
                    frame = {
                        "observation.state": w_state[k],
                        "action": w_action[k],
                        "task": window_task,
                    }
                    bad = False
                    for cam, suffix in cameras.items():
                        img = streams[cam].frame_at(int(ci[cam][k]))
                        if img is None:
                            bad = True
                            break
                        frame[f"observation.images.{suffix}"] = img
                    if bad:
                        break
                    ds.add_frame(frame)
                    wrote += 1

                if wrote >= MIN_WINDOW_FRAMES:
                    ds.save_episode()
                    report.frames += wrote
                else:
                    # Drop the partial buffer rather than saving a stub episode.
                    ds.episode_buffer = None
                    report.reject(plan["ep"].name, f"only {wrote} usable frames")
        finally:
            for s in streams.values():
                s.close()

    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="recordings")
    ap.add_argument("--repo-id", default="ETHRC/yam_grasp_v1")
    ap.add_argument("--out", default=None, help="dataset root (default: HF cache)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--pre-s", type=float, default=DEFAULT_PRE_S)
    ap.add_argument("--post-s", type=float, default=DEFAULT_POST_S)
    ap.add_argument("--zone-x-min", type=float, default=None,
                    help=f"drop grasps with x < this (default {C.GRASP_WORKSPACE_X_MIN})")
    ap.add_argument("--zone-y-max", type=float, default=None,
                    help=f"drop grasps with y > this (default {C.GRASP_ZONE_Y_MAX}, "
                         "i.e. no lateral gate). Set it from tools/review_grasps.py.")
    ap.add_argument("--cameras", choices=sorted(CAMERA_SETS), default="both",
                    help="which cameras become observation.images.*  "
                         "('wrist' = gripper camera only, joints still included)")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="left",
                    help="which physical arms the dataset covers. 'both' "
                         "concatenates left then right into a 14-DoF state and "
                         "action (default: left, which reproduces every dataset "
                         "exported before 2026-08-08)")
    ap.add_argument("--window-mode", choices=("grasp", "full", "grasp-pose"),
                    default="grasp",
                    help="grasp = one training episode per successful grasp "
                         "(default); full = one training episode per recorded "
                         "take, which is what a bimanual handoff needs; "
                         "grasp-pose = grasp windows that OPEN at the start of the "
                         "measured final descent instead of a fixed --pre-s, and "
                         "that must close within --close-idx-frac of the chunk")
    ap.add_argument("--urdf", default=DEFAULT_URDF,
                    help="URDF for the pose predicate's FK (grasp-pose mode only). "
                         "Explicit because fk.py's own default is cwd-relative.")
    ap.add_argument("--chunk-frames", type=int, default=LEFT_CHUNK_FRAMES,
                    help=f"the TRAINING chunk_size in frames of the {DEFAULT_FPS} Hz "
                         f"grid (default {LEFT_CHUNK_FRAMES}, the left policy's). "
                         "NOT n_action_steps: the runtime re-queries every 16 steps, "
                         "so chunk_size is what bounds a single plan.")
    ap.add_argument("--close-idx-frac", type=float, default=CLOSE_IDX_CHUNK_FRAC,
                    help=f"grasp-pose gate: drop a window whose jaw close lands past "
                         f"this fraction of the chunk (default {CLOSE_IDX_CHUNK_FRAC}). "
                         "A starting point, not a measured threshold — the full "
                         "distribution is printed either way.")
    ap.add_argument("--max-idle-divergence", type=float,
                    default=IDLE_ARM_DIVERGENCE_MAX_RAD,
                    help="drop a window if an arm that never moved was commanded "
                         "this far (rad) from where it actually was; 0 disables")
    ap.add_argument("--keep", default=None,
                    help="JSON keep-list from tools/review_grasps.py: only the grasps "
                         "it names are exported. An entry that matches no grasp is a "
                         "hard error, not a silent skip.")
    ap.add_argument("--gripper-open-ref", type=float, default=None,
                    help="gripper's PHYSICAL open value (this rig: 1.0). Given "
                         "together with --gripper-closed-ref the channel becomes "
                         "absolute and comparable across episodes; omitted, each "
                         "episode is scaled by its own percentiles, which is only "
                         "valid when it spans the full open->closed range.")
    ap.add_argument("--gripper-closed-ref", type=float, default=None,
                    help="gripper's PHYSICAL closed value (this rig: 0.0)")
    ap.add_argument("--pregrasp-margin-s", type=float, default=None,
                    help="grasp-pose only: seconds of lead kept before the descent-"
                         f"plane crossing (default {PREGRASP_MARGIN_FRAMES / DEFAULT_FPS:g}). "
                         "Raising it opens every window earlier — raise --chunk-frames "
                         "with it so the close stays inside the chunk, and --pre-s so "
                         "the FK search span covers the longer lead.")
    ap.add_argument("--pregrasp-plane-m", type=float, default=None,
                    help="grasp-pose only: metres above the grasp's own close height "
                         f"for the descent-start plane (default {PREGRASP_PLANE_M:g}). "
                         "Raising it starts windows nearer the top of the approach; "
                         "a window with no crossing falls back to the clamp, loudly.")
    ap.add_argument("--windows", default=None,
                    help="full mode only: JSON from tools/segment_home_episodes.py "
                         "with {'windows': {episode: [[t0, t1], ...]}} (absolute "
                         "seconds). Each window becomes one training episode; an "
                         "episode missing from the file is rejected, not exported whole.")
    ap.add_argument("--targets", default=None,
                    help="recordings root holding <episode>/target_cycles.json. Appends "
                         "[target_u, target_v, target_valid] to observation.state "
                         "(state width becomes dof+3). An episode being exported that "
                         "has no target_cycles.json is REJECTED loudly, never exported "
                         "unconditioned into a conditioned dataset.")
    ap.add_argument("--target-mask", type=float, default=0.0,
                    help="fraction of WINDOWS exported blind (target_valid=-1, u=v=0). "
                         "0.5 is the validated instance (NoMaD). Masked per window, "
                         "never per frame: the goal is constant over a cycle by "
                         "construction and per-frame flicker is a train/deploy mismatch.")
    ap.add_argument("--targets-absent", choices=("reject", "mask"), default="reject",
                    help="what to do with an episode that has no target_cycles.json "
                         "inside a --targets export. 'reject' (default) drops it loudly. "
                         "'mask' exports it 10-D with target_valid=-1 everywhere -- ONLY "
                         "for widening an old corpus so it can be merged with a "
                         "conditioned one (merge requires identical features).")
    ap.add_argument("--target-seed", type=int, default=0,
                    help="seed for --target-mask; the same seed reproduces the same "
                         "blind windows exactly. Recorded in the export manifest.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be exported, write nothing")
    a = ap.parse_args(argv)
    if a.targets:
        global TARGETS, TARGET_MANIFEST
        doc = _target_windows.load_targets(a.targets, p_mask=a.target_mask,
                                           seed=a.target_seed)
        TARGETS, TARGET_MANIFEST = doc["targets"], doc["manifest"]
        global TARGETS_ABSENT
        TARGETS_ABSENT = a.targets_absent
        m = TARGET_MANIFEST
        print(f"targets         : {m['supplied']} conditioned / {m['cycles']} cycles "
              f"across {len(TARGETS)} episodes from {a.targets}")
        print(f"                  masked {m['masked']} (p={a.target_mask}, "
              f"seed={a.target_seed}), no moved_n {m['no_moved_n']}, "
              f"no frame {m['no_frame']}, bad {m['bad']}")
        if not m["supplied"]:
            ap.error("--targets: not one cycle carries a usable target. Exporting now "
                     "would write a dataset whose target_valid is -1 everywhere -- a "
                     "dead channel that trains and evaluates without ever complaining. "
                     "Run training_experiment/tools/target_corpus_audit.py first.")
    if a.windows:
        if a.window_mode != "full":
            ap.error("--windows only applies to --window-mode full")
        global WINDOWS
        WINDOWS = load_windows(a.windows)
        print(f"windows         : {sum(len(v) for v in WINDOWS.values())} windows "
              f"across {len(WINDOWS)} episodes from {a.windows}")
    global PREGRASP_MARGIN_S_CLI, PREGRASP_PLANE_M_CLI
    PREGRASP_MARGIN_S_CLI = a.pregrasp_margin_s
    PREGRASP_PLANE_M_CLI = a.pregrasp_plane_m
    if a.keep:
        global KEEP
        KEEP = load_keep_list(a.keep)
        print(f"keep-list: {sum(len(v) for v in KEEP.values())} grasps "
              f"across {len(KEEP)} episodes")

    xm = C.GRASP_WORKSPACE_X_MIN if a.zone_x_min is None else a.zone_x_min
    ym = C.GRASP_ZONE_Y_MAX if a.zone_y_max is None else a.zone_y_max
    cams = CAMERA_SETS[a.cameras]
    arms = ARM_SETS[a.arms]
    print(f"zone            : x >= {xm}" + ("" if ym is None else f", y <= {ym}"))
    print(f"cameras         : {a.cameras}  -> "
          + ", ".join(f"observation.images.{s}" for s in cams.values()))
    print(f"arms            : {a.arms}  -> {len(joint_names(arms))}-DoF "
          f"state/action, windows={a.window_mode}")

    if (a.gripper_open_ref is None) != (a.gripper_closed_ref is None):
        ap.error("--gripper-open-ref and --gripper-closed-ref go together: one "
                 "alone would silently fall back to the percentile scale")
    if a.gripper_open_ref is not None:
        print(f"gripper         : absolute, open={a.gripper_open_ref} "
              f"closed={a.gripper_closed_ref}")
    else:
        print("gripper         : per-episode percentile scale (no refs given)")

    if a.window_mode == "grasp-pose":
        print(f"close-index gate: <= {close_idx_gate(a.chunk_frames, a.close_idx_frac)} "
              f"of {a.chunk_frames} frames ({a.close_idx_frac:g} x chunk), "
              f"pre-grasp plane +{(PREGRASP_PLANE_M if PREGRASP_PLANE_M_CLI is None else PREGRASP_PLANE_M_CLI) * 1000:.0f} mm, "
              f"margin {(PREGRASP_MARGIN_FRAMES / DEFAULT_FPS if PREGRASP_MARGIN_S_CLI is None else PREGRASP_MARGIN_S_CLI) * DEFAULT_FPS:.0f} frames")

    rep = export(Path(a.root), a.repo_id, Path(a.out) if a.out else None,
                 a.fps, a.pre_s, a.post_s, a.dry_run, a.zone_x_min, a.zone_y_max,
                 cams, arms, a.window_mode, a.max_idle_divergence,
                 a.gripper_open_ref, a.gripper_closed_ref,
                 a.urdf, a.chunk_frames, a.close_idx_frac)

    print(f"\nepisodes kept   : {len(rep.kept)}")
    print(f"grasp windows   : {rep.windows}")
    if not a.dry_run:
        print(f"frames written  : {rep.frames}")
    print(f"episodes dropped: {len(rep.rejected)}")
    # Never silently drop data: every rejection is printed with its reason, so a
    # corpus that exports 3 of 100 episodes is obvious instead of looking fine.
    for r in rep.rejected:
        print(f"   {r.episode:<34} {r.reason}")
    # The close-index distribution — the decisive property, reported whether or
    # not anything failed the gate (check_handover_pose.py:17-31).
    if rep.close_idx:
        idxs = np.array([i for _, i, _, _ in rep.close_idx])
        kept = np.array([k for _, _, _, k in rep.close_idx])
        modes = {}
        for _, _, m, _ in rep.close_idx:
            modes[m] = modes.get(m, 0) + 1
        print(f"\nclose frame index over {len(idxs)} candidate windows: "
              f"min {idxs.min()}  median {int(np.median(idxs))}  "
              f"p90 {int(np.percentile(idxs, 90))}  max {idxs.max()}")
        print("  window starts   : "
              + "  ".join(f"{m}={n}" for m, n in sorted(modes.items())))
        print(f"  passed the gate : {int(kept.sum())} / {len(idxs)}")

    # Which arm was actually driven in each written window. Nothing gates on it,
    # but a bimanual corpus where one arm never moves in ANY window is a
    # recording mistake worth seeing before a training run, not after.
    if len(arms) > 1 and rep.activity:
        print("\nper-window arm activity (ptp rad / leader-follower divergence rad):")
        for ep_name, act in rep.activity:
            summary = "  ".join(
                f"{arm}={'MOVED' if v['moving'] else 'idle '} "
                f"{v['ptp_rad']:.3f}/{v['divergence_rad']:.3f}"
                for arm, v in act.items())
            print(f"   {ep_name:<34} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

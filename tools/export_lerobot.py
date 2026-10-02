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
    feats = {
        "observation.state": {"dtype": "float32", "shape": dof, "names": names},
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


def read_arm(ep: Path, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t_state, state, t_action, action) for one arm, gripper channel normalised.

    action = gello (leader, what the operator commanded), observation.state =
    yam (follower, where the arm actually was). Swapping them trains a policy to
    predict where the arm already is: the loss curve looks fine and the policy
    is useless on hardware. Pinned by test.
    """
    t_yam, p_yam = read_positions(ep / f"yam_{arm}.mcap", f"yam_{arm}")
    t_gel, p_gel = read_positions(ep / f"gello_{arm}.mcap", f"gello_{arm}")
    if t_yam.size == 0 or t_gel.size == 0:
        raise RuntimeError(f"empty joint stream for {arm}")
    # Gripper scale over the WHOLE episode so all its windows share one scale.
    state = p_yam.astype(np.float32).copy()
    action = p_gel.astype(np.float32).copy()
    state[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(state[:, C.GRIPPER_JOINT_INDEX])
    action[:, C.GRIPPER_JOINT_INDEX] = normalize_gripper(action[:, C.GRIPPER_JOINT_INDEX])
    return t_yam, state, t_gel, action


def plan_episode(ep: Path, pre_s: float, post_s: float, fps: int, report: Report,
                 x_min: float | None = None, y_max: float | None = None,
                 cameras: dict | None = None,
                 arms: tuple[str, ...] | None = None,
                 window_mode: str = "grasp"):
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
    """
    arms = resolve_arms(arms)

    grasps: list[dict] = []
    for arm in arms:
        g, why = usable_grasps(ep, x_min=x_min, y_max=y_max, arm=arm)
        if why:
            if window_mode == "grasp":
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
            streams[arm] = read_arm(ep, arm)
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

    if window_mode == "full":
        windows = [(t0, t1)] if t1 > t0 else []
        if not windows:
            report.reject(ep.name, "arms' recorded spans do not overlap")
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
            "arms": arms, "streams": streams}
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
           max_idle_divergence: float = IDLE_ARM_DIVERGENCE_MAX_RAD) -> Report:
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
                            arms, window_mode)
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

        streams = {c: CameraStream(*plan["cams"][c]) for c in cameras}
        try:
            for (lo, hi) in plan["windows"]:
                n = int((hi - lo) * fps)
                if n < MIN_WINDOW_FRAMES:
                    report.reject(plan["ep"].name, f"window only {n} frames")
                    continue
                grid = lo + np.arange(n) / fps

                w_state, w_action, activity = window_rows(plan, grid)
                veto = idle_arm_veto(activity, max_idle_divergence)
                if veto:
                    report.reject(plan["ep"].name, veto)
                    continue
                report.note_activity(plan["ep"].name, activity)

                ci = {c: nearest_index(streams[c].t, grid) for c in cameras}
                stale = {c: np.abs(streams[c].t[ci[c]] - grid) > MAX_CAM_STALENESS_S
                         for c in cameras}

                wrote = 0
                for k in range(n):
                    if any(stale[c][k] for c in cameras):
                        continue                      # no honest image for this instant
                    frame = {
                        "observation.state": w_state[k],
                        "action": w_action[k],
                        "task": plan["task"],
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
    ap.add_argument("--window-mode", choices=("grasp", "full"), default="grasp",
                    help="grasp = one training episode per successful grasp "
                         "(default); full = one training episode per recorded "
                         "take, which is what a bimanual handoff needs")
    ap.add_argument("--max-idle-divergence", type=float,
                    default=IDLE_ARM_DIVERGENCE_MAX_RAD,
                    help="drop a window if an arm that never moved was commanded "
                         "this far (rad) from where it actually was; 0 disables")
    ap.add_argument("--keep", default=None,
                    help="JSON keep-list from tools/review_grasps.py: only the grasps "
                         "it names are exported. An entry that matches no grasp is a "
                         "hard error, not a silent skip.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be exported, write nothing")
    a = ap.parse_args(argv)
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

    rep = export(Path(a.root), a.repo_id, Path(a.out) if a.out else None,
                 a.fps, a.pre_s, a.post_s, a.dry_run, a.zone_x_min, a.zone_y_max,
                 cams, arms, a.window_mode, a.max_idle_divergence)

    print(f"\nepisodes kept   : {len(rep.kept)}")
    print(f"grasp windows   : {rep.windows}")
    if not a.dry_run:
        print(f"frames written  : {rep.frames}")
    print(f"episodes dropped: {len(rep.rejected)}")
    # Never silently drop data: every rejection is printed with its reason, so a
    # corpus that exports 3 of 100 episodes is obvious instead of looking fine.
    for r in rep.rejected:
        print(f"   {r.episode:<34} {r.reason}")
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

"""Offline episode labeler — the orchestrator.

    label_from_arrays(...)  pure function: joint timeline → Annotations (testable).
    label_episode_dir(dir)  reads MCAP + optional cockpit_events/compartments/kit,
                            writes annotations.json.

Idempotent: same inputs → same annotations.json. Raw recordings are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.fk import ForwardKinematics, tracking_error
from robots_realtime.labeling.fuse import GraspCandidate, build_annotations
from robots_realtime.labeling.mcap_io import read_jsonl, read_positions
from robots_realtime.labeling.placement import Compartment, load_compartments
from robots_realtime.labeling.schema import (
    Annotations,
    Flag,
    KitItem,
    TrackingError,
)
from robots_realtime.labeling.segmentation import (
    GripperRangeUnknown,
    detect_grip_intervals,
    is_unknown,
)


def annotations_path(episode_dir: str | Path, arm: str = "left") -> Path:
    """Where this arm's annotations live inside an episode directory.

    The left arm keeps the historical bare name so every existing episode,
    dataset export and review tool still finds its file. A second arm in the
    same episode gets its own file — one episode directory now holds one
    annotations file PER ARM, because a bimanual take is labelled twice (once
    per arm) and the two label sets must not overwrite each other.
    """
    episode_dir = Path(episode_dir)
    return episode_dir / ("annotations.json" if arm == "left"
                          else f"annotations_{arm}.json")


def _joints_at(times: np.ndarray, joints: np.ndarray, t: float) -> np.ndarray:
    """Arm-joint vector nearest to time t."""
    return joints[int(np.argmin(np.abs(times - t)))]


def label_from_arrays(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    fk: ForwardKinematics,
    arm: str = "left",
    episode_id: str = "episode",
    gripper_open_ref: float | None = None,
    gripper_closed_ref: float | None = None,
    cockpit_events: list[dict] | None = None,
    compartments: list[Compartment] | None = None,
    kitting_list: list[KitItem] | None = None,
    commanded_positions: tuple[np.ndarray, np.ndarray] | None = None,
    clock_offset_s: float = 0.0,
    min_transport_m: float = 0.0,
    geometric_targets: bool = False,
) -> Annotations:
    times = np.asarray(times, float)
    positions = np.asarray(positions, float)
    arm_joints = positions[:, : C.N_ARM_JOINTS]
    gripper = positions[:, C.GRIPPER_JOINT_INDEX]

    ee_pos = fk.ee_positions(arm_joints)
    ee_z = ee_pos[:, 2]

    # A dead / rangeless gripper channel is not "an episode with no grasps" — it
    # is an episode we cannot label at all. Before 2026-08-08 normalize_width
    # returned all-zeros here, which reads as "jaws fully shut for the whole
    # take": the most confident possible value, produced from no information.
    gripper_unknown: str | None = None
    try:
        intervals = detect_grip_intervals(
            times, gripper, ee_z=ee_z,
            open_ref=gripper_open_ref, closed_ref=gripper_closed_ref,
        )
    except GripperRangeUnknown as e:
        gripper_unknown = str(e)
        intervals = []

    candidates: list[GraspCandidate] = []
    for iv in intervals:
        grasp_pose = fk.ee_pose(_joints_at(times, arm_joints, iv.t_close))
        release_pose = (fk.ee_pose(_joints_at(times, arm_joints, iv.t_open))
                        if iv.t_open is not None else None)
        candidates.append(GraspCandidate(
            iv.t_close, iv.t_open, iv.outcome, iv.lifted, grasp_pose, release_pose))

    t_start = float(times[0]) if times.size else 0.0
    t_end = float(times[-1]) if times.size else 0.0
    outcome = "success" if candidates else "aborted"

    ann = build_annotations(
        episode_id, arm, t_start, t_end, candidates,
        kitting_list=kitting_list, cockpit_events=cockpit_events,
        compartments=compartments, clock_offset_s=clock_offset_s, outcome=outcome,
        min_transport_m=min_transport_m, geometric_targets=geometric_targets,
    )

    # The guard that could not fire (AUDIT.md S1.4). It used to read
    #     if normalize_width(gripper).min() > 0.15 and candidates:
    # — but on a never-moved gripper normalize_width returned all-zeros, so
    # .min() was 0.0, so `0.0 > 0.15` was False and no flag was raised. The one
    # input it existed to catch was the one input it certified as healthy. It
    # also required `candidates`, and a dead channel produces none, so it was
    # doubly unreachable. Both conditions are gone: the unknown case is now its
    # own branch and is flagged whether or not anything was detected.
    if gripper_unknown is not None:
        ann.flags.append(Flag("gripper_range_unknown", gripper_unknown))
    elif gripper_closed_ref is None and candidates:
        from robots_realtime.labeling.segmentation import normalize_width
        norm = normalize_width(gripper, on_degenerate="unknown")
        if not is_unknown(norm) and float(np.nanmin(norm)) > 0.15:
            ann.flags.append(Flag(
                "gripper_range_unknown",
                "gripper never near full close and no known limits; empty-grasp "
                "detection may be unreliable — pass gripper limits",
            ))

    # Commanded-vs-achieved tracking error over each transport segment.
    if commanded_positions is not None:
        c_times, c_pos = commanded_positions
        c_times = np.asarray(c_times, float)
        c_arm = np.asarray(c_pos, float)[:, : C.N_ARM_JOINTS]
        for seg in [s for s in ann.segments if s.phase == "transport"]:
            m = (times >= seg.t_start) & (times <= seg.t_end)
            cm = (c_times >= seg.t_start) & (c_times <= seg.t_end)
            if m.sum() < 2 or cm.sum() < 2:
                continue
            achieved = ee_pos[m]
            commanded = fk.ee_positions(c_arm[cm])
            rms, mx = tracking_error(commanded, achieved)
            ann.tracking.append(TrackingError("transport", arm, rms, mx))

    return ann


def _load_kit(episode_dir: Path) -> list[KitItem] | None:
    """kit.json = [{bag_id, part_no, name, compartment}, ...] if the operator/cockpit saved one."""
    p = episode_dir / "kit.json"
    if not p.exists():
        return None
    return [KitItem(**{k: item.get(k) for k in ("bag_id", "part_no", "name", "compartment")})
            for item in json.loads(p.read_text())]


def label_episode_dir(episode_dir: str | Path, arm: str = "left",
                      urdf_path: str | Path = "urdf/yam.urdf",
                      gripper_open_ref: float | None = None,
                      gripper_closed_ref: float | None = None,
                      min_transport_m: float = 0.0,
                      geometric_targets: bool = False,
                      write: bool = True) -> Annotations:
    episode_dir = Path(episode_dir)
    fk = ForwardKinematics(urdf_path)

    mcap = episode_dir / f"yam_{arm}.mcap"
    if not mcap.exists():
        raise RuntimeError(f"no yam_{arm}.mcap in {episode_dir.name} — incomplete episode, skipped")
    times, positions = read_positions(mcap, f"yam_{arm}")
    if times.size == 0:
        raise RuntimeError(f"no joint data in {episode_dir.name}/yam_{arm}.mcap")

    commanded = None
    gello = episode_dir / f"gello_{arm}.mcap"
    if gello.exists():
        ct, cp = read_positions(gello, f"gello_{arm}")
        if ct.size:
            commanded = (ct, cp)

    comps = None
    comps_path = episode_dir / "compartments.json"
    if comps_path.exists():
        comps = load_compartments(comps_path)

    ann = label_from_arrays(
        times, positions, fk=fk, arm=arm, episode_id=episode_dir.name,
        gripper_open_ref=gripper_open_ref, gripper_closed_ref=gripper_closed_ref,
        cockpit_events=read_jsonl(episode_dir / "cockpit_events.jsonl") or None,
        compartments=comps, kitting_list=_load_kit(episode_dir),
        commanded_positions=commanded, min_transport_m=min_transport_m,
        geometric_targets=geometric_targets,
    )
    if write:
        ann.save(annotations_path(episode_dir, arm))
    return ann


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Label one recorded kitting episode.")
    ap.add_argument("episode_dir")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--open-ref", type=float, default=None, help="gripper open joint value")
    ap.add_argument("--closed-ref", type=float, default=None, help="gripper closed joint value")
    ap.add_argument("--min-transport", type=float, default=0.0,
                    help="min EE XY travel (m) grasp→release to count as a placement "
                         "(kitting: ~0.10; 0 disables — drops 'released-at-pick' false placements)")
    ap.add_argument("--geometric-targets", action="store_true",
                    help="reassign each place target to the nearest distinct compartment by "
                         "geometry (use when the operator picks out of kit order)")
    args = ap.parse_args(argv)
    try:
        ann = label_episode_dir(args.episode_dir, arm=args.arm,
                                gripper_open_ref=args.open_ref, gripper_closed_ref=args.closed_ref,
                                min_transport_m=args.min_transport,
                                geometric_targets=args.geometric_targets)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"⏭  skipped {Path(args.episode_dir).name}: {e}")  # clean skip, not a traceback
        return 0
    print(f"wrote {annotations_path(args.episode_dir, args.arm)}")
    print(f"  bags placed: {len(ann.place_events)}  grasp attempts: {len(ann.grasp_attempts)}"
          f"  flags: {len(ann.flags)}")
    for f in ann.flags:
        print(f"  ⚠ {f.kind}: {f.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

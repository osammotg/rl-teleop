"""Process model — aggregate many labeled episodes into learned priors.

The "intelligence layer": given a corpus of annotations.json, roll up
  * grasp-pose statistics per part number (how each part is grasped),
  * drop-position distributions per compartment (where each compartment is
    released into) and how often the release lands in-region,
  * per-part grasp success rate (successful vs slip/drop/empty attempts).

Cross-episode only; needs a corpus to be meaningful. Consumed by the planner /
cockpit and as a prior for training.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from robots_realtime.labeling.schema import Annotations, load_annotations


@dataclass
class GraspStats:
    part_no: str
    n: int
    mean_pose: list           # [x,y,z,qw,qx,qy,qz]
    pos_std: list             # [sx,sy,sz]
    success_rate: float       # successful attempts / total attempts


@dataclass
class DropStats:
    compartment: int
    n: int
    mean_xy: list             # [x, y]
    xy_std: list              # [sx, sy]
    mean_release_height: float | None
    in_region_rate: float | None


@dataclass
class ProcessModel:
    n_episodes: int = 0
    grasps: dict = field(default_factory=dict)   # part_no -> GraspStats
    drops: dict = field(default_factory=dict)    # str(compartment) -> DropStats

    def to_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "grasps": {k: asdict(v) for k, v in self.grasps.items()},
            "drops": {k: asdict(v) for k, v in self.drops.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _part_of(ann: Annotations, bag_id: int) -> str | None:
    for k in ann.episode_meta.kitting_list:
        if k.bag_id == bag_id:
            return k.part_no
    return None


def aggregate(episodes: list[Annotations]) -> ProcessModel:
    # part_no -> list[pose], part_no -> [n_success, n_total]
    grasp_poses: dict[str, list] = {}
    grasp_outcomes: dict[str, list[int]] = {}
    drop_xy: dict[int, list] = {}
    drop_h: dict[int, list] = {}
    drop_in: dict[int, list[bool]] = {}

    for ann in episodes:
        for g in ann.grasp_attempts:
            part = _part_of(ann, g.bag_id)
            if part is None:
                continue
            grasp_outcomes.setdefault(part, [0, 0])
            grasp_outcomes[part][1] += 1
            if g.outcome == "success":
                grasp_outcomes[part][0] += 1
                if g.ee_pose is not None:
                    grasp_poses.setdefault(part, []).append(g.ee_pose)
        for p in ann.place_events:
            comp = p.target_compartment
            if comp is None or p.achieved_ee_pose is None:
                continue
            drop_xy.setdefault(comp, []).append(p.achieved_ee_pose[:2])
            if p.release_height_m is not None:
                drop_h.setdefault(comp, []).append(p.release_height_m)
            if p.in_target_region is not None:
                drop_in.setdefault(comp, []).append(bool(p.in_target_region))

    grasps = {}
    for part, poses in grasp_poses.items():
        arr = np.asarray(poses, float)
        ns, nt = grasp_outcomes[part]
        grasps[part] = GraspStats(
            part_no=part, n=len(poses),
            mean_pose=arr.mean(axis=0).tolist(),
            pos_std=arr[:, :3].std(axis=0).tolist(),
            success_rate=(ns / nt) if nt else 0.0,
        )
    # parts with only failed grasps (no poses) still get a success_rate entry
    for part, (ns, nt) in grasp_outcomes.items():
        if part not in grasps:
            grasps[part] = GraspStats(part, 0, [], [], (ns / nt) if nt else 0.0)

    drops = {}
    for comp, xys in drop_xy.items():
        arr = np.asarray(xys, float)
        heights = drop_h.get(comp)
        ins = drop_in.get(comp)
        drops[str(comp)] = DropStats(
            compartment=comp, n=len(xys),
            mean_xy=arr.mean(axis=0).tolist(),
            xy_std=arr.std(axis=0).tolist(),
            mean_release_height=(float(np.mean(heights)) if heights else None),
            in_region_rate=(float(np.mean(ins)) if ins else None),
        )

    return ProcessModel(n_episodes=len(episodes), grasps=grasps, drops=drops)


def aggregate_dir(recordings_dir: str | Path) -> ProcessModel:
    """Aggregate every episode subdir that has an annotations.json."""
    recordings_dir = Path(recordings_dir)
    episodes = [load_annotations(d) for d in sorted(recordings_dir.iterdir())
                if d.is_dir() and (d / "annotations.json").exists()]
    return aggregate(episodes)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate labeled episodes into a process model.")
    ap.add_argument("recordings_dir")
    ap.add_argument("--out", default=None, help="write process_model.json here")
    args = ap.parse_args(argv)
    m = aggregate_dir(args.recordings_dir)
    out = Path(args.out) if args.out else Path(args.recordings_dir) / "process_model.json"
    m.save(out)
    print(f"aggregated {m.n_episodes} episodes → {out}")
    print(f"  parts: {len(m.grasps)}  compartments: {len(m.drops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

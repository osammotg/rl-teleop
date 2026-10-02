"""Placement correctness: where the bag was released vs the target compartment.

The box is fixed in the robot base frame (calibrated once), so each compartment
is an axis-aligned rectangle in base XY. At release we take the end-effector XY
and ask: which compartment is it over, is that the target, and how far from the
compartment center did it land.

Calibration format (compartments.json), robot base frame, metres:

    {"frame": "robot_base",
     "compartments": [
        {"id": 1, "x_min": .., "x_max": .., "y_min": .., "y_max": ..},
        ... 7 entries ...]}

Measuring these 7 rects is a one-time prerequisite (see the plan). The logic here
is exercised by tests with synthetic regions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from robots_realtime.labeling import constants as C


@dataclass
class Compartment:
    id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x_min - margin <= x <= self.x_max + margin
                and self.y_min - margin <= y <= self.y_max + margin)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))


@dataclass
class PlacementResult:
    detected_compartment: int | None    # which compartment the release is over
    in_target_region: bool | None       # detected == target (None if no target)
    xy_offset_m: float | None           # release XY → target compartment center


def load_compartments(path: str | Path) -> list[Compartment]:
    data = json.loads(Path(path).read_text())
    return [Compartment(**{k: c[k] for k in ("id", "x_min", "x_max", "y_min", "y_max")})
            for c in data["compartments"]]


def compartment_at(x: float, y: float, compartments: list[Compartment],
                   margin: float = C.IN_REGION_MARGIN_M) -> int | None:
    """The compartment whose (margin-expanded) rect contains (x, y), or None.

    If the point falls in more than one expanded rect (margins overlap), pick the
    one whose center is nearest, so a bag on a boundary resolves deterministically.
    """
    hits = [c for c in compartments if c.contains(x, y, margin)]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0].id
    return min(hits, key=lambda c: (c.center[0] - x) ** 2 + (c.center[1] - y) ** 2).id


def assign_targets_geometric(place_events, compartments: list[Compartment]) -> int:
    """Reassign each placement's target to a DISTINCT compartment by minimizing
    total release→center XY distance (optimal assignment), then HONESTLY recompute
    detected/in_target_region/xy_offset against the new target via classify_release.

    Use when the kit-order target is unreliable (operator picks out of kit order):
    the demonstrated label becomes 'which compartment this bag was actually placed
    in', recovered from geometry instead of pick order. One bag per compartment
    (kitting invariant) → assignment, not nearest-independently. Mutates the given
    PlaceEvent objects in place. Returns the number reassigned.
    """
    pts = [p for p in place_events if p.achieved_ee_pose is not None]
    comps = list(compartments or [])
    if not pts or not comps:
        return 0
    cost = [[(p.achieved_ee_pose[0] - c.center[0]) ** 2 + (p.achieved_ee_pose[1] - c.center[1]) ** 2
             for c in comps] for p in pts]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(np.asarray(cost, dtype=float))
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        # greedy fallback: take the cheapest (placement, compartment) pairs first,
        # each placement and each compartment used at most once.
        triples = sorted((cost[i][j], i, j) for i in range(len(pts)) for j in range(len(comps)))
        used_r, used_c, pairs = set(), set(), []
        for _, i, j in triples:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j); pairs.append((i, j))
    for i, j in pairs:
        p, c = pts[i], comps[j]
        p.target_compartment = c.id
        res = classify_release((p.achieved_ee_pose[0], p.achieved_ee_pose[1]), c.id, comps)
        p.detected_compartment = res.detected_compartment
        p.in_target_region = res.in_target_region
        p.xy_offset_m = res.xy_offset_m
    return len(pairs)


def classify_release(ee_xy: tuple[float, float], target_compartment: int | None,
                     compartments: list[Compartment],
                     max_reach_m: float = 0.15) -> PlacementResult:
    x, y = ee_xy
    detected = compartment_at(x, y, compartments)
    if detected is None and compartments:
        # A deliberate placement can land just outside the calibrated rect. Fall back to
        # the NEAREST compartment center (Voronoi) so real demonstrations aren't dropped;
        # only leave it None if the release is implausibly far from any compartment.
        nearest = min(compartments, key=lambda c: ((x - c.center[0]) ** 2 + (y - c.center[1]) ** 2))
        if (x - nearest.center[0]) ** 2 + (y - nearest.center[1]) ** 2 <= max_reach_m ** 2:
            detected = nearest.id
    in_region: bool | None = None
    offset: float | None = None
    if target_compartment is not None:
        in_region = detected == target_compartment
        tgt = next((c for c in compartments if c.id == target_compartment), None)
        if tgt is not None:
            cx, cy = tgt.center
            offset = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return PlacementResult(detected, in_region, offset)

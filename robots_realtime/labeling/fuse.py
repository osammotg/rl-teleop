"""Fuse gripper intervals + FK poses + cockpit intent into per-bag annotations.

Single-arm kitting is sequential: one bag in flight at a time. So grasp intervals
map to bags in time order. Within a bag, a slip/drop/empty interval is a failed
attempt and the next interval that succeeds is the re-grasp that finally places it.

Cockpit events (optional) supply intent — which bag, part, and target compartment —
and are matched to the placing grasp by time. Without them, bags are numbered
sequentially and compartments come from the kitting list if present.

Guards (all produce a loud Flag, never a silent drop):
  * clock_out_of_window — a cockpit event outside the episode window (clock desync)
  * overlapping_grasp   — two intervals overlap in time (invariant violation)
  * ocr_null            — a kit item whose part could not be read
  * unplaced_grasp      — a grasp that never resolved to a place
"""
from __future__ import annotations

from dataclasses import dataclass

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.placement import (
    Compartment,
    assign_targets_geometric,
    classify_release,
)
from robots_realtime.labeling.schema import (
    Annotations,
    EpisodeMeta,
    Flag,
    GraspAttempt,
    KitItem,
    PlaceEvent,
    Segment,
)
from robots_realtime.labeling.segmentation import transported


@dataclass
class GraspCandidate:
    """A gripper interval enriched with FK poses (built by label_episode)."""
    t_close: float
    t_open: float | None
    outcome: str                 # success | slip | drop | empty
    lifted: bool | None
    grasp_pose: list | None = None    # EE pose at t_close
    release_pose: list | None = None  # EE pose at t_open


def _in_window(t: float, t_start: float, t_end: float) -> bool:
    return (t_start - C.CLOCK_WINDOW_SLACK_S) <= t <= (t_end + C.CLOCK_WINDOW_SLACK_S)


def _transported(c: GraspCandidate, min_transport_m: float) -> bool:
    """Did the EE travel horizontally between grasp and release? A success grasp
    that re-opens where it closed (EE barely moved) is a re-grip/fumble at the
    pick, not a placement.

    Delegates to segmentation.transported so the OFFLINE gate here and the LIVE
    cockpit gate in live.LiveLabeler can never disagree about what counts as a
    placement. Do not reimplement the threshold locally.
    """
    return transported(c.grasp_pose, c.release_pose, min_transport_m)


def _is_terminal(c: GraspCandidate, min_transport_m: float = 0.0) -> bool:
    """A grasp that actually carried a bag, moved it, and released it = the placing grasp."""
    return (c.outcome == "success" and c.lifted is not False and c.t_open is not None
            and _transported(c, min_transport_m))


def build_annotations(
    episode_id: str,
    arm: str,
    t_start: float,
    t_end: float,
    candidates: list[GraspCandidate],
    kitting_list: list[KitItem] | None = None,
    cockpit_events: list[dict] | None = None,
    compartments: list[Compartment] | None = None,
    clock_offset_s: float = 0.0,
    outcome: str = "unknown",
    min_transport_m: float = 0.0,
    geometric_targets: bool = False,
) -> Annotations:
    flags: list[Flag] = []
    kit = list(kitting_list or [])

    # -- validate cockpit events against the episode clock window ------------
    place_events_c: list[dict] = []
    for e in cockpit_events or []:
        te = float(e.get("t", 0.0)) + clock_offset_s
        if not _in_window(te, t_start, t_end):
            flags.append(Flag("clock_out_of_window", f"event {e.get('type')} @ {te:.3f}", t=te))
            continue
        if e.get("type") == "place_confirmed":
            place_events_c.append({**e, "t": te})
    place_events_c.sort(key=lambda e: e["t"])

    # -- overlap (single-bag-in-flight) invariant ---------------------------
    cand = sorted(candidates, key=lambda c: c.t_close)
    for a, b in zip(cand, cand[1:]):
        if a.t_open is not None and b.t_close < a.t_open:
            flags.append(Flag("overlapping_grasp",
                              f"grasp @ {b.t_close:.3f} starts before prior released @ {a.t_open:.3f}",
                              t=b.t_close))

    # -- group candidates into bags -----------------------------------------
    grasp_attempts: list[GraspAttempt] = []
    place_events: list[PlaceEvent] = []
    segments: list[Segment] = []

    bag_counter = 0
    pending: list[GraspCandidate] = []      # attempts on the current bag

    def close_bag(terminal: GraspCandidate) -> None:
        nonlocal bag_counter
        bag_counter += 1
        # bag identity / target: cockpit place event nearest the release, else kit order
        bag_id = bag_counter
        target_comp = None
        matched = _match_place_event(terminal, place_events_c)
        if matched is not None:
            bag_id = int(matched.get("bag_id", bag_counter) or bag_counter)
            target_comp = matched.get("comp", matched.get("compartment"))
        elif bag_counter - 1 < len(kit):
            k = kit[bag_counter - 1]
            bag_id = k.bag_id
            target_comp = k.compartment

        attempts = pending + [terminal]
        for i, c in enumerate(attempts, start=1):
            grasp_attempts.append(GraspAttempt(
                bag_id=bag_id, attempt=i, arm=arm, t=c.t_close,
                ee_pose=c.grasp_pose, outcome=c.outcome,
                regrasp_of=(i - 1) if i > 1 else None,   # 2nd+ attempt re-grasps prior
            ))
        # Retargeting: the operator let it slip and re-grasped. Flag loudly so these
        # demos can be filtered out of (or studied separately from) the clean set.
        if len(attempts) > 1:
            outs = ",".join(c.outcome for c in attempts[:-1])
            flags.append(Flag("retargeting",
                              f"bag {bag_id} re-grasped {len(attempts) - 1}x ({outs} → retarget) before place",
                              t=terminal.t_close, bag_id=bag_id))

        # place event from the terminal grasp's release
        rp = terminal.release_pose
        place = PlaceEvent(bag_id=bag_id, t=terminal.t_open, target_compartment=target_comp,
                           achieved_ee_pose=rp)
        if rp is not None and compartments:
            res = classify_release((rp[0], rp[1]), target_comp, compartments)
            place.detected_compartment = res.detected_compartment   # where it actually landed
            place.in_target_region = res.in_target_region
            place.xy_offset_m = res.xy_offset_m
            place.release_height_m = rp[2]
        place_events.append(place)

        # coarse phase segments for this bag
        segments.append(Segment(bag_id, arm, "grasp", terminal.t_close, terminal.t_close))
        segments.append(Segment(bag_id, arm, "transport", terminal.t_close, terminal.t_open))
        segments.append(Segment(bag_id, arm, "place", terminal.t_open, terminal.t_open))

    for c in cand:
        if _is_terminal(c, min_transport_m):
            close_bag(c)
            pending = []
        else:
            if (c.outcome == "success" and c.lifted is not False and c.t_open is not None
                    and not _transported(c, min_transport_m)):
                flags.append(Flag(
                    "no_transport",
                    f"success grasp @ {c.t_close:.3f} re-opened without transporting "
                    f"(<{min_transport_m*100:.0f}cm from grasp) — re-grip at pick, not a placement",
                    t=c.t_close))
            pending.append(c)

    # any leftover attempts that never placed → flag, don't drop
    for c in pending:
        flags.append(Flag("unplaced_grasp", f"{c.outcome} grasp @ {c.t_close:.3f} never placed",
                          t=c.t_close))

    # -- OCR-null kit items --------------------------------------------------
    for k in kit:
        if k.part_no is None:
            flags.append(Flag("ocr_null", f"bag {k.bag_id} part unreadable — needs manual assign",
                              bag_id=k.bag_id))

    # Optional: recover each placement's target from geometry instead of pick
    # order (operator picks out of kit order → kit-order targets are wrong). One
    # bag per compartment → optimal assignment. Loud flag so it's never silent.
    if geometric_targets and compartments:
        n = assign_targets_geometric(place_events, compartments)
        if n:
            flags.append(Flag("geometric_targets",
                              f"reassigned {n} place target(s) to nearest distinct "
                              f"compartment by geometry (kit-order targets overridden)"))

    meta = EpisodeMeta(episode_id=episode_id, arm=arm, kitting_list=kit, outcome=outcome,
                       t_start=t_start, t_end=t_end, clock_offset_s=clock_offset_s)
    return Annotations(episode_meta=meta, segments=segments, grasp_attempts=grasp_attempts,
                       place_events=place_events, flags=flags)


def _match_place_event(terminal: GraspCandidate, place_events_c: list[dict]) -> dict | None:
    """Nearest-in-time place_confirmed event to this grasp's release (within 3s)."""
    if not place_events_c or terminal.t_open is None:
        return None
    best = min(place_events_c, key=lambda e: abs(e["t"] - terminal.t_open))
    return best if abs(best["t"] - terminal.t_open) <= 3.0 else None

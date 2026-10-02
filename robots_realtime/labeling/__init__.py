"""Kitting teleop trajectory auto-labeling.

Turns a recorded episode (MCAP joint/gripper streams + camera MP4s + cockpit
event log) into an immutable ``annotations.json`` sidecar describing, per bag:
grasp/release keyframes, phase segments, placement correctness, and FK/IK
tracking error. Human corrections live in a separate ``corrections.json`` and
are merged on read.

Design notes (see ~/Desktop/kitting/cockpit/kitting-labeling-plan.md):
  * The raw recording is never modified. ``annotations.json`` is regenerable.
  * No gripper force sensor exists — grasp/slip come from gripper WIDTH
    (position, motor index 6) + vision, never "gripper effort".
  * Single-arm (left) kitting; the schema is arm-agnostic.
"""

from robots_realtime.labeling.schema import (  # noqa: F401
    LABELER_VERSION,
    Annotations,
    EpisodeMeta,
    GraspAttempt,
    PlaceEvent,
    Segment,
    TrackingError,
    load_annotations,
    merge_corrections,
)

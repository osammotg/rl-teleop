"""Schema round-trip + corrections-merge precedence."""
from __future__ import annotations

import json

from robots_realtime.labeling.schema import (
    LABELER_VERSION,
    Annotations,
    EpisodeMeta,
    GraspAttempt,
    KitItem,
    PlaceEvent,
    load_annotations,
    merge_corrections,
)


def _sample() -> Annotations:
    return Annotations(
        episode_meta=EpisodeMeta(
            episode_id="ep1",
            arm="left",
            kitting_list=[
                KitItem(bag_id=1, part_no="UNN-10126-151", name="Flügelmutter", compartment=5),
                KitItem(bag_id=2, part_no=None, compartment=None),  # OCR failed
            ],
            outcome="success",
            t_start=100.0,
            t_end=142.0,
        ),
        grasp_attempts=[
            GraspAttempt(bag_id=1, attempt=1, arm="left", t=105.0,
                         ee_pose=[0.4, -0.3, 0.2, 1, 0, 0, 0], close_width_norm=0.4,
                         outcome="success"),
        ],
        place_events=[
            PlaceEvent(bag_id=1, t=110.0, target_compartment=5,
                       achieved_ee_pose=[0.5, -0.1, 0.15, 1, 0, 0, 0],
                       in_target_region=True, xy_offset_m=0.01),
        ],
    )


def test_round_trip_preserves_data():
    ann = _sample()
    restored = Annotations.from_dict(json.loads(json.dumps(ann.to_dict())))
    assert restored.to_dict() == ann.to_dict()
    assert restored.episode_meta.labeler_version == LABELER_VERSION
    assert restored.episode_meta.kitting_list[1].part_no is None
    assert restored.grasp_attempts[0].outcome == "success"


def test_unknown_fields_ignored_forward_compat():
    d = _sample().to_dict()
    d["episode_meta"]["some_future_field"] = 123
    d["grasp_attempts"][0]["future"] = "x"
    ann = Annotations.from_dict(d)  # must not raise
    assert ann.episode_meta.episode_id == "ep1"


def test_corrections_win_and_are_partial():
    ann = _sample()
    corrections = {
        # human fixes the unread bag: assign part + compartment
        "kitting_list": {"2": {"part_no": "DNN-15122-009", "compartment": 3}},
        # human corrects a wrong compartment on the place event
        "place_events": {"1": {"target_compartment": 6}},
    }
    merged = merge_corrections(ann, corrections)
    assert merged.episode_meta.kitting_list[1].part_no == "DNN-15122-009"
    assert merged.episode_meta.kitting_list[1].compartment == 3
    # partial: the machine-derived pose survives the correction
    assert merged.place_events[0].target_compartment == 6
    assert merged.place_events[0].achieved_ee_pose == [0.5, -0.1, 0.15, 1, 0, 0, 0]
    # original untouched
    assert ann.place_events[0].target_compartment == 5


def test_load_merges_corrections(tmp_path):
    ann = _sample()
    ann.save(tmp_path / "annotations.json")
    (tmp_path / "corrections.json").write_text(
        json.dumps({"place_events": {"1": {"in_target_region": False}}})
    )
    loaded = load_annotations(tmp_path)
    assert loaded.place_events[0].in_target_region is False


def test_load_without_corrections(tmp_path):
    ann = _sample()
    ann.save(tmp_path / "annotations.json")
    loaded = load_annotations(tmp_path)
    assert loaded.place_events[0].in_target_region is True

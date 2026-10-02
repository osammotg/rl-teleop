"""Process-model aggregation over synthetic labeled episodes."""
from __future__ import annotations

from robots_realtime.labeling.aggregator import aggregate
from robots_realtime.labeling.schema import (
    Annotations,
    EpisodeMeta,
    GraspAttempt,
    KitItem,
    PlaceEvent,
)


def _episode(part, comp, grasp_z, place_xy, outcome="success", in_region=True):
    return Annotations(
        episode_meta=EpisodeMeta(
            episode_id="e", kitting_list=[KitItem(1, part, compartment=comp)]),
        grasp_attempts=[GraspAttempt(1, 1, "left", 0.0,
                                     ee_pose=[0.4, -0.3, grasp_z, 1, 0, 0, 0],
                                     outcome=outcome)],
        place_events=[PlaceEvent(1, 1.0, target_compartment=comp,
                                 achieved_ee_pose=[place_xy[0], place_xy[1], 0.15, 1, 0, 0, 0],
                                 in_target_region=in_region, release_height_m=0.15)],
    )


def test_grasp_stats_per_part():
    eps = [_episode("P1", 5, 0.20, (0.5, -0.1)),
           _episode("P1", 5, 0.22, (0.52, -0.1)),
           _episode("P2", 3, 0.30, (0.3, 0.1))]
    m = aggregate(eps)
    assert m.n_episodes == 3
    assert m.grasps["P1"].n == 2
    assert abs(m.grasps["P1"].mean_pose[2] - 0.21) < 1e-6   # mean grasp z
    assert m.grasps["P1"].success_rate == 1.0
    assert m.grasps["P2"].n == 1


def test_success_rate_counts_failures():
    eps = [_episode("P1", 5, 0.2, (0.5, -0.1), outcome="success"),
           _episode("P1", 5, 0.2, (0.5, -0.1), outcome="slip")]
    m = aggregate(eps)
    assert m.grasps["P1"].success_rate == 0.5


def test_drop_distribution_per_compartment():
    eps = [_episode("P1", 5, 0.2, (0.50, -0.10)),
           _episode("P1", 5, 0.2, (0.54, -0.14))]
    m = aggregate(eps)
    d = m.drops["5"]
    assert d.n == 2
    assert abs(d.mean_xy[0] - 0.52) < 1e-6
    assert d.in_region_rate == 1.0
    assert abs(d.mean_release_height - 0.15) < 1e-9


def test_in_region_rate_mixed():
    eps = [_episode("P1", 5, 0.2, (0.5, -0.1), in_region=True),
           _episode("P1", 5, 0.2, (0.9, -0.1), in_region=False)]
    m = aggregate(eps)
    assert m.drops["5"].in_region_rate == 0.5


def test_empty_corpus():
    m = aggregate([])
    assert m.n_episodes == 0 and m.grasps == {} and m.drops == {}

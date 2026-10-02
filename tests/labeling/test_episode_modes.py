"""Episode modes: does a gated placement save a take, hands-free?

`--episode-mode grasp` is the "never touch the keyboard" path — the labeler
already knows the instant a placement passes the transport gate, and this wires
that to rr-session's /record/advance. These tests drive the real LiveLabeler
with synthetic gripper+pose samples and assert the callback fires exactly on
gated placements: not on re-grips, not on empty closes, not on slips.

No robot, no HTTP, no cameras.
"""
from __future__ import annotations

import time

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.live import LiveLabeler


def _packets(n=3):
    return [{"part": f"UNN-{i}", "name": f"part{i}", "comp": i + 1} for i in range(n)]


def _feed(lab, samples):
    """samples: (t, width, [x,y,z])"""
    for t, w, ee in samples:
        lab.push(t, w, ee)


def _grasp_cycle(t0, x0, x1, *, hold_s=None, lift=0.12, open_w=1.0, closed_w=0.0):
    """One open→close→carry→release cycle as a sample list.

    hold_s defaults to comfortably above MIN_HOLD_S so the segmenter commits.
    """
    hold_s = hold_s if hold_s is not None else C.MIN_HOLD_S * 4
    s = [(t0, open_w, [x0, 0.0, 0.0])]
    # close and hold, rising (a real pick lifts)
    n = 12
    for i in range(n):
        f = (i + 1) / n
        s.append((t0 + 0.05 + hold_s * f, closed_w,
                  [x0 + (x1 - x0) * f, 0.0, lift * min(1.0, f * 2)]))
    # release at the far end
    s.append((t0 + 0.05 + hold_s + 0.05, open_w, [x1, 0.0, lift]))
    return s


def test_grasp_mode_fires_once_per_gated_placement():
    fired = []
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.on_place = lambda: fired.append(time.time())
    lab.seed(_packets(3))

    # two clean pick-and-places, each carrying well past MIN_TRANSPORT_M
    _feed(lab, _grasp_cycle(100.0, 0.0, 0.40))
    _feed(lab, _grasp_cycle(110.0, 0.0, 0.40))

    time.sleep(0.2)                       # callback runs on a daemon thread
    assert len(fired) == 2, f"expected 2 auto-advances, got {len(fired)}"
    assert lab.ti == 2


def test_full_mode_never_fires():
    """on_place unset = the operator ends episodes. Must stay silent."""
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed(_packets(2))
    _feed(lab, _grasp_cycle(100.0, 0.0, 0.40))
    time.sleep(0.2)
    assert lab.ti == 1, "the pointer should still advance in full mode"
    # nothing to assert about firing beyond 'it did not crash without a callback'


def test_regrip_at_the_pick_does_not_save_a_take():
    """Grasped and put straight back down = a fumble, not a placement. Saving a
    take here would pollute the dataset with non-episodes."""
    fired = []
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.on_place = lambda: fired.append(1)
    lab.seed(_packets(2))

    # carries only 2 cm — under MIN_TRANSPORT_M (0.10)
    _feed(lab, _grasp_cycle(100.0, 0.0, 0.02))
    time.sleep(0.2)
    assert fired == [], "a re-grip must not save a take"
    assert lab.ti == 0, "pointer must stay on the same packet"
    assert lab.regrasps >= 1


def test_twitch_below_min_hold_is_ignored():
    """Sub-MIN_HOLD_S open/close is gripper jitter. The corpus had 124 of these;
    raising MIN_HOLD_S to 0.5s removed 79% of them."""
    fired = []
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.on_place = lambda: fired.append(1)
    lab.seed(_packets(2))

    _feed(lab, _grasp_cycle(100.0, 0.0, 0.40, hold_s=C.MIN_HOLD_S * 0.4))
    time.sleep(0.2)
    assert fired == [], "a twitch shorter than MIN_HOLD_S must not count"


def test_callback_exception_does_not_break_the_labeler():
    """If rr-session is down, auto-advance fails — the labeler must keep going
    so the operator can still finish by hand."""
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)

    def boom():
        raise RuntimeError("rr-session unreachable")

    lab.on_place = boom
    lab.seed(_packets(2))
    _feed(lab, _grasp_cycle(100.0, 0.0, 0.40))
    time.sleep(0.2)
    assert lab.ti == 1, "pointer must still advance even though the callback threw"

    # and a second cycle still works
    _feed(lab, _grasp_cycle(110.0, 0.0, 0.40))
    time.sleep(0.2)
    assert lab.ti == 2


def test_callback_runs_off_the_feed_thread():
    """A slow HTTP call must not stall the joint feed."""
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.on_place = lambda: time.sleep(1.0)
    lab.seed(_packets(2))

    t0 = time.monotonic()
    _feed(lab, _grasp_cycle(100.0, 0.0, 0.40))
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"push() blocked for {elapsed:.2f}s on the callback"

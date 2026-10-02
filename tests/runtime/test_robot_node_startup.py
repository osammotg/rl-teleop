"""Startup-hold tests — no robot, no CAN bus, no arm powered.

WHAT THIS GUARDS. A YAM has no brakes. The driver's control loop holds the LAST
COMMANDED position, and until 2026-07-29 an arm session with no startup pose and
no publisher never called command_joint_pos() at all — so there was no last
command, the motors were limp, and the arm sank into the table. Measured that
day: an arm parked at home, unpowered 16 h, came up reading joint2 +53.3 deg off
home and resting on the plate at -5.8 cm clearance.

`startup_hold` fixes it by commanding the arm's OWN measured position once, which
is a zero-motion command and therefore safe from any pose — including one already
lying on the table, where a startup ramp would sweep blindly instead
(_move_to_pose has no clearance model, and it runs in setup() before the step
loop, where nothing can observe or abort it).

THE TEST THAT MATTERS MOST is test_hold_is_a_constant_not_a_measurement. The hold
target is latched once. If it were ever re-read from the live measurement, the arm
would ratchet downward — every joint settles slightly below what it is told, so
commanding the measurement bakes in the sag and then sags again from there — and it
would do that while every reading looked perfectly healthy. That is the only
silent failure mode in this change.

Run:  .venv/bin/python -m pytest tests/runtime/test_robot_node_startup.py -q

NOT `uv run pytest`: that re-syncs from uv.lock and swaps torch to a cu126 build
with no kernels for this sm_120 GPU. See TODOS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.runtime.environment.robot_node import RobotNode


class FakeRobot:
    """Records every command. Reports a position that SAGS on each read.

    The sag is the point: it makes the ratchet failure detectable. A fake that
    returned a constant position would pass the ratchet test no matter how the
    implementation behaved, which would make the most important test in this file
    worthless.
    """

    def __init__(self, q0=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), sag=0.01, can_read=True):
        self._q = np.array(q0, dtype=np.float64)
        self._sag = float(sag)
        self._can_read = can_read
        self.commands: list[np.ndarray] = []
        self.moves: list[tuple] = []

    # -- driver surface RobotNode uses --------------------------------------
    def get_joint_pos(self):
        if not self._can_read:
            raise AttributeError("driver has no get_joint_pos")
        self._q = self._q - self._sag          # every read is a little lower
        return self._q.copy()

    def command_joint_pos(self, q):
        self.commands.append(np.array(q, dtype=np.float64))

    def get_observations(self):
        return {"joint_pos": self._q[:6].copy()}

    def move_joints(self, target, time_interval_s=0.0):
        self.moves.append((np.array(target, dtype=np.float64), time_interval_s))


def make_node(robot, **kw):
    """A RobotNode wired to a fake robot, with the bus stubbed out."""
    node = RobotNode(robot=robot, name="yam_test", cmd_topic="ik_cmd/joint_pos", **kw)
    node._published: list = []
    node.publish = lambda topic, data, ts=None: node._published.append((topic, data))
    # Bus reads are driven per-test.
    node._fake_cmd = None
    node._fake_cmd_ts = None
    node.get_latest = lambda topic: node._fake_cmd
    node.get_timestamp = lambda topic: node._fake_cmd_ts
    return node


def send(node, q, ts):
    node._fake_cmd = {"joint_pos": list(q)}
    node._fake_cmd_ts = ts


# ── setup() ─────────────────────────────────────────────────────────────────
def test_hold_commands_the_measured_position_without_moving():
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True)
    node.setup()
    assert len(r.commands) == 1, "hold must issue exactly one command at boot"
    # zero motion: the command IS the measured position
    np.testing.assert_allclose(r.commands[0], r._q)
    assert not r.moves, "hold must not interpolate anywhere"


def test_hold_refuses_to_start_when_position_cannot_be_read():
    """Falling through to limp is the failure this option exists to prevent, so
    an unreadable driver is a hard error rather than a warning."""
    r = FakeRobot(can_read=False)
    node = make_node(r, startup_hold=True)
    with pytest.raises(RuntimeError, match="startup_hold"):
        node.setup()
    assert not r.commands


def test_startup_joint_pos_wins_over_hold():
    r = FakeRobot()
    node = make_node(r, startup_hold=True, startup_joint_pos=[0.0] * 7,
                     startup_duration_s=0.0)
    node.setup()
    assert r.moves, "an explicit startup pose must still be honoured"
    assert node._hold_q is None, "hold must not latch when a startup pose is set"


# ── REGRESSION guards: unchanged behaviour for every existing config ────────
def test_regression_startup_joint_pos_alone_is_unchanged():
    """6 bimanual teleop configs set startup_joint_pos and no hold."""
    r = FakeRobot()
    node = make_node(r, startup_joint_pos=[0.0] * 7, startup_duration_s=0.0)
    node.setup()
    assert r.moves and node._hold_q is None


def test_regression_neither_option_commands_nothing_at_boot():
    """The pre-2026-07-29 default. Preserved so enabling the fix stays opt-in."""
    r = FakeRobot()
    node = make_node(r)
    node.setup()
    assert not r.commands and not r.moves
    assert node._hold_q is None


def test_regression_paused_commands_nothing():
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True)
    node.setup()
    r.commands.clear()
    node._paused = True
    node.step()
    assert not r.commands, "a paused node must not command, hold or otherwise"


# ── step() ──────────────────────────────────────────────────────────────────
def test_hold_is_maintained_while_no_publisher_exists():
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True)
    node.setup()
    r.commands.clear()
    for _ in range(5):
        node.step()
    assert len(r.commands) == 5, "the hold must be re-commanded, not issued once"


def test_hold_is_a_constant_not_a_measurement():
    """THE test. See the module docstring.

    The fake sags 10 mrad on every position read. If the implementation re-read
    the position to build its hold target, these commands would walk downward.
    They must all be byte-identical instead.
    """
    # First prove the fake actually sags when read, on a THROWAWAY instance.
    # Checking this on the instance under test cannot work: a correct
    # implementation never re-reads the position, so the fake would never sag and
    # the guard would fail against working code. (It did, on the first run.)
    probe = FakeRobot(sag=0.01)
    assert not np.allclose(probe.get_joint_pos(), probe.get_joint_pos()), \
        "the fake must sag on each read, or this test cannot detect a ratchet"

    r = FakeRobot(sag=0.01)
    node = make_node(r, startup_hold=True)
    node.setup()
    latched = r.commands[0].copy()
    r.commands.clear()
    for _ in range(40):
        node.step()
    assert len(r.commands) == 40
    for i, c in enumerate(r.commands):
        np.testing.assert_array_equal(
            c, latched,
            err_msg=f"hold command {i} drifted from the latched pose -- the arm "
                    f"would ratchet into the table")


def test_publisher_takes_over_and_releases_the_hold():
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True)
    node.setup()
    r.commands.clear()
    send(node, [1.0] * 7, ts=100.0)
    node.step()
    assert node._hold_q is None, "the command stream owns the arm now"
    assert len(r.commands) == 1


def test_stale_command_keeps_holding_and_does_not_relatch():
    """After a publisher dies, get_latest still returns its last envelope forever,
    so the stale-target branch holds the arm. That is the documented
    hold-on-Ctrl-C behaviour the autonomous sort loop depends on."""
    # ramp_duration_s=0 so commands pass straight through. With the 1.5 s default
    # these 10 instant ticks all land inside the handoff ramp and read as blended
    # values near the arm's own pose, which looks like a failure and is not one.
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True, ramp_duration_s=0.0)
    node.setup()
    send(node, [1.0] * 7, ts=100.0)
    node.step()
    r.commands.clear()
    for _ in range(10):          # publisher gone: same ts, same envelope
        node.step()
    assert node._hold_q is None, "must not re-latch and fight the stale target"
    assert len(r.commands) == 10
    for c in r.commands:
        np.testing.assert_allclose(c, [1.0] * 7)


def test_handoff_from_hold_ramps_from_the_measured_position():
    """The first command must ease in from where the arm actually is, not snap."""
    r = FakeRobot(sag=0.0)
    node = make_node(r, startup_hold=True, ramp_duration_s=10.0)
    node.setup()
    start = r._q.copy()
    r.commands.clear()
    send(node, [5.0] * 7, ts=100.0)
    node.step()
    first = r.commands[0]
    assert np.all(np.abs(first - start) < np.abs(np.array([5.0] * 7) - start)), \
        "first command should be between the arm's pose and the target, not at it"


# ── config plumbing ─────────────────────────────────────────────────────────
def test_startup_hold_survives_build_kwargs():
    """A YAML key that build_kwargs drops is a silent no-op, which for this option
    means an arm that is limp while the config says it is held."""
    kw = RobotNode.build_kwargs({"name": "yam_left", "startup_hold": True})
    assert kw.get("startup_hold") is True


def test_read_joint_pos_returns_none_instead_of_raising():
    r = FakeRobot(can_read=False)
    node = make_node(r)
    assert node._read_joint_pos() is None


def test_read_joint_pos_returns_the_full_command_vector():
    r = FakeRobot(sag=0.0)
    node = make_node(r)
    q = node._read_joint_pos()
    assert q is not None and q.shape == (7,), "must include the gripper, not 6 joints"

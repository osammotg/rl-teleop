import numpy as np
import pytest
import yaml
from unittest.mock import patch

from robots_realtime.agents.teleoperation.yam_leader_agent import YamLeaderAgent

class FakeEncoderState:
    """Encoder state container."""
    def __init__(self, position: float, io_inputs: np.ndarray):
        self.position = position
        self.io_inputs = io_inputs

class FakeMotorChain:
    """Motor chain component."""
    def __init__(self):
        self.same_bus_device_driver = True
        self.get_same_bus_device_states_calls = 0
        self.last_encoder_state = None

    def get_same_bus_device_states(self):
        self.get_same_bus_device_states_calls += 1
        if self.last_encoder_state is None:
            self.last_encoder_state = FakeEncoderState(1.0, np.zeros(8))
        return [self.last_encoder_state]

class FakeYAMRobot:
    """MotorChainRobot mock."""
    def __init__(self):
        self.motor_chain = FakeMotorChain()
        self._kp = np.ones(6, dtype=np.float64)
        self.get_observations_calls = 0
        self.update_kp_kd_calls = 0
        self.command_joint_pos_calls = 0
        self.close_calls = 0
        self.last_commanded_pos = None
        self.last_kp_kd = None

    def get_observations(self):
        self.get_observations_calls += 1
        return {"joint_pos": np.zeros(7, dtype=np.float32)}

    def update_kp_kd(self, kp, kd):
        self.update_kp_kd_calls += 1
        self.last_kp_kd = (kp, kd)

    def command_joint_pos(self, pos):
        self.command_joint_pos_calls += 1
        self.last_commanded_pos = pos

    def close(self):
        self.close_calls += 1

@pytest.fixture
def mock_instantiate():
    """Monkeypatch _instantiate_from_target_yaml."""
    with patch("robots_realtime.agents.teleoperation.yam_leader_agent._instantiate_from_target_yaml") as mock:
        mock.return_value = FakeYAMRobot()
        yield mock

def test_yam_leader_act_smoke_with_gripper(mock_instantiate):
    """Verify output shape/type for gripper inclusion."""
    robot_config = "fake/path/to/config.yaml"
    agent = YamLeaderAgent(
        robot_config=robot_config,
        robot_name="test_arm",
        include_gripper=True,
        bilateral_kp=0.0,
        start_enabled=True
    )
    
    obs = {"test_arm": {"joint_pos": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)}}
    action = agent.act(obs)

    assert "pos" in action
    pos = action["pos"]
    assert pos.shape == (7,)
    assert pos.dtype == np.float32
    assert isinstance(pos, np.ndarray)
    assert agent.robot.get_observations_calls == 1
    assert agent.robot.motor_chain.get_same_bus_device_states_calls == 1

def test_yam_leader_bilateral_feedback(mock_instantiate):
    """Verify bilateral control triggers motor commands."""
    robot_config = "fake/path/to/config.yaml"
    bilateral_kp = 0.5
    agent = YamLeaderAgent(
        robot_config=robot_config,
        robot_name="test_arm",
        include_gripper=False,
        bilateral_kp=bilateral_kp,
        warmup_steps=0,
        start_enabled=True
    )
    
    agent.robot._kp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    follower_pos = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    obs = {"test_arm": {"joint_pos": follower_pos}}

    agent.act(obs)

    assert agent.robot.update_kp_kd_calls > 0
    expected_kp = agent.robot._kp * bilateral_kp
    np.testing.assert_allclose(agent.robot.last_kp_kd[0], expected_kp)
    assert agent.robot.command_joint_pos_calls > 0
    np.testing.assert_allclose(agent.robot.last_commanded_pos, follower_pos)

def test_yam_leader_gripper_logic(mock_instantiate):
    """Verify gripper command derivation."""
    robot_config = "fake/path/to/config.yaml"
    agent = YamLeaderAgent(
        robot_config=robot_config,
        robot_name="test_arm",
        include_gripper=True,
        gripper_open_pos=0.0,
        gripper_close_pos=1.0,
        start_enabled=True
    )
    
    agent.robot.motor_chain.last_encoder_state = FakeEncoderState(0.5, np.zeros(8))
    action = agent.act({})
    gripper_val = action["pos"][6]
    assert np.isclose(gripper_val, 0.5)

def test_yam_leader_button_toggle(mock_instantiate):
    """Verify button press toggles state."""
    robot_config = "fake/path/to/config.yaml"
    agent = YamLeaderAgent(
        robot_config=robot_config,
        robot_name="test_arm",
        enable_button_index=0,
        start_enabled=True
    )
    
    agent.robot.motor_chain.last_encoder_state = FakeEncoderState(0.0, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    agent.act({})
    assert agent._enabled is False

def test_yam_leader_close(mock_instantiate):
    """Verify robot close is called."""
    agent = YamLeaderAgent(robot_config="fake.yaml")
    agent.close()
    assert agent.robot.close_calls == 1

def test_force_feedback_gains_applied_once(mock_instantiate):
    """10 post-warmup acts should call update_kp_kd exactly once, command_joint_pos 10 times."""
    agent = YamLeaderAgent(
        robot_config="fake.yaml",
        robot_name="test",
        include_gripper=False,
        bilateral_kp=0.05,
        warmup_steps=0,
        start_enabled=True,
    )
    follower_obs = {"test": {"joint_pos": np.ones(6, dtype=np.float32)}}
    for _ in range(10):
        agent.act(follower_obs)
    assert agent.robot.update_kp_kd_calls == 1
    assert agent.robot.command_joint_pos_calls == 10

def test_warmup_skips_feedback(mock_instantiate):
    """During warmup, no feedback commands sent."""
    agent = YamLeaderAgent(
        robot_config="fake.yaml",
        robot_name="test",
        include_gripper=False,
        bilateral_kp=0.05,
        warmup_steps=5,
        start_enabled=True,
    )
    follower_obs = {"test": {"joint_pos": np.ones(6, dtype=np.float32)}}
    for i in range(5):
        agent.act(follower_obs)
    assert agent.robot.command_joint_pos_calls == 0

def test_close_restores_gains(mock_instantiate):
    """close() restores original kp/kd if feedback was active."""
    agent = YamLeaderAgent(
        robot_config="fake.yaml",
        robot_name="test",
        include_gripper=False,
        bilateral_kp=0.05,
        warmup_steps=0,
        start_enabled=True,
    )
    original_kp = agent.robot._kp.copy()
    agent.act({"test": {"joint_pos": np.ones(6, dtype=np.float32)}})
    assert agent._feedback_gains_applied is True
    agent.close()
    assert agent._feedback_gains_applied is False
    assert agent.robot.update_kp_kd_calls == 2  # one for apply, one for restore

def test_bilateral_kp_zero_no_feedback(mock_instantiate):
    """When bilateral_kp==0, no feedback gains or position commands are sent."""
    agent = YamLeaderAgent(
        robot_config="fake.yaml",
        robot_name="test_arm",
        include_gripper=False,
        bilateral_kp=0.0,
        warmup_steps=0,
        start_enabled=True,
    )

    assert agent._bilateral_enabled is False
    assert agent._original_kp is None

    obs = {"test_arm": {"joint_pos": np.ones(6, dtype=np.float32)}}

    for _ in range(5):
        agent.act(obs)

    assert agent.robot.update_kp_kd_calls == 0, "bilateral_kp=0 should never call update_kp_kd"
    assert agent.robot.command_joint_pos_calls == 0, "bilateral_kp=0 should never call command_joint_pos"

def test_config_guardrail_flat_out_and_bilateral_kp():
    """Verify yam_bimanual_yam_leader.yaml has correct loop_mode and bilateral_kp."""
    config_path = "configs/yam/yam_bimanual_yam_leader.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    leader_names = ["yam_leader_left", "yam_leader_right"]
    
    for name in leader_names:
        # Find the node by name
        node = next((n for n in config['nodes'] if n.get('name') == name), None)
        assert node is not None, f"Node {name} not found in config"

        # 1. Both yam_leader_left and yam_leader_right use loop_mode: flat_out
        assert node.get("loop_mode") == "flat_out", f"{name} must have loop_mode: flat_out"

        # 2. Neither sets poll_freq: 8 (or any poll_freq at all)
        assert "poll_freq" not in node, f"{name} should not have poll_freq set"

        # 3. Both keep bilateral_kp > 0 (currently 0.05)
        # bilateral_kp is inside agent_kwargs
        agent_kwargs = node.get("agent_kwargs", {})
        bilateral_kp = agent_kwargs.get("bilateral_kp")
        assert bilateral_kp is not None, f"{name} must have bilateral_kp defined in agent_kwargs"
        assert bilateral_kp > 0, f"{name} bilateral_kp must be > 0 (found {bilateral_kp})"

if __name__ == "__main__":
    pytest.main([__file__])


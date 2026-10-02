"""YAM Leader agent that reads joint positions from a YAM arm with teaching handle.

This agent interfaces with the i2rt MotorChainRobot to read joint positions and
gripper commands from a YAM leader arm equipped with a teaching handle.

The teaching handle has:
- A trigger that controls the follower gripper (encoder position)
- Top buttons for enable/disable and user-programmable functions (io_inputs)

Bilateral Control:
------------------
When bilateral_kp > 0, the agent implements position-position bilateral control:
    leader  ──pos──►  follower
    leader  ◄──pos──  follower

The follower's joint positions are read from observations and commanded back to
the leader motors as position targets. The motor's internal PD controller
generates a restoring torque proportional to: τ ∝ Kp × (q_follower − q_leader)

This provides haptic feedback to the operator, allowing them to feel forces and
collisions experienced by the follower arm.
"""

import importlib
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml as _yaml
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6


def _resolve(obj):
    """Recursively instantiate any dict containing a ``_target_`` key.

    Special handling for ``_callable_`` key which returns a function reference
    without calling it (used for callback parameters like get_same_bus_device_driver).
    """
    if isinstance(obj, dict):
        if "_callable_" in obj:
            # Return function reference without calling it
            obj = dict(obj)
            target: str = obj.pop("_callable_")
            module_path, func_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, func_name)
        if "_target_" in obj:
            obj = dict(obj)
            target: str = obj.pop("_target_")
            kwargs = {k: _resolve(v) for k, v in obj.items()}
            module_path, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, cls_name)(**kwargs)
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(item) for item in obj]
    return obj


def _instantiate_from_target_yaml(config_path: str):
    """Load a YAML config and recursively instantiate all ``_target_`` objects."""
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    return _resolve(cfg)


class YamLeaderAgent(Agent):
    """Teleoperation agent for YAM leader arm with teaching handle.

    Reads the leader's 6 joint positions from the YAM arm and the gripper
    command from the teaching handle encoder (trigger position).

    The robot config should specify a MotorChainRobot with:
    - 6 DM motors for the arm (0x01-0x06)
    - get_same_bus_device_driver configured to read the passive encoder (0x50E)

    Args:
        robot_config: Path to the robot config YAML file
        robot_name: Key used in the returned action dict (must match follower robot)
        joint_signs: Per-joint sign multipliers (length 6). Use -1 to flip a joint.
        joint_offsets: Per-joint offsets in radians added after sign flip.
        include_gripper: If True, include the gripper value as a 7th element.
        gripper_open_pos: Encoder position corresponding to fully open gripper (default 0.0)
        gripper_close_pos: Encoder position corresponding to fully closed gripper (default ~1.0)
        enable_button_index: Which button (0 or 1) enables/disables synchronization.
            When not enabled, returns None for actions to prevent follower movement.
        bilateral_kp: Bilateral control stiffness factor (0.0 to 1.0).
            0.0 disables bilateral feedback (pass-through teleoperation).
            Higher values increase haptic feedback strength.
        warmup_steps: Number of control steps to skip bilateral feedback at startup,
            allowing the follower to converge to the leader's initial position.
    """

    def __init__(
        self,
        robot_config: str,
        robot_name: str = "yam_left",
        joint_signs: Optional[List[int]] = None,
        joint_offsets: Optional[List[float]] = None,
        include_gripper: bool = True,
        gripper_open_pos: float = 0.0,
        gripper_close_pos: float = 1.0,
        enable_button_index: int = 0,
        bilateral_kp: float = 0.0,
        warmup_steps: int = 5,
        start_enabled: bool = True,
        profile: bool = False,
        profile_log_interval: float = 5.0,
        profile_slow_ms: float = 50.0,
    ) -> None:
        self.robot_name = robot_name
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets = np.array(joint_offsets or [0.0] * NUM_ARM_JOINTS, dtype=np.float64)
        self.include_gripper = include_gripper
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos
        self.enable_button_index = enable_button_index
        self.bilateral_kp = bilateral_kp
        self._warmup_steps = warmup_steps
        self._step = 0
        self._bilateral_enabled = bilateral_kp > 0.0
        self._use_button = enable_button_index >= 0
        self._start_enabled = start_enabled
        self._encoder_failed = False  # Track if encoder setup failed
        self._profile = profile
        self._profile_log_interval = profile_log_interval
        self._profile_slow_ms = profile_slow_ms
        self._profile_accum: Dict[str, List[float]] = {}
        self._profile_act_times: List[float] = []
        self._profile_last_log = time.perf_counter()

        # Instantiate the i2rt MotorChainRobot from YAML config
        # The YAML config includes get_same_bus_device_driver which sets up the encoder chain
        print(f"\n{'='*60}")
        print(f"[{robot_name}] YamLeaderAgent.__init__() starting")
        print(f"[{robot_name}] robot_config: {robot_config}")
        print(f"{'='*60}\n")
        self.robot = _instantiate_from_target_yaml(robot_config)
        print(f"[{robot_name}] Robot instantiated successfully")

        # Check if encoder chain was set up (it should be from YAML config)
        if hasattr(self.robot.motor_chain, 'same_bus_device_driver') and self.robot.motor_chain.same_bus_device_driver:
            print(f"[{robot_name}] Teaching handle encoder configured from YAML")
            logger.info(f"Teaching handle encoder configured for {robot_name}")
            self._encoder_failed = False
        else:
            print(f"[{robot_name}] WARNING: No encoder chain found - gripper control will not work")
            logger.warning(f"No encoder chain configured for {robot_name}")
            self._encoder_failed = True

        # Store original kp/kd for bilateral control scaling
        if self._bilateral_enabled and hasattr(self.robot, '_kp'):
            self._original_kp = self.robot._kp.copy()
            self._kd = getattr(self.robot, '_kd', np.zeros(NUM_ARM_JOINTS)).copy()
        else:
            self._original_kp = None
            self._kd = None

        logger.info(f"YamLeaderAgent initialized for {robot_name} (bilateral_kp={bilateral_kp})")

        # Track enable state
        self._enabled = start_enabled
        self._last_button_state = 0
        self._feedback_gains_applied = False

        if start_enabled:
            logger.info(f"YamLeaderAgent {robot_name}: Starting ENABLED")

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Read leader arm and teaching handle, return follower command.

        If bilateral control is enabled, also reads the follower's position from
        observations and commands it back to the leader motors for haptic feedback.

        Returns:
            {robot_name: {"pos": np.ndarray}} - 6 arm joints in radians
            (+ 1 gripper value if include_gripper is True)

            Returns empty dict if not enabled by button press.
        """
        profile_start = time.perf_counter() if self._profile else 0.0
        segments: Dict[str, float] = {}

        t = time.perf_counter() if self._profile else 0.0
        robot_obs = self.robot.get_observations()
        self._record_profile_segment("get_observations", t, segments)
        joint_pos = robot_obs["joint_pos"][:NUM_ARM_JOINTS]

        joint_pos = self.joint_signs * joint_pos + self.joint_offsets

        t = time.perf_counter() if self._profile else 0.0
        encoder_states = self.robot.motor_chain.get_same_bus_device_states()
        self._record_profile_segment("get_same_bus_device_states", t, segments)

        if encoder_states and len(encoder_states) > 0:
            encoder = encoder_states[0]
            gripper_cmd = np.clip(encoder.position, self.gripper_open_pos, self.gripper_close_pos)
            if self._step % 100 == 0:
                print(f"[{self.robot_name}] encoder position: {encoder.position:.3f}, gripper_cmd: {gripper_cmd:.3f}")

            if self._use_button:
                button_state = encoder.io_inputs[self.enable_button_index] if len(encoder.io_inputs) > self.enable_button_index else 0
                if button_state > 0.5 and self._last_button_state < 0.5:
                    self._enabled = not self._enabled
                    state_str = 'ENABLED' if self._enabled else 'DISABLED'
                    logger.info(f"YamLeaderAgent {self.robot_name}: {state_str}")
                self._last_button_state = button_state
        else:
            gripper_cmd = self.gripper_open_pos
            if self._step % 100 == 0:
                print(f"[{self.robot_name}] WARNING: No encoder data from teaching handle")

        if self._bilateral_enabled and self._enabled and self._step >= self._warmup_steps:
            t = time.perf_counter() if self._profile else 0.0
            follower_pos = self._extract_follower_pos(obs)
            self._record_profile_segment("extract_follower_pos", t, segments)
            if follower_pos is not None:
                t = time.perf_counter() if self._profile else 0.0
                self._send_bilateral_feedback(follower_pos[:NUM_ARM_JOINTS])
                self._record_profile_segment("send_bilateral_feedback", t, segments)

        self._step += 1

        if not self._enabled:
            self._finish_profile(profile_start, segments)
            return {}

        if self.include_gripper:
            encoder_normalized = (gripper_cmd - self.gripper_open_pos) / (self.gripper_close_pos - self.gripper_open_pos)
            gripper_pos_normalized = 1.0 - encoder_normalized
            pos = np.concatenate([joint_pos, [gripper_pos_normalized]])
            if self._step % 100 == 0:
                print(f"[{self.robot_name}] encoder: {gripper_cmd:.3f}, encoder_norm: {encoder_normalized:.3f}, gripper_norm: {gripper_pos_normalized:.3f}")
        else:
            pos = joint_pos

        self._finish_profile(profile_start, segments)
        return {"pos": pos.astype(np.float32)}

    def _record_profile_segment(self, key: str, start: float, segments: Dict[str, float]) -> None:
        if not self._profile:
            return
        elapsed_ms = (time.perf_counter() - start) * 1e3
        segments[key] = elapsed_ms
        self._profile_accum.setdefault(key, []).append(elapsed_ms)

    def _finish_profile(self, start: float, segments: Dict[str, float]) -> None:
        if not self._profile:
            return
        total_ms = (time.perf_counter() - start) * 1e3
        self._profile_act_times.append(total_ms)
        if total_ms >= self._profile_slow_ms:
            detail = ", ".join(f"{key}={value:.2f}ms" for key, value in sorted(segments.items()))
            logger.warning("[%s] slow YamLeaderAgent.act %.2fms (%s)", self.robot_name, total_ms, detail)
        now = time.perf_counter()
        if now - self._profile_last_log < self._profile_log_interval:
            return
        self._profile_last_log = now
        lines = [f"[{self.robot_name}] YamLeaderAgent profile over {self._profile_log_interval:.1f}s"]
        if self._profile_act_times:
            lines.append(self._format_profile_line("act_total", self._profile_act_times))
            self._profile_act_times.clear()
        for key in sorted(self._profile_accum):
            values = self._profile_accum[key]
            if values:
                lines.append(self._format_profile_line(key, values))
                values.clear()
        logger.info("\n".join(lines))

    @staticmethod
    def _format_profile_line(key: str, values: List[float]) -> str:
        ordered = sorted(values)
        return (
            f"  {key:<30s}: n={len(values):4d} mean={sum(values) / len(values):7.2f}ms "
            f"p50={ordered[len(ordered) // 2]:7.2f}ms "
            f"p95={ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]:7.2f}ms "
            f"max={ordered[-1]:7.2f}ms"
        )

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Bilateral control helpers
    # ------------------------------------------------------------------ #

    def _apply_feedback_gains_once(self) -> None:
        """Apply scaled feedback gains once when feedback first activates."""
        if self._feedback_gains_applied:
            return
        if self._original_kp is None:
            return
        scaled_kp = self._original_kp * self.bilateral_kp
        scaled_kd = np.zeros(NUM_ARM_JOINTS)
        self.robot.update_kp_kd(kp=scaled_kp, kd=scaled_kd)
        self._feedback_gains_applied = True

    def _restore_feedback_gains_if_needed(self) -> None:
        """Restore original gains if feedback gains were previously applied."""
        if self._feedback_gains_applied and self._original_kp is not None:
            self.robot.update_kp_kd(kp=self._original_kp, kd=self._kd)
            self._feedback_gains_applied = False

    def _extract_follower_pos(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract follower arm joint positions (radians) from observations."""
        robot_obs = obs.get(self.robot_name)
        if robot_obs is None:
            return None
        return robot_obs.get("joint_pos")

    def _send_bilateral_feedback(self, follower_pos_rad: np.ndarray) -> None:
        """Write follower positions to leader motors as position targets.

        The inverse coordinate transform is applied:
            leader_cmd = (follower_pos - offsets) * signs

        Feedback gains are applied once (after warmup), not every tick.
        """
        leader_cmd_rad = (follower_pos_rad - self.joint_offsets) * self.joint_signs
        self._apply_feedback_gains_once()
        self.robot.command_joint_pos(leader_cmd_rad)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the robot connection."""
        self._restore_feedback_gains_if_needed()
        if hasattr(self.robot, 'close'):
            self.robot.close()
        logger.info(f"YamLeaderAgent {self.robot_name} closed")

    def reset(self) -> None:
        """Reset the agent state."""
        self._step = 0

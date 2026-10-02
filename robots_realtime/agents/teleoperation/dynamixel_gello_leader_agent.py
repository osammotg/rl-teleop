"""Agent that reads joint positions from a Dynamixel-backed GELLO leader arm."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import dynamixel_sdk
import numpy as np

from dm_env.specs import Array

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6
DEFAULT_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

# Dynamixel X-series (Protocol 2.0) control-table addresses used for the
# optional gripper "spring". This mirrors yams-robot-server's Dynamixel GELLO
# setup: put the gripper motor in Current-based Position Control mode, cap its
# output torque via Current_Limit, enable torque, and command the open position.
# The motor gently holds the trigger open and the operator can overpower it to
# squeeze; on release it springs back to open.
_ADDR_OPERATING_MODE = 11   # EEPROM (torque must be off to write)
_ADDR_CURRENT_LIMIT = 38    # EEPROM; output-torque cap, overpowerable by hand
_ADDR_TORQUE_ENABLE = 64
_ADDR_GOAL_POSITION = 116
_OP_CURRENT_BASED_POSITION = 5
_TORQUE_OFF = 0
_TORQUE_ON = 1


class _DynamixelPositionReader:
    """Reads positions from Dynamixel motors via serial."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        motor_ids: List[int],
        protocol_version: float,
        present_position_addr: int,
        present_position_len: int,
        num_read_retries: int = 3,
    ) -> None:
        self._port = port
        self._baudrate = int(baudrate)
        self._motor_ids = list(motor_ids)
        self._present_position_addr = int(present_position_addr)
        self._present_position_len = int(present_position_len)
        self._num_read_retries = num_read_retries
        self._last_read_time = 0.0

        self._port_handler = dynamixel_sdk.PortHandler(port)
        self._packet_handler = dynamixel_sdk.PacketHandler(float(protocol_version))
        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port {port}")
        if not self._port_handler.setBaudRate(self._baudrate):
            self._port_handler.closePort()
            raise RuntimeError(f"Failed to set Dynamixel baudrate {self._baudrate} on {port}")

        self._group_sync_read = dynamixel_sdk.GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            self._present_position_addr,
            self._present_position_len,
        )
        for motor_id in self._motor_ids:
            if not self._group_sync_read.addParam(int(motor_id)):
                self._port_handler.closePort()
                raise RuntimeError(f"Failed to add Dynamixel id={motor_id} to sync read group on {port}")

    def _format_comm_result(self, comm_result: int) -> str:
        try:
            message = self._packet_handler.getTxRxResult(comm_result)
        except Exception:
            message = ""
        return f"{comm_result} ({message})" if message else str(comm_result)

    def ping_ids(self, motor_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Ping motor IDs and return per-ID communication status."""
        results: List[Dict[str, Any]] = []
        target_ids = self._motor_ids if motor_ids is None else motor_ids
        for motor_id in target_ids:
            try:
                model_number, comm_result, error = self._packet_handler.ping(self._port_handler, int(motor_id))
            except Exception as exc:
                results.append(
                    {
                        "motor_id": int(motor_id),
                        "ok": False,
                        "model_number": None,
                        "comm_result": None,
                        "comm_result_text": "",
                        "error": None,
                        "error_text": "",
                        "exception": str(exc),
                    }
                )
                continue

            error_text = ""
            if error:
                try:
                    error_text = self._packet_handler.getRxPacketError(error)
                except Exception:
                    error_text = ""
            results.append(
                {
                    "motor_id": int(motor_id),
                    "ok": comm_result == dynamixel_sdk.COMM_SUCCESS and error == 0,
                    "model_number": int(model_number) if model_number is not None else None,
                    "comm_result": int(comm_result),
                    "comm_result_text": self._format_comm_result(int(comm_result)),
                    "error": int(error),
                    "error_text": error_text,
                }
            )
        return results

    def get_positions(self) -> np.ndarray:
        last_comm_result = 0
        last_missing_id: Optional[int] = None

        for attempt in range(self._num_read_retries + 1):
            comm_result = self._group_sync_read.fastSyncRead()
            last_comm_result = comm_result
            if comm_result != dynamixel_sdk.COMM_SUCCESS:
                if attempt < self._num_read_retries:
                    time.sleep(0.001)
                    continue
                break

            positions: List[int] = []
            missing_id: Optional[int] = None
            for motor_id in self._motor_ids:
                if not self._group_sync_read.isAvailable(
                    int(motor_id),
                    self._present_position_addr,
                    self._present_position_len,
                ):
                    missing_id = int(motor_id)
                    break
                value = self._group_sync_read.getData(
                    int(motor_id),
                    self._present_position_addr,
                    self._present_position_len,
                )
                if value >= 2**31:
                    value -= 2**32
                positions.append(value)

            if missing_id is None:
                self._last_read_time = time.time()
                return np.array(positions, dtype=np.float64)

            last_missing_id = missing_id
            if attempt < self._num_read_retries:
                time.sleep(0.001)
                continue
            break

        if last_missing_id is not None:
            raise RuntimeError(
                f"Dynamixel read failed for id={last_missing_id} after {self._num_read_retries + 1} attempts: "
                f"comm_result={last_comm_result}, error=0"
            )
        raise RuntimeError(
            f"Dynamixel sync read failed after {self._num_read_retries + 1} attempts: "
            f"comm_result={self._format_comm_result(last_comm_result)}"
        )

    def seconds_since_last_read(self) -> float:
        if self._last_read_time <= 0.0:
            return float("inf")
        return time.time() - self._last_read_time

    def enable_gripper_spring(self, gripper_id: int, open_ticks: int, current_limit: int) -> None:
        """Hold the gripper motor at ``open_ticks`` with a capped, overpowerable torque.

        Puts the gripper motor in Current-based Position Control mode and bounds
        its output via Current_Limit so the trigger is easy to squeeze and
        gently springs back to the open position when released. Only the gripper
        motor is touched; arm joints stay free.
        """
        pk, ph = self._packet_handler, self._port_handler

        def _w(fn, addr, value, what):
            comm, err = fn(ph, int(gripper_id), addr, int(value))
            if comm != dynamixel_sdk.COMM_SUCCESS or err != 0:
                logger.warning(
                    "Gripper spring: failed to write %s on id=%d (comm=%d, err=%d)",
                    what, gripper_id, comm, err,
                )

        # Operating mode and current limit live in EEPROM — torque must be off to write.
        _w(pk.write1ByteTxRx, _ADDR_TORQUE_ENABLE, _TORQUE_OFF, "torque-off")
        _w(pk.write1ByteTxRx, _ADDR_OPERATING_MODE, _OP_CURRENT_BASED_POSITION, "operating-mode")
        _w(pk.write2ByteTxRx, _ADDR_CURRENT_LIMIT, current_limit, "current-limit")
        _w(pk.write1ByteTxRx, _ADDR_TORQUE_ENABLE, _TORQUE_ON, "torque-on")
        _w(pk.write4ByteTxRx, _ADDR_GOAL_POSITION, open_ticks, "goal-position")

    def release_gripper(self, gripper_id: int) -> None:
        """Disable torque on the gripper motor (best-effort)."""
        try:
            self._packet_handler.write1ByteTxRx(
                self._port_handler, int(gripper_id), _ADDR_TORQUE_ENABLE, _TORQUE_OFF
            )
        except Exception:  # teardown must not raise
            pass

    def close(self) -> None:
        self._port_handler.closePort()


class DynamixelGelloLeaderAgent(Agent):
    """Read-only teleoperation agent for a Dynamixel GELLO leader."""

    use_joint_state_as_action: bool = False

    def __init__(
        self,
        port: str = "/dev/leader-left",
        robot_name: str = "left",
        motor_ids: Optional[List[int]] = None,
        baudrate: int = 1_000_000,
        protocol_version: float = 2.0,
        present_position_addr: int = 132,
        present_position_len: int = 4,
        ticks_per_rev: int = 4096,
        joint_signs: Optional[List[int]] = None,
        joint_offsets_ticks: Optional[List[float]] = None,
        joint_scales: Optional[List[float]] = None,
        joint_offsets_rad: Optional[List[float]] = None,
        include_gripper: bool = False,
        gripper_index: int = 6,
        gripper_open_ticks: int = 2280,
        gripper_closed_ticks: int = 1670,
        gripper_scale: float = 1.0,
        gripper_range_ticks: Optional[int] = None,
        stale_timeout_s: float = 1.0,
        max_delta_rad: float = 3.14,
        num_read_retries: int = 3,
        gripper_spring: bool = False,
        gripper_spring_current_limit: int = 100,
        reader: Any = None,
    ) -> None:
        self.port = port
        self.robot_name = robot_name
        self.motor_ids = list(DEFAULT_MOTOR_IDS if motor_ids is None else motor_ids)
        self.ticks_per_rev = int(ticks_per_rev)
        self.include_gripper = bool(include_gripper)
        self.gripper_index = int(gripper_index)
        self.gripper_open_ticks = int(gripper_open_ticks)
        self.gripper_closed_ticks = int(gripper_closed_ticks)
        self.gripper_scale = float(gripper_scale)
        self.gripper_range_ticks = None if gripper_range_ticks is None else int(gripper_range_ticks)
        self.stale_timeout_s = float(stale_timeout_s)
        self.max_delta_rad = float(max_delta_rad)
        self.num_read_retries = num_read_retries
        self.gripper_spring = bool(gripper_spring)
        self.gripper_spring_current_limit = int(gripper_spring_current_limit)
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_ticks = np.array(joint_offsets_ticks or [0.0] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_scales = np.array(joint_scales or [1.0] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_rad = np.array(joint_offsets_rad or [0.0] * NUM_ARM_JOINTS, dtype=np.float64)
        if self.joint_signs.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"joint_signs must have length {NUM_ARM_JOINTS}, got {self.joint_signs.shape}")
        if self.joint_offsets_ticks.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_offsets_ticks must have length {NUM_ARM_JOINTS}, got {self.joint_offsets_ticks.shape}"
            )
        if self.joint_scales.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"joint_scales must have length {NUM_ARM_JOINTS}, got {self.joint_scales.shape}")
        if self.joint_offsets_rad.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_offsets_rad must have length {NUM_ARM_JOINTS}, got {self.joint_offsets_rad.shape}"
            )
        if self.ticks_per_rev <= 0:
            raise ValueError("ticks_per_rev must be positive")
        if self.include_gripper and self.gripper_open_ticks == self.gripper_closed_ticks:
            raise ValueError("gripper_open_ticks and gripper_closed_ticks must differ when include_gripper=True")
        if self.include_gripper and self.gripper_range_ticks is not None and self.gripper_range_ticks <= 0:
            raise ValueError("gripper_range_ticks must be positive when provided")

        self._ticks_to_rad = (2.0 * np.pi) / self.ticks_per_rev
        self._last_stale_log = 0.0
        self._last_pos = np.zeros(NUM_ARM_JOINTS + (1 if self.include_gripper else 0), dtype=np.float32)

        self._reader = reader or _DynamixelPositionReader(
            port=port,
            baudrate=baudrate,
            motor_ids=self.motor_ids,
            protocol_version=protocol_version,
            present_position_addr=present_position_addr,
            present_position_len=present_position_len,
            num_read_retries=self.num_read_retries,
        )

        # Optional gripper trigger spring: actively hold the gripper motor at the
        # open position with a capped current so the trigger pops back out when
        # released (the rest of the arm stays read-only).
        self._gripper_motor_id: Optional[int] = None
        if self.include_gripper and self.gripper_spring and hasattr(self._reader, "enable_gripper_spring"):
            if self.gripper_index >= len(self.motor_ids):
                raise ValueError(
                    f"gripper_index {self.gripper_index} out of range for motor_ids {self.motor_ids}"
                )
            self._gripper_motor_id = int(self.motor_ids[self.gripper_index])
            self._reader.enable_gripper_spring(
                self._gripper_motor_id, self.gripper_open_ticks, self.gripper_spring_current_limit
            )
            logger.info(
                "DynamixelGelloLeaderAgent[%s]: gripper spring enabled on id=%d "
                "(open_ticks=%d, current_limit=%d)",
                self.port, self._gripper_motor_id, self.gripper_open_ticks, self.gripper_spring_current_limit,
            )

    def _map_gripper(self, gripper_ticks: float) -> float:
        if self.gripper_range_ticks is not None:
            gripper_rad_raw = gripper_ticks * self._ticks_to_rad
            gripper_range_rad = self.gripper_range_ticks * self._ticks_to_rad
            return 1.0 - float(np.clip(abs(gripper_rad_raw) / gripper_range_rad, 0.0, 1.0))

        travel = float(self.gripper_open_ticks - self.gripper_closed_ticks)
        normalized = (float(gripper_ticks) - self.gripper_closed_ticks) / travel * self.gripper_scale
        return float(np.clip(normalized, 0.0, 1.0))

    def _map_arm_ticks(self, arm_ticks: np.ndarray) -> np.ndarray:
        calibrated_ticks = (arm_ticks.astype(np.float64) + self.joint_offsets_ticks) * self.joint_scales
        arm_rad = calibrated_ticks * self._ticks_to_rad - np.pi
        arm_rad = (arm_rad + np.pi) % (2.0 * np.pi) - np.pi
        return self.joint_signs * arm_rad + self.joint_offsets_rad

    def _positions_to_action(self, ticks: np.ndarray) -> np.ndarray:
        if ticks.shape[0] < NUM_ARM_JOINTS:
            raise RuntimeError(f"Expected at least {NUM_ARM_JOINTS} Dynamixel positions, got {ticks.shape[0]}")
        arm_rad = self._map_arm_ticks(ticks[:NUM_ARM_JOINTS])
        if self.include_gripper:
            if ticks.shape[0] <= self.gripper_index:
                raise RuntimeError(f"Missing gripper position at index {self.gripper_index}")
            pos = np.concatenate([arm_rad, [self._map_gripper(float(ticks[self.gripper_index]))]])
        else:
            pos = arm_rad
        return pos.astype(np.float32)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ticks = np.asarray(self._reader.get_positions(), dtype=np.float64)
            pos = self._positions_to_action(ticks)
            if np.all(np.isfinite(pos)):
                self._last_pos = pos
            else:
                logger.warning("DynamixelGelloLeaderAgent[%s]: non-finite action from reader", self.port)
                pos = self._last_pos
        except RuntimeError as exc:
            logger.warning("DynamixelGelloLeaderAgent[%s]: read failed, reusing previous action: %s", self.port, exc)
            pos = self._last_pos

        if self.stale_timeout_s > 0:
            stale_s = self._reader.seconds_since_last_read()
            now = time.monotonic()
            if stale_s > self.stale_timeout_s and (now - self._last_stale_log) > 5.0:
                logger.warning(
                    "DynamixelGelloLeaderAgent[%s]: stale read, no Dynamixel reads in %.2fs",
                    self.port,
                    stale_s,
                )
                self._last_stale_log = now

        return {self.robot_name: {"pos": pos.astype(np.float32)}}

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    def close(self) -> None:
        if self._gripper_motor_id is not None and hasattr(self._reader, "release_gripper"):
            self._reader.release_gripper(self._gripper_motor_id)
        self._reader.close()

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read Dynamixel positions via CLI.")
    parser.add_argument("--port", type=str, default="/dev/leader-left", help="Serial port")
    parser.add_argument("--motor-ids", type=str, default="1,2,3,4,5,6,7", help="Comma-separated IDs")
    parser.add_argument("--baudrate", type=int, default=1000000, help="Baud rate")
    parser.add_argument("--protocol-version", type=float, default=2.0, help="Protocol version")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--discover", action="store_true", help="Auto-discovery mode")
    parser.add_argument("--scan-id-start", type=int, help="First Dynamixel ID to ping when discovering")
    parser.add_argument("--scan-id-end", type=int, help="Last Dynamixel ID to ping when discovering")
    parser.add_argument("--num-read-retries", type=int, default=3, help="Number of retries per motor read")
    args = parser.parse_args()

    ids = [int(x.strip()) for x in args.motor_ids.split(",")]
    if (args.scan_id_start is None) != (args.scan_id_end is None):
        parser.error("--scan-id-start and --scan-id-end must be provided together")
    if args.scan_id_start is not None and args.scan_id_start > args.scan_id_end:
        parser.error("--scan-id-start must be <= --scan-id-end")

    results = {
        "port": args.port,
        "baudrate": args.baudrate,
        "motor_ids": ids,
        "samples": args.samples,
        "samples_collected": 0,
        "all_finite": True,
        "positions_rad": [],
        "last_positions_rad": [],
        "timestamps": [],
        "errors": [],
        "discovery": [],
        "discovered_ids": [],
        "scan_id_start": args.scan_id_start,
        "scan_id_end": args.scan_id_end,
    }

    exit_code = 0
    try:
        reader = _DynamixelPositionReader(
            port=args.port,
            baudrate=args.baudrate,
            motor_ids=ids,
            protocol_version=args.protocol_version,
            present_position_addr=132,
            present_position_len=4,
            num_read_retries=args.num_read_retries,
        )

        if args.discover:
            discover_ids = (
                list(range(args.scan_id_start, args.scan_id_end + 1))
                if args.scan_id_start is not None
                else ids
            )
            discovery = reader.ping_ids(discover_ids)
            results["discovery"] = discovery
            results["discovered_ids"] = [item["motor_id"] for item in discovery if item["ok"]]

        samples_to_collect = args.samples
        max_attempts = max(args.samples * 5, args.samples)
        attempts = 0
        
        while results["samples_collected"] < samples_to_collect and attempts < max_attempts:
            attempts += 1
            try:
                ticks = reader.get_positions()
                rads = (ticks / 4096.0) * (2.0 * np.pi)

                results["positions_rad"].append(rads.tolist())
                results["last_positions_rad"] = rads.tolist()
                results["timestamps"].append(time.time())
                results["samples_collected"] += 1

                if not np.all(np.isfinite(rads)):
                    results["all_finite"] = False
            except RuntimeError as exc:
                results["errors"].append(str(exc))
                # If we've already collected enough finite samples, we don't mark all_finite as False
                if results["samples_collected"] < samples_to_collect:
                    # We don't fail the whole thing unless it's a fatal error, 
                    # but here we just continue to next sample.
                    # The requirement says "CLI should set all_finite: true if it ultimately collected 
                    # the requested number of finite samples, even if transient errors were recovered".
                    # Our reader handles retries, so if it reaches here, it's NOT a transient error 
                    # that was recovered. It's a persistent failure for THIS sample.
                    pass
            except Exception as exc:
                results["errors"].append(f"Unexpected error: {exc}")
                results["all_finite"] = False
                exit_code = 1
                break

            if args.samples > 1:
                time.sleep(0.01)

        if results["samples_collected"] < samples_to_collect:
            if results["samples_collected"] == 0:
                exit_code = 1
            else:
                # We partially collected some samples but not enough.
                # If we reached max_attempts, it's a failure.
                if attempts >= max_attempts:
                    exit_code = 1
        
        # Check if we actually collected enough finite samples
        if results["samples_collected"] < samples_to_collect:
             results["all_finite"] = False

        reader.close()

    except Exception as exc:
        results["errors"].append(f"Setup error: {exc}")
        results["all_finite"] = False
        exit_code = 1

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

    sys.exit(exit_code)

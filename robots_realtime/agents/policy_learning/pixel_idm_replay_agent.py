from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_NPZ_PATH = "PIXEL_IDM_INFERENCE/full_inference_actions.npz"
DEFAULT_LEFT_LIMITS_PATH = "robot_configs/yam/xdof_hq/left.yaml"
DEFAULT_RIGHT_LIMITS_PATH = "robot_configs/yam/xdof_hq/right.yaml"


@dataclass(frozen=True)
class PixelIDMValidationReport:
    npz_path: str
    array_key: str
    window_idx: int
    source_shape: tuple[int, ...]
    selected_shape: tuple[int, ...]
    first_state: np.ndarray
    last_state: np.ndarray
    min_per_dim: np.ndarray
    max_per_dim: np.ndarray
    max_frame_delta: np.ndarray
    clamped_arm_values: int = 0
    clamped_gripper_values: int = 0
    handoff_left_distance: float | None = None
    handoff_right_distance: float | None = None

    def format(self) -> str:
        lines = [
            "[PixelIDMReplayAgent] validation report",
            f"  npz_path: {self.npz_path}",
            f"  array_key: {self.array_key}",
            f"  window_idx: {self.window_idx}",
            f"  source_shape: {self.source_shape}",
            f"  selected_shape: {self.selected_shape}",
            f"  clamped_arm_values: {self.clamped_arm_values}",
            f"  clamped_gripper_values: {self.clamped_gripper_values}",
            f"  first_state: {np.array2string(self.first_state, precision=5, suppress_small=False)}",
            f"  last_state: {np.array2string(self.last_state, precision=5, suppress_small=False)}",
            f"  min_per_dim: {np.array2string(self.min_per_dim, precision=5, suppress_small=False)}",
            f"  max_per_dim: {np.array2string(self.max_per_dim, precision=5, suppress_small=False)}",
            f"  max_frame_delta: {np.array2string(self.max_frame_delta, precision=5, suppress_small=False)}",
        ]
        if self.handoff_left_distance is not None:
            lines.append(f"  handoff_left_distance: {self.handoff_left_distance:.6f}")
        if self.handoff_right_distance is not None:
            lines.append(f"  handoff_right_distance: {self.handoff_right_distance:.6f}")
        return "\n".join(lines)

    def with_handoff(self, left_distance: float, right_distance: float) -> PixelIDMValidationReport:
        return PixelIDMValidationReport(
            npz_path=self.npz_path,
            array_key=self.array_key,
            window_idx=self.window_idx,
            source_shape=self.source_shape,
            selected_shape=self.selected_shape,
            first_state=self.first_state,
            last_state=self.last_state,
            min_per_dim=self.min_per_dim,
            max_per_dim=self.max_per_dim,
            max_frame_delta=self.max_frame_delta,
            clamped_arm_values=self.clamped_arm_values,
            clamped_gripper_values=self.clamped_gripper_values,
            handoff_left_distance=left_distance,
            handoff_right_distance=right_distance,
        )


def _load_arm_limits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    limits = np.asarray(cfg["joint_limits"], dtype=np.float64)
    if limits.shape != (6, 2):
        raise ValueError(f"{path} must contain joint_limits with shape (6, 2), got {limits.shape}")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError(f"{path} has invalid joint_limits")
    return limits[:, 0], limits[:, 1]


def _validate_arm_limits(
    name: str,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    under = lower[None, :] - values
    over = values - upper[None, :]
    max_under = float(np.maximum(under, 0.0).max(initial=0.0))
    max_over = float(np.maximum(over, 0.0).max(initial=0.0))
    if max(max_under, max_over) > tolerance:
        raise ValueError(
            f"{name} arm joints exceed configured limits "
            f"(max_under={max_under:.6g}, max_over={max_over:.6g}, tolerance={tolerance:.6g})"
        )
    clamped = np.clip(values, lower[None, :], upper[None, :])
    changed = int(np.count_nonzero(~np.isclose(clamped, values, rtol=0.0, atol=0.0)))
    return clamped, changed


def interpolate_action_window(window: np.ndarray, *, source_hz: float, command_hz: float) -> np.ndarray:
    """Linearly upsample source action frames onto the command-rate replay grid."""
    if source_hz <= 0:
        raise ValueError(f"source_hz must be > 0, got {source_hz}")
    if command_hz <= 0:
        raise ValueError(f"command_hz must be > 0, got {command_hz}")
    if window.shape[0] < 2:
        return window.copy()
    last_idx = window.shape[0] - 1
    n_ticks = int(np.floor((last_idx / source_hz) * command_hz)) + 1
    ticks = np.arange(max(1, n_ticks), dtype=np.float64)
    frame_pos = (ticks / command_hz) * source_hz
    frame_pos = np.minimum(frame_pos, float(last_idx))
    lo = np.floor(frame_pos).astype(np.int64)
    hi = np.minimum(lo + 1, last_idx)
    alpha = (frame_pos - lo).astype(np.float32)[:, None]
    return ((1.0 - alpha) * window[lo] + alpha * window[hi]).astype(np.float32)


def load_and_validate_pixel_idm_window(
    npz_path: str | Path = DEFAULT_NPZ_PATH,
    window_idx: int = 0,
    array_key: str = "predicted_actions",
    left_limits_path: str | Path = DEFAULT_LEFT_LIMITS_PATH,
    right_limits_path: str | Path = DEFAULT_RIGHT_LIMITS_PATH,
    limit_tolerance: float = 1e-6,
    gripper_tolerance: float = 1e-6,
    gripper_clamp_tolerance: float = 2e-2,
    max_arm_frame_delta_rad: float = 0.5,
    max_gripper_frame_delta: float = 0.25,
    source_hz: float | None = None,
    command_hz: float | None = None,
) -> tuple[np.ndarray, PixelIDMValidationReport]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as data:
        if array_key not in data:
            raise ValueError(f"{npz_path} does not contain required array {array_key!r}")
        actions = np.asarray(data[array_key])

    if actions.ndim != 3 or actions.shape[-1] != 14:
        raise ValueError(f"{array_key!r} must have shape (N, T, 14), got {actions.shape}")
    if not 0 <= int(window_idx) < actions.shape[0]:
        raise IndexError(f"window_idx={window_idx} is out of range for N={actions.shape[0]}")

    return validate_pixel_idm_window(
        actions[int(window_idx)],
        source_name=str(npz_path),
        array_key=array_key,
        window_idx=window_idx,
        source_shape=tuple(actions.shape),
        left_limits_path=left_limits_path,
        right_limits_path=right_limits_path,
        limit_tolerance=limit_tolerance,
        gripper_tolerance=gripper_tolerance,
        gripper_clamp_tolerance=gripper_clamp_tolerance,
        max_arm_frame_delta_rad=max_arm_frame_delta_rad,
        max_gripper_frame_delta=max_gripper_frame_delta,
        source_hz=source_hz,
        command_hz=command_hz,
    )


def validate_pixel_idm_window(
    window: np.ndarray,
    *,
    source_name: str = "<memory>",
    array_key: str = "predicted_actions",
    window_idx: int = 0,
    source_shape: tuple[int, ...] | None = None,
    left_limits_path: str | Path = DEFAULT_LEFT_LIMITS_PATH,
    right_limits_path: str | Path = DEFAULT_RIGHT_LIMITS_PATH,
    limit_tolerance: float = 1e-6,
    gripper_tolerance: float = 1e-6,
    gripper_clamp_tolerance: float = 2e-2,
    max_arm_frame_delta_rad: float = 0.5,
    max_gripper_frame_delta: float = 0.25,
    source_hz: float | None = None,
    command_hz: float | None = None,
) -> tuple[np.ndarray, PixelIDMValidationReport]:
    window = np.array(window, dtype=np.float32, copy=True)
    if window.ndim != 2 or window.shape[1] != 14:
        raise ValueError(f"{array_key!r} window must have shape (T, 14), got {window.shape}")
    if window.shape[0] < 2:
        raise ValueError(f"selected window must have T >= 2, got shape {window.shape}")
    if not np.all(np.isfinite(window)):
        raise ValueError("selected window contains NaN or Inf")

    left_gripper = window[:, 6]
    right_gripper = window[:, 13]
    grip_min = float(min(left_gripper.min(), right_gripper.min()))
    grip_max = float(max(left_gripper.max(), right_gripper.max()))
    if gripper_clamp_tolerance < gripper_tolerance:
        raise ValueError(
            f"gripper_clamp_tolerance ({gripper_clamp_tolerance}) must be >= gripper_tolerance ({gripper_tolerance})"
        )
    if grip_min < -gripper_clamp_tolerance or grip_max > 1.0 + gripper_clamp_tolerance:
        raise ValueError(
            "gripper columns 6 and 13 must be normalized command-space values in [0, 1] "
            f"(observed min={grip_min:.6g}, max={grip_max:.6g}, "
            f"clamp_tolerance={gripper_clamp_tolerance:.6g})"
        )
    gripper_before = window[:, [6, 13]].copy()
    window[:, 6] = np.clip(window[:, 6], 0.0, 1.0)
    window[:, 13] = np.clip(window[:, 13], 0.0, 1.0)
    clamped_gripper_values = int(np.count_nonzero(~np.isclose(window[:, [6, 13]], gripper_before, rtol=0.0, atol=0.0)))

    left_lower, left_upper = _load_arm_limits(left_limits_path)
    right_lower, right_upper = _load_arm_limits(right_limits_path)
    left, left_clamped = _validate_arm_limits("left", window[:, :6], left_lower, left_upper, limit_tolerance)
    right, right_clamped = _validate_arm_limits("right", window[:, 7:13], right_lower, right_upper, limit_tolerance)
    window[:, :6] = left
    window[:, 7:13] = right

    continuity_window = window
    if source_hz is not None and command_hz is not None:
        continuity_window = interpolate_action_window(window, source_hz=source_hz, command_hz=command_hz)

    frame_delta = np.abs(np.diff(continuity_window, axis=0))
    max_delta = frame_delta.max(axis=0)
    max_arm_delta = float(max(max_delta[:6].max(initial=0.0), max_delta[7:13].max(initial=0.0)))
    max_grip_delta = float(max(max_delta[6], max_delta[13]))
    if max_arm_delta > max_arm_frame_delta_rad:
        raise ValueError(
            f"selected window has discontinuous arm commands: max frame delta {max_arm_delta:.6g} rad "
            f"> {max_arm_frame_delta_rad:.6g} rad"
        )
    if max_grip_delta > max_gripper_frame_delta:
        raise ValueError(
            f"selected window has discontinuous gripper commands: max frame delta {max_grip_delta:.6g} "
            f"> {max_gripper_frame_delta:.6g}"
        )

    report = PixelIDMValidationReport(
        npz_path=source_name,
        array_key=array_key,
        window_idx=int(window_idx),
        source_shape=tuple(source_shape) if source_shape is not None else tuple(window.shape),
        selected_shape=tuple(window.shape),
        first_state=window[0].copy(),
        last_state=window[-1].copy(),
        min_per_dim=window.min(axis=0),
        max_per_dim=window.max(axis=0),
        max_frame_delta=max_delta,
        clamped_arm_values=left_clamped + right_clamped,
        clamped_gripper_values=clamped_gripper_values,
    )
    return window, report


def command_state_from_observation(obs: dict[str, Any]) -> np.ndarray | None:
    joint_pos = obs.get("joint_pos")
    if joint_pos is None:
        return None
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    if joint_pos.shape == (7,):
        return joint_pos.copy()
    if joint_pos.shape != (6,):
        return None

    gripper_pos = obs.get("gripper_pos")
    if gripper_pos is None:
        return None
    gripper_pos = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
    if gripper_pos.shape[0] != 1:
        return None
    return np.concatenate([joint_pos, gripper_pos]).astype(np.float32, copy=False)


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


class PixelIDMReplayAgent:
    use_joint_state_as_action = False

    def __init__(
        self,
        npz_path: str = DEFAULT_NPZ_PATH,
        window_idx: int = 0,
        array_key: str = "predicted_actions",
        source_hz: float = 10.0,
        command_hz: float = 100.0,
        handoff_duration_s: float = 3.0,
        left_limits_path: str = DEFAULT_LEFT_LIMITS_PATH,
        right_limits_path: str = DEFAULT_RIGHT_LIMITS_PATH,
        validate_only: bool = False,
        limit_tolerance: float = 1e-6,
        gripper_tolerance: float = 1e-6,
        gripper_clamp_tolerance: float = 2e-2,
        max_arm_frame_delta_rad: float = 0.5,
        max_gripper_frame_delta: float = 0.25,
    ) -> None:
        if source_hz <= 0:
            raise ValueError(f"source_hz must be > 0, got {source_hz}")
        if command_hz <= 0:
            raise ValueError(f"command_hz must be > 0, got {command_hz}")
        if handoff_duration_s < 0:
            raise ValueError(f"handoff_duration_s must be >= 0, got {handoff_duration_s}")

        self.source_hz = float(source_hz)
        self.command_hz = float(command_hz)
        self.handoff_duration_s = float(handoff_duration_s)
        self.validate_only = bool(validate_only)
        self.window, self.report = load_and_validate_pixel_idm_window(
            npz_path=npz_path,
            window_idx=window_idx,
            array_key=array_key,
            left_limits_path=left_limits_path,
            right_limits_path=right_limits_path,
            limit_tolerance=limit_tolerance,
            gripper_tolerance=gripper_tolerance,
            gripper_clamp_tolerance=gripper_clamp_tolerance,
            max_arm_frame_delta_rad=max_arm_frame_delta_rad,
            max_gripper_frame_delta=max_gripper_frame_delta,
            source_hz=self.source_hz,
            command_hz=self.command_hz,
        )
        print(self.report.format(), flush=True)

        self._handoff_ticks = round(self.handoff_duration_s * self.command_hz)
        self._tick = 0
        self._live_left: np.ndarray | None = None
        self._live_right: np.ndarray | None = None
        self._ready = False
        self._printed_validate_only = False

    def reset(self) -> None:
        self._tick = 0
        self._live_left = None
        self._live_right = None
        self._ready = False
        self._printed_validate_only = False

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._ready:
            left_obs = obs.get("left")
            right_obs = obs.get("right")
            if not isinstance(left_obs, dict) or not isinstance(right_obs, dict):
                return {}
            left = command_state_from_observation(left_obs)
            right = command_state_from_observation(right_obs)
            if left is None or right is None:
                return {}
            self._live_left = left
            self._live_right = right
            self._ready = True
            first_left, first_right = self.window[0, :7], self.window[0, 7:]
            report = self.report.with_handoff(
                left_distance=float(np.linalg.norm(first_left - left)),
                right_distance=float(np.linalg.norm(first_right - right)),
            )
            print(report.format(), flush=True)

        if self.validate_only:
            if not self._printed_validate_only:
                print("[PixelIDMReplayAgent] validate_only=true; no commands will be published", flush=True)
                self._printed_validate_only = True
            return {}

        left, right = self.command_at_tick(self._tick)
        chunk = self._chunk_at_tick(self._tick)
        self._tick += 1
        return {
            "left": {"pos": left.astype(np.float32, copy=False)},
            "right": {"pos": right.astype(np.float32, copy=False)},
            "_chunk": chunk,
        }

    def command_at_tick(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        if self._live_left is None or self._live_right is None:
            raise RuntimeError("live left/right observations are required before computing commands")

        first_left = self.window[0, :7]
        first_right = self.window[0, 7:]
        if self._handoff_ticks > 0 and tick < self._handoff_ticks:
            alpha = smoothstep(tick / self._handoff_ticks)
            left = (1.0 - alpha) * self._live_left + alpha * first_left
            right = (1.0 - alpha) * self._live_right + alpha * first_right
            return left.astype(np.float32), right.astype(np.float32)

        replay_tick = max(0, tick - self._handoff_ticks)
        frame_pos = (replay_tick / self.command_hz) * self.source_hz
        last_idx = self.window.shape[0] - 1
        if frame_pos >= last_idx:
            row = self.window[-1]
        else:
            lo = int(np.floor(frame_pos))
            hi = lo + 1
            alpha = frame_pos - lo
            row = (1.0 - alpha) * self.window[lo] + alpha * self.window[hi]
        return row[:7].astype(np.float32), row[7:].astype(np.float32)

    def _chunk_at_tick(self, tick: int) -> dict[str, np.ndarray]:
        start_tick = max(0, tick - self._handoff_ticks)
        frame_pos = (start_tick / self.command_hz) * self.source_hz
        start_frame = min(int(np.floor(frame_pos)), self.window.shape[0] - 1)
        remaining = self.window[start_frame:]
        return {
            "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
            "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a PIXEL IDM replay action window.")
    parser.add_argument("--npz-path", default=DEFAULT_NPZ_PATH)
    parser.add_argument("--window-idx", type=int, default=0)
    parser.add_argument("--array-key", default="predicted_actions")
    parser.add_argument("--left-limits-path", default=DEFAULT_LEFT_LIMITS_PATH)
    parser.add_argument("--right-limits-path", default=DEFAULT_RIGHT_LIMITS_PATH)
    parser.add_argument("--limit-tolerance", type=float, default=1e-6)
    parser.add_argument("--gripper-clamp-tolerance", type=float, default=2e-2)
    parser.add_argument("--source-hz", type=float, default=10.0)
    parser.add_argument("--command-hz", type=float, default=100.0)
    args = parser.parse_args(argv)

    _, report = load_and_validate_pixel_idm_window(
        npz_path=args.npz_path,
        window_idx=args.window_idx,
        array_key=args.array_key,
        left_limits_path=args.left_limits_path,
        right_limits_path=args.right_limits_path,
        limit_tolerance=args.limit_tolerance,
        gripper_clamp_tolerance=args.gripper_clamp_tolerance,
        source_hz=args.source_hz,
        command_hz=args.command_hz,
    )
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

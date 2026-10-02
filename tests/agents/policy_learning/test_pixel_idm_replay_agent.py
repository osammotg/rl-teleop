from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robots_realtime.agents.policy_learning.pixel_idm_replay_agent import (
    DEFAULT_LEFT_LIMITS_PATH,
    DEFAULT_RIGHT_LIMITS_PATH,
    PixelIDMReplayAgent,
    load_and_validate_pixel_idm_window,
    validate_pixel_idm_window,
)


def _write_npz(path: Path, window: np.ndarray, key: str = "predicted_actions") -> Path:
    np.savez(path, **{key: window[None, :, :].astype(np.float32)})
    return path


def _valid_window() -> np.ndarray:
    first = np.array(
        [
            0.10,
            0.20,
            0.20,
            -0.10,
            0.10,
            -0.20,
            0.30,
            0.20,
            0.30,
            0.40,
            -0.20,
            0.20,
            -0.30,
            0.70,
        ],
        dtype=np.float32,
    )
    second = first + np.array(
        [0.01, 0.01, 0.02, 0.01, 0.00, -0.01, 0.05, -0.01, 0.02, 0.01, 0.00, 0.01, 0.01, -0.05],
        dtype=np.float32,
    )
    third = second + np.array(
        [0.02, 0.00, 0.01, -0.02, 0.01, 0.01, 0.05, 0.01, -0.01, 0.02, 0.01, 0.00, -0.01, -0.05],
        dtype=np.float32,
    )
    return np.stack([first, second, third], axis=0)


def _obs(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        "left": {"joint_pos": left[:6].astype(np.float32), "gripper_pos": np.array([left[6]], dtype=np.float32)},
        "right": {"joint_pos": right[:6].astype(np.float32), "gripper_pos": np.array([right[6]], dtype=np.float32)},
    }


def test_rejects_current_pixel_idm_artifact_default_window_with_strict_limits() -> None:
    with pytest.raises(ValueError, match="left arm joints exceed configured limits"):
        load_and_validate_pixel_idm_window(
            "PIXEL_IDM_INFERENCE/full_inference_actions.npz",
            window_idx=0,
        )


def test_accepts_current_pixel_idm_artifact_default_window_with_explicit_tolerance() -> None:
    window, report = load_and_validate_pixel_idm_window(
        "PIXEL_IDM_INFERENCE/full_inference_actions.npz",
        window_idx=0,
        limit_tolerance=5e-2,
    )

    assert report.source_shape == (10, 8, 14)
    assert window.shape == (8, 14)
    assert np.isfinite(window).all()


def test_accepts_in_memory_generated_window_with_same_validation() -> None:
    window, report = validate_pixel_idm_window(
        _valid_window(),
        source_name="<generated>",
        source_shape=_valid_window().shape,
        left_limits_path=DEFAULT_LEFT_LIMITS_PATH,
        right_limits_path=DEFAULT_RIGHT_LIMITS_PATH,
        source_hz=10,
        command_hz=30,
    )

    assert report.npz_path == "<generated>"
    assert report.source_shape == _valid_window().shape
    assert window.shape == (3, 14)


def test_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "actions.npz"
    np.savez(path, target_actions=_valid_window()[None, :, :])

    with pytest.raises(ValueError, match="required array"):
        load_and_validate_pixel_idm_window(path)


def test_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "actions.npz"
    np.savez(path, predicted_actions=np.zeros((3, 14), dtype=np.float32))

    with pytest.raises(ValueError, match=r"shape \(N, T, 14\)"):
        load_and_validate_pixel_idm_window(path)


def test_rejects_nan_or_inf(tmp_path: Path) -> None:
    window = _valid_window()
    window[1, 2] = np.nan
    path = _write_npz(tmp_path / "actions.npz", window)

    with pytest.raises(ValueError, match="NaN or Inf"):
        load_and_validate_pixel_idm_window(path)


def test_rejects_grippers_outside_normalized_command_space(tmp_path: Path) -> None:
    window = _valid_window()
    window[1, 13] = -0.1
    path = _write_npz(tmp_path / "actions.npz", window)

    with pytest.raises(ValueError, match="gripper columns"):
        load_and_validate_pixel_idm_window(path)


def test_rejects_arm_joints_outside_yam_limits(tmp_path: Path) -> None:
    window = _valid_window()
    window[1, 0] = 4.0
    path = _write_npz(tmp_path / "actions.npz", window)

    with pytest.raises(ValueError, match="exceed configured limits"):
        load_and_validate_pixel_idm_window(path)


def test_no_implicit_normalization_is_applied(tmp_path: Path) -> None:
    window = _valid_window()
    window[:, 0] = 4.0
    path = _write_npz(tmp_path / "actions.npz", window)

    with pytest.raises(ValueError, match="exceed configured limits"):
        load_and_validate_pixel_idm_window(path)


def test_handoff_starts_at_live_state_and_ends_at_first_frame(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "actions.npz", _valid_window())
    agent = PixelIDMReplayAgent(
        npz_path=str(path),
        command_hz=10,
        source_hz=10,
        handoff_duration_s=1.0,
        left_limits_path=DEFAULT_LEFT_LIMITS_PATH,
        right_limits_path=DEFAULT_RIGHT_LIMITS_PATH,
    )
    live_left = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    live_right = np.array([0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    assert agent.act(_obs(live_left, live_right))["left"]["pos"].shape == (7,)

    left0, right0 = agent.command_at_tick(0)
    np.testing.assert_allclose(left0, live_left)
    np.testing.assert_allclose(right0, live_right)

    left_end, right_end = agent.command_at_tick(10)
    np.testing.assert_allclose(left_end, _valid_window()[0, :7], atol=1e-6)
    np.testing.assert_allclose(right_end, _valid_window()[0, 7:], atol=1e-6)


def test_replay_interpolation_preserves_source_boundaries(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "actions.npz", _valid_window())
    agent = PixelIDMReplayAgent(
        npz_path=str(path),
        command_hz=100,
        source_hz=10,
        handoff_duration_s=0.0,
    )
    live = np.zeros(7, dtype=np.float32)
    live[6] = 0.5
    agent.act(_obs(live, live))

    for frame_idx in range(_valid_window().shape[0]):
        left, right = agent.command_at_tick(frame_idx * 10)
        np.testing.assert_allclose(left, _valid_window()[frame_idx, :7], atol=1e-6)
        np.testing.assert_allclose(right, _valid_window()[frame_idx, 7:], atol=1e-6)


def test_final_command_is_held_after_replay_completion(tmp_path: Path) -> None:
    window = _valid_window()
    path = _write_npz(tmp_path / "actions.npz", window)
    agent = PixelIDMReplayAgent(npz_path=str(path), command_hz=100, source_hz=10, handoff_duration_s=0.0)
    live = np.zeros(7, dtype=np.float32)
    live[6] = 0.5
    agent.act(_obs(live, live))

    left, right = agent.command_at_tick(10_000)
    np.testing.assert_allclose(left, window[-1, :7], atol=1e-6)
    np.testing.assert_allclose(right, window[-1, 7:], atol=1e-6)


def test_agent_act_returns_bimanual_command_space_and_chunk(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "actions.npz", _valid_window())
    agent = PixelIDMReplayAgent(npz_path=str(path), command_hz=100, source_hz=10, handoff_duration_s=0.0)
    live = np.zeros(7, dtype=np.float32)
    live[6] = 0.5

    action = agent.act(_obs(live, live))

    assert set(action) == {"left", "right", "_chunk"}
    assert action["left"]["pos"].shape == (7,)
    assert action["right"]["pos"].shape == (7,)
    assert action["left"]["pos"].dtype == np.float32
    assert action["right"]["pos"].dtype == np.float32
    assert action["_chunk"]["left"].shape[1] == 7
    assert action["_chunk"]["right"].shape[1] == 7


def test_validate_only_never_publishes_commands(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "actions.npz", _valid_window())
    agent = PixelIDMReplayAgent(npz_path=str(path), validate_only=True)
    live = np.zeros(7, dtype=np.float32)
    live[6] = 0.5

    assert agent.act(_obs(live, live)) == {}

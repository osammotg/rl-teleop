from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from mcap.reader import make_reader

DIM_NAMES = [
    "left/j0",
    "left/j1",
    "left/j2",
    "left/j3",
    "left/j4",
    "left/j5",
    "left/gripper",
    "right/j0",
    "right/j1",
    "right/j2",
    "right/j3",
    "right/j4",
    "right/j5",
    "right/gripper",
]


@dataclass
class ComparisonResult:
    npz_path: str
    array_key: str
    window_idx: int | None
    recordings_root: str
    episodes_used: list[str]
    recorded_10hz_shape: tuple[int, int]
    pixel_shape: tuple[int, ...]
    recorded_min: list[float]
    recorded_max: list[float]
    recorded_p001: list[float]
    recorded_p999: list[float]
    pixel_min: list[float]
    pixel_max: list[float]
    pixel_first: list[float]
    pixel_last: list[float]
    pixel_max_frame_delta: list[float]
    recorded_p999_frame_delta_10hz: list[float]
    recorded_max_frame_delta_10hz: list[float]
    outside_recorded_hard_dims: list[str]
    outside_recorded_p001_p999_dims: list[str]
    pixel_over_recorded_p999_delta_ratio: list[float]
    nearest_recorded_l2_per_pixel_frame: list[float]
    nearest_recorded_l2_min: float
    nearest_recorded_l2_max: float
    nearest_recorded_l2_mean: float
    joint_limit_violations: list[str]
    gripper_range_ok: bool


def read_arm_mcap(path: Path) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    timestamps: list[float] = []
    with open(path, "rb") as f:
        for _schema, channel, msg in make_reader(f).iter_messages():
            if not channel.topic.endswith("/joint_state"):
                continue
            data = json.loads(msg.data)
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            gripper_pos = np.asarray(data["gripper_pos"], dtype=np.float64).reshape(-1)
            if joint_pos.shape != (6,) or gripper_pos.shape != (1,):
                continue
            states.append(np.concatenate([joint_pos, gripper_pos]))
            timestamps.append(msg.log_time / 1e9)
    if not states:
        raise ValueError(f"no joint_state messages found in {path}")
    return np.asarray(timestamps, dtype=np.float64), np.asarray(states, dtype=np.float64)


def interp_states(ts: np.ndarray, states: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid), states.shape[1]), dtype=np.float64)
    for dim in range(states.shape[1]):
        out[:, dim] = np.interp(grid, ts, states[:, dim])
    return out


def load_recorded_10hz(episodes: list[Path], sample_hz: float) -> tuple[np.ndarray, list[str]]:
    sampled: list[np.ndarray] = []
    used: list[str] = []
    for episode in episodes:
        left_path = episode / "yam_left.mcap"
        right_path = episode / "yam_right.mcap"
        if not left_path.exists() or not right_path.exists():
            continue
        left_ts, left = read_arm_mcap(left_path)
        right_ts, right = read_arm_mcap(right_path)
        start = max(left_ts[0], right_ts[0])
        end = min(left_ts[-1], right_ts[-1])
        if end <= start:
            continue
        grid = np.arange(start, end, 1.0 / sample_hz)
        if len(grid) < 2:
            continue
        left_sampled = interp_states(left_ts, left, grid)
        right_sampled = interp_states(right_ts, right, grid)
        sampled.append(np.concatenate([left_sampled, right_sampled], axis=1))
        used.append(str(episode))
    if not sampled:
        raise RuntimeError("no usable yam_left/yam_right recording pairs found")
    return np.concatenate(sampled, axis=0), used


def load_pixel_actions(path: Path, array_key: str, window_idx: int | None) -> np.ndarray:
    with np.load(path) as data:
        if array_key not in data:
            raise KeyError(f"{path} does not contain {array_key!r}; keys={list(data.keys())}")
        actions = np.asarray(data[array_key], dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] != 14:
        raise ValueError(f"{array_key!r} must have shape (N, T, 14), got {actions.shape}")
    if window_idx is None:
        return actions
    if not 0 <= window_idx < actions.shape[0]:
        raise IndexError(f"window_idx={window_idx} out of range for N={actions.shape[0]}")
    return actions[window_idx]


def flatten_pixel_actions(pixel: np.ndarray) -> np.ndarray:
    if pixel.ndim == 2:
        return pixel
    if pixel.ndim == 3:
        return pixel.reshape(-1, pixel.shape[-1])
    raise ValueError(f"unexpected pixel action shape: {pixel.shape}")


def pixel_frame_deltas(pixel: np.ndarray) -> np.ndarray:
    if pixel.ndim == 2:
        if pixel.shape[0] < 2:
            return np.zeros((0, pixel.shape[-1]), dtype=np.float64)
        return np.abs(np.diff(pixel, axis=0))
    if pixel.ndim == 3:
        if pixel.shape[1] < 2:
            return np.zeros((0, pixel.shape[-1]), dtype=np.float64)
        return np.abs(np.diff(pixel, axis=1)).reshape(-1, pixel.shape[-1])
    raise ValueError(f"unexpected pixel action shape: {pixel.shape}")


def load_joint_limits(left_path: Path, right_path: Path) -> tuple[np.ndarray, np.ndarray]:
    def one(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        limits = np.asarray(cfg["joint_limits"], dtype=np.float64)
        return limits[:, 0], limits[:, 1]

    left_lo, left_hi = one(left_path)
    right_lo, right_hi = one(right_path)
    lower = np.concatenate([left_lo, [0.0], right_lo, [0.0]])
    upper = np.concatenate([left_hi, [1.0], right_hi, [1.0]])
    return lower, upper


def nearest_l2(pixel: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    # Small pixel arrays, large recorded arrays: loop over pixel frames to avoid
    # materializing an unnecessary (P, R, 14) tensor.
    dists = []
    for row in pixel:
        diff = recorded - row[None, :]
        dists.append(float(np.sqrt(np.min(np.sum(diff * diff, axis=1)))))
    return np.asarray(dists, dtype=np.float64)


def compare(args: argparse.Namespace) -> ComparisonResult:
    episodes = sorted(Path(args.recordings_root).glob(args.episode_glob))
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    recorded, used = load_recorded_10hz(episodes, sample_hz=args.sample_hz)
    pixel = load_pixel_actions(Path(args.npz_path), args.array_key, args.window_idx)
    pixel_flat = flatten_pixel_actions(pixel)

    rec_min = recorded.min(axis=0)
    rec_max = recorded.max(axis=0)
    rec_p001 = np.percentile(recorded, 0.1, axis=0)
    rec_p999 = np.percentile(recorded, 99.9, axis=0)
    pix_min = pixel_flat.min(axis=0)
    pix_max = pixel_flat.max(axis=0)
    pixel_deltas = pixel_frame_deltas(pixel)
    pix_delta = pixel_deltas.max(axis=0) if len(pixel_deltas) else np.zeros(14)
    rec_delta = np.abs(np.diff(recorded, axis=0))
    rec_p999_delta = np.percentile(rec_delta, 99.9, axis=0)
    rec_max_delta = rec_delta.max(axis=0)

    outside_hard = [
        DIM_NAMES[i]
        for i in range(14)
        if pix_min[i] < rec_min[i] - args.recorded_margin or pix_max[i] > rec_max[i] + args.recorded_margin
    ]
    outside_p = [
        DIM_NAMES[i]
        for i in range(14)
        if pix_min[i] < rec_p001[i] - args.recorded_margin or pix_max[i] > rec_p999[i] + args.recorded_margin
    ]

    ratio = pix_delta / np.maximum(rec_p999_delta, 1e-9)
    nearest = nearest_l2(pixel_flat, recorded)

    lower, upper = load_joint_limits(Path(args.left_limits_path), Path(args.right_limits_path))
    limit_violations = [
        DIM_NAMES[i]
        for i in range(14)
        if pix_min[i] < lower[i] - args.limit_tolerance or pix_max[i] > upper[i] + args.limit_tolerance
    ]

    return ComparisonResult(
        npz_path=args.npz_path,
        array_key=args.array_key,
        window_idx=args.window_idx,
        recordings_root=args.recordings_root,
        episodes_used=used,
        recorded_10hz_shape=tuple(recorded.shape),
        pixel_shape=tuple(pixel.shape),
        recorded_min=rec_min.tolist(),
        recorded_max=rec_max.tolist(),
        recorded_p001=rec_p001.tolist(),
        recorded_p999=rec_p999.tolist(),
        pixel_min=pix_min.tolist(),
        pixel_max=pix_max.tolist(),
        pixel_first=pixel_flat[0].tolist(),
        pixel_last=pixel_flat[-1].tolist(),
        pixel_max_frame_delta=pix_delta.tolist(),
        recorded_p999_frame_delta_10hz=rec_p999_delta.tolist(),
        recorded_max_frame_delta_10hz=rec_max_delta.tolist(),
        outside_recorded_hard_dims=outside_hard,
        outside_recorded_p001_p999_dims=outside_p,
        pixel_over_recorded_p999_delta_ratio=ratio.tolist(),
        nearest_recorded_l2_per_pixel_frame=nearest.tolist(),
        nearest_recorded_l2_min=float(nearest.min()),
        nearest_recorded_l2_max=float(nearest.max()),
        nearest_recorded_l2_mean=float(nearest.mean()),
        joint_limit_violations=limit_violations,
        gripper_range_ok=bool(
            np.all(pixel_flat[:, [6, 13]] >= -args.limit_tolerance)
            and np.all(pixel_flat[:, [6, 13]] <= 1.0 + args.limit_tolerance)
        ),
    )


def print_summary(result: ComparisonResult) -> None:
    print("PIXEL IDM vs recorded YAM trajectory comparison")
    print(f"  npz: {result.npz_path} [{result.array_key}] window={result.window_idx}")
    print(f"  episodes used: {len(result.episodes_used)}")
    print(f"  recorded 10 Hz states: {result.recorded_10hz_shape}")
    print(f"  pixel states: {result.pixel_shape}")
    print(f"  gripper range ok: {result.gripper_range_ok}")
    print(f"  outside recorded hard range dims: {result.outside_recorded_hard_dims}")
    print(f"  outside recorded 0.1%-99.9% dims: {result.outside_recorded_p001_p999_dims}")
    print(f"  joint-limit violations: {result.joint_limit_violations}")
    print(
        "  nearest recorded 14D L2: "
        f"min={result.nearest_recorded_l2_min:.4f} "
        f"mean={result.nearest_recorded_l2_mean:.4f} "
        f"max={result.nearest_recorded_l2_max:.4f}"
    )
    print()
    print("  dim                         pixel_min   pixel_max   rec_min     rec_max     rec_p0.1   rec_p99.9  pix_delta  rec_p99.9_delta")
    for i, name in enumerate(DIM_NAMES):
        print(
            f"  {name:25s}"
            f"{result.pixel_min[i]:10.5f}"
            f"{result.pixel_max[i]:10.5f}"
            f"{result.recorded_min[i]:10.5f}"
            f"{result.recorded_max[i]:10.5f}"
            f"{result.recorded_p001[i]:10.5f}"
            f"{result.recorded_p999[i]:11.5f}"
            f"{result.pixel_max_frame_delta[i]:10.5f}"
            f"{result.recorded_p999_frame_delta_10hz[i]:17.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PIXEL IDM actions against recorded YAM trajectories.")
    parser.add_argument("--npz-path", default="PIXEL_IDM_INFERENCE/full_inference_actions.npz")
    parser.add_argument("--array-key", default="predicted_actions")
    parser.add_argument("--window-idx", type=int, default=0, help="Selected window; use --all-windows to compare every frame.")
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--recordings-root", default="recordings/20260604")
    parser.add_argument("--episode-glob", default="episode_*")
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--recorded-margin", type=float, default=1e-3)
    parser.add_argument("--limit-tolerance", type=float, default=1e-6)
    parser.add_argument("--left-limits-path", default="robot_configs/yam/xdof_hq/left.yaml")
    parser.add_argument("--right-limits-path", default="robot_configs/yam/xdof_hq/right.yaml")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if args.all_windows:
        args.window_idx = None
    return args


def main() -> int:
    args = parse_args()
    result = compare(args)
    print_summary(result)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(result), indent=2))
        print(f"\nWrote JSON report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render a grid video showing all episodes in a directory playing simultaneously.

For each episode, composites top camera with wrist camera PiP overlays (reusing
the logic from composite_episode_view_renderer.py), then tiles all episodes
into a grid layout with episode labels.

Usage:
    uv run python scripts/grid_episode_viewer.py /path/to/day_dir
    uv run python scripts/grid_episode_viewer.py /path/to/day_dir --cols 4 --cell-width 480
    uv run python scripts/grid_episode_viewer.py /path/to/day_dir --output grid.mp4 --no-play
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np


def _load_video(path: Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    """Load an MP4 as (T, H, W, 3) uint8. If resize=(w, h), downscale each frame on load."""
    import imageio.v3 as iio

    frames = iio.imread(str(path), plugin="pyav")
    if resize is not None:
        from PIL import Image

        w, h = resize
        out = np.empty((len(frames), h, w, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            out[i] = np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR))
        return out
    return frames


def _load_timestamps(episode_dir: Path, camera: str) -> np.ndarray:
    ts_path = episode_dir / f"{camera}-rgb-timestamp.npy"
    if not ts_path.exists():
        raise FileNotFoundError(f"Timestamp sidecar not found: {ts_path}")
    return np.load(str(ts_path))


def _sample_frame(frames: np.ndarray, ts: np.ndarray, query_t: float) -> np.ndarray:
    idx = int(np.searchsorted(ts, query_t, side="right")) - 1
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img)


def _composite(
    main: np.ndarray,
    left_pip: np.ndarray | None,
    right_pip: np.ndarray | None,
    pip_w: int,
    pip_h: int,
    margin: int = 8,
) -> np.ndarray:
    out = main.copy()
    fh, fw = main.shape[:2]
    border = 2

    def _paste(inset: np.ndarray, x: int, y: int) -> None:
        bx0 = max(0, x - border)
        by0 = max(0, y - border)
        bx1 = min(fw, x + pip_w + border)
        by1 = min(fh, y + pip_h + border)
        out[by0:by1, bx0:bx1] = 0
        out[y : y + pip_h, x : x + pip_w] = inset

    if left_pip is not None:
        resized = _resize_frame(left_pip, pip_w, pip_h)
        _paste(resized, margin, margin)

    if right_pip is not None:
        resized = _resize_frame(right_pip, pip_w, pip_h)
        _paste(resized, fw - margin - pip_w, margin)

    return out


class EpisodeData:
    """Holds loaded camera data for one episode, downscaled to cell size."""

    def __init__(self, episode_dir: Path, cell_size: tuple[int, int]):
        """cell_size is (width, height) of the target grid cell."""
        self.name = episode_dir.name
        cell_w, cell_h = cell_size
        top_mp4 = episode_dir / "camera_top-images-rgb.mp4"
        if not top_mp4.exists():
            raise FileNotFoundError(f"No top camera: {top_mp4}")

        self.top_frames = _load_video(top_mp4, resize=(cell_w, cell_h))
        self.top_ts = _load_timestamps(episode_dir, "camera_top")
        self.h, self.w = cell_h, cell_w

        left_mp4 = episode_dir / "camera_left-images-rgb.mp4"
        right_mp4 = episode_dir / "camera_right-images-rgb.mp4"

        pip_scale = 0.25
        self.pip_w = int(cell_w * pip_scale)
        self.pip_h = self.pip_w
        pip_resize: tuple[int, int] | None = None

        if left_mp4.exists():
            # peek at original aspect ratio before loading downscaled
            import av as _av

            with _av.open(str(left_mp4)) as _c:
                _s = _c.streams.video[0]
                lw, lh = _s.width, _s.height
            self.pip_h = int(self.pip_w * lh / lw)
            pip_resize = (self.pip_w, self.pip_h)
        elif right_mp4.exists():
            import av as _av

            with _av.open(str(right_mp4)) as _c:
                _s = _c.streams.video[0]
                rw, rh = _s.width, _s.height
            self.pip_h = int(self.pip_w * rh / rw)
            pip_resize = (self.pip_w, self.pip_h)

        self.left_frames = _load_video(left_mp4, resize=pip_resize) if left_mp4.exists() else None
        self.left_ts = _load_timestamps(episode_dir, "camera_left") if left_mp4.exists() else None
        self.right_frames = _load_video(right_mp4, resize=pip_resize) if right_mp4.exists() else None
        self.right_ts = _load_timestamps(episode_dir, "camera_right") if right_mp4.exists() else None

    @property
    def duration(self) -> float:
        return float(self.top_ts[-1] - self.top_ts[0])

    @property
    def num_frames(self) -> int:
        return len(self.top_frames)

    def get_composite_frame(self, t: float) -> np.ndarray:
        top_f = _sample_frame(self.top_frames, self.top_ts, self.top_ts[0] + t)
        left_f = (
            _sample_frame(self.left_frames, self.left_ts, self.left_ts[0] + t)
            if self.left_frames is not None
            else None
        )
        right_f = (
            _sample_frame(self.right_frames, self.right_ts, self.right_ts[0] + t)
            if self.right_frames is not None
            else None
        )
        # frames are already at target size, skip resize in _composite
        out = top_f.copy()
        fh, fw = out.shape[:2]
        margin = 8
        border = 2

        def _paste(inset: np.ndarray, x: int, y: int) -> None:
            ih, iw = inset.shape[:2]
            bx0 = max(0, x - border)
            by0 = max(0, y - border)
            bx1 = min(fw, x + iw + border)
            by1 = min(fh, y + ih + border)
            out[by0:by1, bx0:bx1] = 0
            out[y : y + ih, x : x + iw] = inset

        if left_f is not None:
            _paste(left_f, margin, margin)
        if right_f is not None:
            _paste(right_f, fw - margin - right_f.shape[1], margin)

        return out


def _draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 4, frame.shape[0] - th - 8
    draw.rectangle([x - 2, y - 2, x + tw + 2, y + th + 2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def build_grid(
    day_dir: Path,
    output: Path,
    cols: int | None = None,
    cell_width: int = 240,
    fps: int = 30,
    crf: int = 28,
    preset: str = "fast",
    label: bool = False,
) -> None:
    episode_dirs = sorted(
        [d for d in day_dir.iterdir() if d.is_dir() and (d / "camera_top-images-rgb.mp4").exists()]
    )
    if not episode_dirs:
        raise RuntimeError(f"No episodes with camera_top found in {day_dir}")

    n = len(episode_dirs)
    if cols is None:
        cols = min(4, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    print(f"Found {n} episodes, grid: {cols}x{rows}")

    # Peek at first episode to get aspect ratio for cell size
    import av as _av

    with _av.open(str(episode_dirs[0] / "camera_top-images-rgb.mp4")) as _c:
        _s = _c.streams.video[0]
        ref_w, ref_h = _s.width, _s.height
    cell_height = int(cell_width * ref_h / ref_w)
    cell_width = (cell_width // 2) * 2
    cell_height = (cell_height // 2) * 2
    cell_size = (cell_width, cell_height)

    print(f"Cell: {cell_width}x{cell_height} — loading all episodes downscaled ...")

    episodes: list[EpisodeData] = []
    for i, ep_dir in enumerate(episode_dirs):
        print(f"  [{i + 1}/{n}] {ep_dir.name}")
        episodes.append(EpisodeData(ep_dir, cell_size=cell_size))
    grid_w = cols * cell_width
    grid_h = rows * cell_height

    max_duration = max(ep.duration for ep in episodes)
    total_frames = int(max_duration * fps)

    print(f"Cell: {cell_width}x{cell_height}, Grid: {grid_w}x{grid_h}")
    print(f"Max duration: {max_duration:.1f}s, {total_frames} frames @ {fps}fps")
    print(f"Writing to: {output}")

    import av

    with av.open(str(output), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = grid_w
        stream.height = grid_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}

        for fi in range(total_frames):
            t = fi / fps
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for idx, ep in enumerate(episodes):
                ep_t = min(t, ep.duration)
                cell = ep.get_composite_frame(ep_t)
                if label:
                    short_name = ep.name.replace("episode_", "ep_")
                    cell = _draw_label(cell, short_name)

                row, col = divmod(idx, cols)
                y0 = row * cell_height
                x0 = col * cell_width
                grid[y0 : y0 + cell_height, x0 : x0 + cell_width] = cell

            av_frame = av.VideoFrame.from_ndarray(grid, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

            if fi % 30 == 0:
                print(f"  frame {fi}/{total_frames} ({t:.1f}s) ...", end="\r")

        for packet in stream.encode():
            container.mux(packet)

    print(f"\nDone. {total_frames} frames, {n} episodes.")


def _play(path: Path) -> None:
    for player in ("ffplay", "mpv", "vlc"):
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            subprocess.run([player, str(path)])
            return
    print(f"No video player found. Open manually: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a grid video of all episodes in a directory"
    )
    parser.add_argument(
        "day_dir",
        help="Directory containing episode_* subdirectories",
    )
    parser.add_argument("--output", "-o", help="Output MP4 path (default: <day_dir>/grid.mp4)")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns (default: auto)")
    parser.add_argument(
        "--cell-width", type=int, default=240, help="Width of each cell in pixels (default: 240)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--crf", type=int, default=28, help="x264 CRF (default: 28)")
    parser.add_argument("--preset", default="fast", help="x264 preset (default: fast)")
    parser.add_argument("--label", action="store_true", help="Show episode name labels")
    parser.add_argument("--no-play", action="store_true", help="Don't open video player after")
    args = parser.parse_args()

    day_dir = Path(args.day_dir)
    if not day_dir.is_dir():
        parser.error(f"Not a directory: {day_dir}")

    output = Path(args.output) if args.output else day_dir / "grid.mp4"

    build_grid(
        day_dir,
        output,
        cols=args.cols,
        cell_width=args.cell_width,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        label=args.label,
    )

    if not args.no_play:
        _play(output)


if __name__ == "__main__":
    main()

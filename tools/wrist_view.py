#!/usr/bin/env python3
"""Read, inspect and eyeball the wrist-camera stream.

The wrist camera (`camera_left`, a 640x480 USB webcam on the left gripper) is
the only view the deployed policy actually gets — `yam_grasp_v2_wrist` is
wrist-only — and until now there was no way to look at it. Every existing tool
consumes it; none of them shows it to a human.

    recording episode                 LeRobot episode
    camera_left-images-rgb.mp4        videos/observation.images.wrist/*.mp4
    camera_left-rgb-timestamp.npy       + meta/episodes/*.parquet (from/to ts)
              \\                        /
               `---- WristStream ----'
                        |
        summary (health)  sheet (contact sheet)  dump (frames)  grasps

Subcommands
    summary   one row per episode: frames, duration, real fps, timestamp health,
              resolution, exposure, sharpness, frozen-frame fraction
    sheet     contact sheet PNG — the whole episode as a grid, at a glance
    dump      individual frames as PNGs
    grasps    the wrist frame at each LABELLED grasp instant, side by side.
              This is the one that answers "what did the gripper see when it
              closed", and it is how you tell a real pick from a witness that
              scored a grasp on an empty gripper.

WHY IT DECODES WITH PyAV AND NOT cv2: the recordings are h264 but LeRobot
re-encodes to AV1, and this box's cv2 (4.12.0) tries a hardware AV1 decoder that
does not exist here — `VideoCapture.read()` returns False on every frame with no
exception. PyAV's libdav1d decodes both. Reading a LeRobot dataset with cv2 does
not fail loudly; it silently yields nothing.

READ-ONLY. recordings/ and outputs/ are symlinks into the live production tree.
Nothing here writes anywhere except --out.

Usage
    python tools/wrist_view.py summary --root recordings
    python tools/wrist_view.py summary --dataset ETHRC/yam_grasp_v2_wrist
    python tools/wrist_view.py sheet --episode recordings/20260727/episode_140454_648f75b2
    python tools/wrist_view.py grasps --episode recordings/20260726/episode_222925_ce5ca281
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# The left wrist camera. camera_top is the fixed overhead view and camera_scan
# looks at the packet mat; neither is on the gripper.
DEFAULT_CAMERA = "camera_left"
# LeRobot feature keys that hold a wrist view, newest naming first. The bimanual
# export writes wrist_left/wrist_right; every dataset before it wrote "wrist".
WRIST_KEYS = ("observation.images.wrist", "observation.images.wrist_left",
              "observation.images.wrist_right")

# A frame whose mean-absolute difference from its predecessor is below this (on
# a 0-255 scale) is not a new picture. A live 640x480 webcam pointed at a static
# scene still moves ~0.3-1.0 from sensor noise alone; a driver that re-delivers
# its last buffer scores exactly 0.0.
FROZEN_FRAME_MAD = 0.15
# Variance of the Laplacian, the standard cheap focus measure. Below this a
# 640x480 frame has no edges worth calling a picture.
BLUR_VAR_FLOOR = 30.0
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


# ── decoding ────────────────────────────────────────────────────────────────
def _open(path: Path):
    import av
    return av.open(str(path))


def decode_frames(mp4: Path, stride: int = 1, limit: int | None = None,
                  start_s: float | None = None, end_s: float | None = None):
    """Yield (index, seconds, RGB uint8 frame) in decode order.

    Sequential decode, never seek — the same rule export_lerobot follows, for
    the same reason: seeking h264/AV1 lands on the nearest keyframe and silently
    returns a different frame than the one asked for.

    ``start_s``/``end_s`` select a slice of a CONCATENATED LeRobot video, where
    many episodes share one file and each episode is a timestamp range.
    """
    container = _open(mp4)
    try:
        kept = 0
        for i, frame in enumerate(container.decode(video=0)):
            t = float(frame.pts * frame.time_base) if frame.pts is not None else float(i)
            if end_s is not None and t >= end_s:
                break
            if start_s is not None and t < start_s:
                continue
            if i % stride:
                continue
            yield i, t, frame.to_ndarray(format="rgb24")
            kept += 1
            if limit is not None and kept >= limit:
                break
    finally:
        container.close()


# ── stream sources ──────────────────────────────────────────────────────────
@dataclass
class WristStream:
    """One wrist video plus whatever timeline evidence exists for it."""
    name: str
    mp4: Path
    # Recording sidecar timestamps (seconds, camera DEVICE clock). None for a
    # LeRobot dataset, which has thrown the real timeline away and re-stamps
    # every frame at frame_index/fps.
    stamps: np.ndarray | None = None
    start_s: float | None = None
    end_s: float | None = None
    source: str = "recording"
    # Authoritative frame count when the container's own count would be wrong.
    # A LeRobot v3 episode is a timestamp RANGE inside a video shared by many
    # episodes, so the container reports the whole file (8962 frames for a
    # 4.8 s episode); meta/episodes carries the per-episode `length`.
    n_frames: int | None = None

    def frames(self, stride: int = 1, limit: int | None = None):
        return decode_frames(self.mp4, stride, limit, self.start_s, self.end_s)


def recording_stream(ep: Path, camera: str = DEFAULT_CAMERA) -> WristStream | None:
    mp4 = ep / f"{camera}-images-rgb.mp4"
    if not mp4.exists():
        return None
    npy = ep / f"{camera}-rgb-timestamp.npy"
    stamps = np.load(npy).astype(float) if npy.exists() else None
    return WristStream(ep.name, mp4, stamps, source="recording")


def recording_streams(root: Path, camera: str = DEFAULT_CAMERA) -> list[WristStream]:
    if (root / f"{camera}-images-rgb.mp4").exists():
        s = recording_stream(root, camera)
        return [s] if s else []
    out = []
    for ep in sorted(p for p in root.rglob("episode_*")
                     if p.is_dir() and ".trash" not in p.parts):
        s = recording_stream(ep, camera)
        if s:
            out.append(s)
    return out


def dataset_root(spec: str) -> Path:
    """A path, or a `org/name` repo-id resolved in the local LeRobot cache."""
    p = Path(spec)
    if (p / "meta" / "info.json").exists():
        return p
    cached = DEFAULT_HF_CACHE / spec
    if (cached / "meta" / "info.json").exists():
        return cached
    raise SystemExit(f"no LeRobot dataset at {spec} or {cached}")


def dataset_streams(root: Path, key: str | None = None,
                    episodes: list[int] | None = None) -> list[WristStream]:
    """One WristStream per LeRobot episode.

    LeRobot v3 packs many episodes into one mp4 and records each episode's slice
    as `videos/<key>/from_timestamp`..`to_timestamp` in meta/episodes. So an
    "episode" here is a timestamp range inside a shared file, not a file.
    """
    import pandas as pd
    info = json.loads((root / "meta" / "info.json").read_text())
    keys = [k for k in info.get("features", {}) if k in WRIST_KEYS]
    if key is None:
        if not keys:
            raise SystemExit(f"{root} has no wrist feature (has {sorted(info['features'])})")
        key = keys[0]

    meta = pd.concat([pd.read_parquet(p)
                      for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    vpath = info["video_path"]
    out = []
    for _, row in meta.iterrows():
        idx = int(row["episode_index"])
        if episodes is not None and idx not in episodes:
            continue
        mp4 = root / vpath.format(video_key=key,
                                  chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                  file_index=int(row[f"videos/{key}/file_index"]))
        out.append(WristStream(
            f"ep{idx:04d}", mp4, None,
            start_s=float(row[f"videos/{key}/from_timestamp"]),
            end_s=float(row[f"videos/{key}/to_timestamp"]),
            source="lerobot", n_frames=int(row["length"])))
    return out


# ── measurement ─────────────────────────────────────────────────────────────
def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian on the green channel (no cv2 dependency)."""
    g = rgb[..., 1].astype(np.float32)
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


@dataclass
class StreamSummary:
    episode: str
    source: str
    frames_decoded: int = 0
    stamps: int | None = None
    frame_stamp_mismatch: int | None = None
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    fps_measured: float = 0.0
    stamp_non_monotonic: int | None = None
    stamp_max_gap_s: float | None = None
    brightness_mean: float = 0.0
    brightness_min: float = 0.0
    brightness_max: float = 0.0
    sharpness_median: float = 0.0
    frozen_frac: float = 0.0
    problems: list[str] = field(default_factory=list)


def summarise(stream: WristStream, stride: int = 10) -> StreamSummary:
    """Decode every ``stride``-th frame and report what the stream actually is.

    Every field here exists because something in this stack can produce it
    silently: a truncated mp4 (disk full mid-session, DATA-PIPELINE 2.6) leaves
    a video shorter than its sidecar with both files internally consistent; a
    RealSense pipeline restart can rewind the device clock with no marker
    (2.5); a webcam that stops delivering re-serves its last buffer forever and
    every downstream consumer reads it as a valid picture.
    """
    s = StreamSummary(stream.name, stream.source)
    brights, sharps, mads = [], [], []
    prev = None
    times = []
    sampled = 0
    for _, t, img in stream.frames(stride=stride):
        sampled += 1
        s.height, s.width = img.shape[:2]
        times.append(t)
        brights.append(float(img.mean()))
        sharps.append(sharpness(img))
        if prev is not None:
            mads.append(float(np.abs(img.astype(np.int16) - prev).mean()))
        prev = img.astype(np.int16)

    if not sampled:
        s.problems.append("no frames decoded")
        return s

    # NOT the sampled count -- a strided decode cannot see how many frames
    # there are, and reporting the sample count as "frames" is exactly the kind
    # of plausible wrong number this file exists to stop producing. And NOT the
    # container count for a LeRobot episode, which shares its video file with
    # every other episode in the chunk.
    if stream.n_frames is not None:
        s.frames_decoded = stream.n_frames
    else:
        with _open(stream.mp4) as c:
            s.frames_decoded = int(c.streams.video[0].frames or 0) or sampled

    s.duration_s = float(times[-1] - times[0])
    s.fps_measured = ((sampled - 1) * stride / s.duration_s
                      if s.duration_s > 0 else 0.0)
    s.brightness_mean = float(np.mean(brights))
    s.brightness_min = float(np.min(brights))
    s.brightness_max = float(np.max(brights))
    s.sharpness_median = float(np.median(sharps))
    s.frozen_frac = float(np.mean(np.array(mads) < FROZEN_FRAME_MAD)) if mads else 0.0

    if stream.stamps is None:
        if stream.source == "recording":
            # No sidecar means no timeline: every consumer that resamples this
            # camera by timestamp (export_lerobot, the grasp aligner below) is
            # blind on this episode.
            s.problems.append("no timestamp sidecar")
    else:
        st = stream.stamps
        s.stamps = int(st.size)
        s.frame_stamp_mismatch = s.frames_decoded - int(st.size)
        d = np.diff(st)
        # Not sorted, and nothing downstream checks: export_lerobot's
        # nearest_index calls np.searchsorted, which does not validate its input
        # and silently returns a wrong index instead of raising.
        s.stamp_non_monotonic = int((d <= 0).sum())
        s.stamp_max_gap_s = float(d.max()) if d.size else 0.0
        if s.stamp_max_gap_s and s.stamp_max_gap_s > 0.5:
            s.problems.append(f"{s.stamp_max_gap_s:.2f}s camera gap")

    if s.frozen_frac > 0.5:
        s.problems.append(f"{s.frozen_frac:.0%} of sampled frames identical to the previous one")
    if s.sharpness_median < BLUR_VAR_FLOOR:
        s.problems.append(f"median sharpness {s.sharpness_median:.1f} < {BLUR_VAR_FLOOR}")
    if s.brightness_max < 16:
        s.problems.append("frame is black throughout")
    if s.frame_stamp_mismatch:
        s.problems.append(f"mp4 has {s.frame_stamp_mismatch:+d} frames vs the timestamp sidecar")
    if s.stamp_non_monotonic:
        s.problems.append(f"{s.stamp_non_monotonic} non-positive timestamp steps")
    return s


# ── rendering ───────────────────────────────────────────────────────────────
def contact_sheet(images: list[np.ndarray], cols: int = 6,
                  cell_w: int = 320, labels: list[str] | None = None) -> np.ndarray:
    """Grid montage of RGB frames, letterboxed, with an optional caption strip."""
    if not images:
        raise ValueError("no images")
    h, w = images[0].shape[:2]
    cell_h = max(1, int(round(cell_w * h / w)))
    bar = 18 if labels else 0
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * (cell_h + bar), cols * cell_w, 3), np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y, x = r * (cell_h + bar), c * cell_w
        # Nearest-neighbour by index arithmetic: this is a review artefact, and
        # a resampling filter would smooth away the very blur it is here to show.
        yi = (np.arange(cell_h) * img.shape[0] // cell_h).clip(0, img.shape[0] - 1)
        xi = (np.arange(cell_w) * img.shape[1] // cell_w).clip(0, img.shape[1] - 1)
        sheet[y:y + cell_h, x:x + cell_w] = img[np.ix_(yi, xi)]
        if labels:
            _draw_text(sheet, labels[i], x + 3, y + cell_h + 3)
    return sheet


_GLYPH_W, _GLYPH_H = 5, 7
# A 5x7 bitmap font, only the characters a caption needs. Drawing text is worth
# ~40 lines to avoid making PIL/cv2-with-freetype a hard dependency of a review
# tool — an unlabelled contact sheet is a wall of pictures nobody can act on.
_FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    " ": ("00000",) * 7,
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "d": ("00001", "00001", "01111", "10001", "10001", "10001", "01111"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "b": ("10000", "10000", "11110", "10001", "10001", "10001", "11110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
}


def _draw_text(canvas: np.ndarray, text: str, x0: int, y0: int,
               colour=(255, 255, 0)) -> None:
    for ch in text:
        glyph = _FONT.get(ch, _FONT.get(ch.lower()))
        if glyph is not None:
            for r, rowbits in enumerate(glyph):
                for c, bit in enumerate(rowbits):
                    if bit == "1":
                        y, x = y0 + r * 2, x0 + c * 2
                        if 0 <= y < canvas.shape[0] - 1 and 0 <= x < canvas.shape[1] - 1:
                            canvas[y:y + 2, x:x + 2] = colour
        x0 += (_GLYPH_W + 1) * 2


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import av
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    with av.open(str(path), "w", format="image2") as out:
        st = out.add_stream("png", rate=1)
        st.width, st.height, st.pix_fmt = rgb.shape[1], rgb.shape[0], "rgb24"
        for packet in st.encode(frame):
            out.mux(packet)
        for packet in st.encode(None):
            out.mux(packet)


# ── grasp-aligned frames ────────────────────────────────────────────────────
def grasp_times(ep: Path, arm: str = "left") -> list[tuple[float, str]]:
    """(t, outcome) for every labelled grasp attempt in a recording episode."""
    from robots_realtime.labeling.label_episode import annotations_path
    p = annotations_path(ep, arm)
    if not p.exists():
        return []
    ann = json.loads(p.read_text())
    return [(float(a["t"]), str(a.get("outcome", "?")))
            for a in (ann.get("grasp_attempts") or []) if a.get("t") is not None]


def frames_at(stream: WristStream, targets: list[float]) -> list[tuple[float, np.ndarray]]:
    """The decoded frame nearest each target time, one forward pass.

    The targets are on the ROBOT wall clock (annotations) and the sidecar is on
    the camera DEVICE clock. Nothing in the recording path reconciles them
    (AUDIT.md S5), so these frames are "the frame whose sidecar stamp is nearest
    that number" — good enough to eyeball a grasp, NOT evidence about latency.
    """
    if stream.stamps is None:
        raise ValueError("grasp alignment needs the recording timestamp sidecar")
    want = sorted(set(int(np.argmin(np.abs(stream.stamps - t))) for t in targets))
    out, wanted = [], set(want)
    for i, _, img in stream.frames():
        if i in wanted:
            out.append((float(stream.stamps[i]), img))
            wanted.discard(i)
            if not wanted:
                break
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────
_HDR = (f"{'episode':<34}{'frames':>7}{'stamps':>7}{'sec':>7}{'fps':>6}"
        f"{'WxH':>10}{'gap_s':>7}{'bright':>8}{'sharp':>8}{'frozen':>8}  problems")


def _row(s: StreamSummary) -> str:
    gap = "-" if s.stamp_max_gap_s is None else f"{s.stamp_max_gap_s:.3f}"
    return (f"{s.episode:<34}{s.frames_decoded:>7}"
            f"{'-' if s.stamps is None else s.stamps:>7}{s.duration_s:>7.1f}"
            f"{s.fps_measured:>6.1f}{f'{s.width}x{s.height}':>10}{gap:>7}"
            f"{s.brightness_mean:>8.1f}{s.sharpness_median:>8.1f}"
            f"{s.frozen_frac:>7.0%}  {'; '.join(s.problems)}")


def _streams(a) -> list[WristStream]:
    if a.dataset:
        eps = [int(x) for x in a.episode_index.split(",")] if a.episode_index else None
        return dataset_streams(dataset_root(a.dataset), a.feature, eps)
    root = Path(a.episode) if a.episode else Path(a.root)
    return recording_streams(root, a.camera)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("summary", "sheet", "dump", "grasps"))
    src = ap.add_argument_group("source (recordings by default)")
    src.add_argument("--root", default="recordings", help="recordings tree to walk")
    src.add_argument("--episode", default=None, help="one recording episode dir")
    src.add_argument("--camera", default=DEFAULT_CAMERA)
    src.add_argument("--dataset", default=None,
                     help="LeRobot dataset path or repo-id (e.g. ETHRC/yam_grasp_v2_wrist)")
    src.add_argument("--episode-index", default=None,
                     help="comma-separated LeRobot episode indices")
    src.add_argument("--feature", default=None, help="LeRobot wrist feature key")
    ap.add_argument("--out", default="analysis/wrist",
                    help="where artefacts are written (never the recordings tree)")
    ap.add_argument("--stride", type=int, default=10,
                    help="decode every Nth frame (summary/sheet)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="max frames per episode")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--arm", default="left", help="which arm's annotations (grasps)")
    ap.add_argument("--json", dest="as_json", action="store_true")
    a = ap.parse_args(argv)

    streams = _streams(a)
    if a.max_episodes:
        streams = streams[:a.max_episodes]
    if not streams:
        print("no wrist streams found", file=sys.stderr)
        return 1
    out = Path(a.out)

    if a.command == "summary":
        rows = [summarise(s, a.stride) for s in streams]
        if a.as_json:
            print(json.dumps([asdict(r) for r in rows], indent=2))
        else:
            print(_HDR)
            print("-" * len(_HDR))
            for r in rows:
                print(_row(r))
            bad = [r for r in rows if r.problems]
            print(f"\n{len(rows)} wrist streams, {len(bad)} with problems")
        return 0

    for s in streams:
        if a.command == "sheet":
            imgs, labels = [], []
            for _, t, img in s.frames(stride=a.stride, limit=a.limit):
                imgs.append(img)
                labels.append(f"{t:.1f}s")
            if not imgs:
                print(f"{s.name}: no frames", file=sys.stderr)
                continue
            p = out / f"{s.name}_{a.camera if s.source == 'recording' else 'wrist'}_sheet.png"
            save_png(p, contact_sheet(imgs, a.cols, labels=labels))
            print(f"{p}  ({len(imgs)} frames)")

        elif a.command == "dump":
            n = 0
            for i, t, img in s.frames(stride=a.stride, limit=a.limit):
                save_png(out / s.name / f"{i:06d}_{t:.3f}.png", img)
                n += 1
            print(f"{out / s.name}  ({n} frames)")

        elif a.command == "grasps":
            ep = Path(s.mp4).parent
            gt = grasp_times(ep, a.arm)
            if not gt:
                print(f"{s.name}: no labelled grasps", file=sys.stderr)
                continue
            got = frames_at(s, [t for t, _ in gt])
            if not got:
                print(f"{s.name}: no frames at grasp times", file=sys.stderr)
                continue
            labels = [f"{t:.1f} {o}" for (t, _), (_, o) in zip(got, gt)]
            p = out / f"{s.name}_grasps.png"
            save_png(p, contact_sheet([g[1] for g in got], a.cols, labels=labels))
            print(f"{p}  ({len(got)} grasps: "
                  + ", ".join(sorted({o for _, o in gt})) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

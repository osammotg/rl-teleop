#!/usr/bin/env python3
"""Decode a LeRobot dataset's mp4s to JPGs, so training never touches a video decoder.

WHY THIS EXISTS RATHER THAN ~/Desktop/lab/lerobot-fast/predecode.py: that script
decodes with `torchcodec.VideoDecoder`, and torchcodec cannot load on this
machine -- it links against libavutil .56/.57/.58/.59 (ffmpeg 4-7) and the system
has ffmpeg 8 (libavutil.so.60). The ffmpeg CLI itself works fine, so this does
the same job through it.

Its companion, `lerobot-fast/predecoded_patch.py`, needs NO changes: it reads
JPGs with `torchvision.io.read_image` and never imports torchcodec. Only the
one-shot decode was blocked.

WHY IT PAYS. Measured on this rig mid-run (ACT, batch 8, 2 cameras):
    updt_s 0.105   data_s 0.061      -> ~37% of every step is video decode
Removing it takes ~0.166 s/step to ~0.105, about 1.6x.

FRAME INDEXING IS THE WHOLE CORRECTNESS STORY. `predecoded_patch` maps a
timestamp to a file as `round(ts * fps)`, so JPG `f{i:06d}.jpg` MUST be the i-th
frame of the mp4, zero-indexed, in presentation order. ffmpeg's image2 muxer
numbers from 1 by default, hence `-start_number 0`. A silent off-by-one here
trains the policy on actions belonging to the previous frame, which does not
raise, does not show in the loss, and costs a run. `--verify` checks it.

LeRobot v3.0 packs many episodes into few mp4s (this dataset: 3 files, 13570
frames), so parallelism is per-FILE and small. Workers are capped and niced so a
training run on the same box keeps its dataloader threads.

Usage:
    uv run python tools/predecode_ffmpeg.py \
        --dataset-root ~/.cache/huggingface/lerobot/ETHRC/yam_grasp_v1 \
        --output-root  ~/.cache/lerobot-predecoded/yam_grasp_v1
    uv run python tools/predecode_ffmpeg.py --dataset-root ... --output-root ... --verify

Then train with:
    LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_v1 \
    uv run python tools/train_act_dark_noise.py ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# mjpeg quality. -q:v 2 is the top of the useful range (1 is near-lossless but
# ~2x the bytes for no measurable gain). yuvj444p disables chroma subsampling,
# matching the PIL `subsampling=0` the original predecode used -- the wrist cam
# is only 640x480 and chroma loss there is real.
FFMPEG_QUALITY = ["-q:v", "2", "-pix_fmt", "yuvj444p"]


def probe_frame_count(mp4: Path) -> int:
    """Exact decoded frame count. -count_frames is slow but authoritative;
    the nb_frames metadata field lies on files written by some encoders."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.split(",")[0])


def decode_one(mp4: Path, out_dir: Path, force: bool) -> tuple[Path, int, int, str]:
    expected = probe_frame_count(mp4)
    have = len(list(out_dir.glob("f*.jpg"))) if out_dir.is_dir() else 0

    if not force and have == expected and expected > 0:
        return (mp4, 0, expected, "cached")
    if have and have != expected:
        # A partial extraction is worse than none: the gap is invisible until a
        # training step asks for the missing index hours in.
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(mp4),
         "-start_number", "0", *FFMPEG_QUALITY, str(out_dir / "f%06d.jpg")],
        check=True, capture_output=True,
    )

    wrote = len(list(out_dir.glob("f*.jpg")))
    if wrote != expected:
        raise RuntimeError(f"{mp4.name}: ffmpeg wrote {wrote} JPGs, expected {expected}")
    return (mp4, wrote, expected, "ok")


def verify(ds_root: Path, out_root: Path, n_probe: int) -> int:
    """Compare JPGs against a real video decode. This is the check that a
    training run cannot do for you."""
    import numpy as np
    import torch
    from torchvision.io import ImageReadMode, read_image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lerobot.datasets.video_utils import decode_video_frames

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    bad = 0

    for mp4 in sorted((ds_root / "videos").rglob("*.mp4")):
        folder = out_root / mp4.relative_to(ds_root).with_suffix("")
        n = len(list(folder.glob("f*.jpg")))
        if n == 0:
            print(f"  FAIL {mp4.name}: no JPGs at {folder}"); bad += 1; continue

        idxs = [0, 1, n // 3, n // 2, (2 * n) // 3, n - 2, n - 1][:n_probe]
        idxs = sorted({i for i in idxs if 0 <= i < n})
        ts = [i / fps for i in idxs]
        ref = decode_video_frames(mp4, ts, 1.0 / fps, backend="pyav")  # (N,C,H,W) float [0,1]

        worst = 0.0
        for k, i in enumerate(idxs):
            jpg = read_image(str(folder / f"f{i:06d}.jpg"), mode=ImageReadMode.RGB)
            jpg = jpg.to(torch.float32) / 255.0
            d = float((jpg - ref[k]).abs().mean())
            worst = max(worst, d)
        # JPEG is lossy, so exact equality is not the bar. An index shift shows
        # up as ~0.05-0.20 mean abs diff; re-encode noise is <0.02.
        status = "OK  " if worst < 0.03 else "FAIL"
        if worst >= 0.03:
            bad += 1
        print(f"  {status} {mp4.relative_to(ds_root)}  frames={n}  "
              f"worst mean|diff|={worst:.4f} over idx {idxs}")

    print("\nverify: PASS" if not bad else f"\nverify: {bad} FILE(S) FAILED")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel ffmpeg processes; keep low if a training run shares the box")
    ap.add_argument("--force", action="store_true", help="re-decode even if cached")
    ap.add_argument("--verify", action="store_true", help="check JPGs against a real decode, then exit")
    ap.add_argument("--probe", type=int, default=7, help="frames per file to verify")
    a = ap.parse_args(argv)

    ds_root = a.dataset_root.expanduser().resolve()
    out_root = a.output_root.expanduser().resolve()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe not on PATH", file=sys.stderr)
        return 1
    if not (ds_root / "meta" / "info.json").exists():
        print(f"not a LeRobot dataset: {ds_root}", file=sys.stderr)
        return 1

    if a.verify:
        return verify(ds_root, out_root, a.probe)

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    videos = sorted((ds_root / "videos").rglob("*.mp4"))
    if not videos:
        print(f"no mp4s under {ds_root/'videos'}", file=sys.stderr)
        return 1

    print(f"dataset {ds_root}")
    print(f"  {info.get('total_episodes')} episodes, {info.get('total_frames')} frames, "
          f"{info.get('fps')} fps, {len(videos)} mp4 file(s)")
    print(f"  -> {out_root}  (workers={a.workers})")

    total = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:   # ffmpeg is the process; threads just wait
        futs = {ex.submit(decode_one, v,
                          out_root / v.relative_to(ds_root).with_suffix(""), a.force): v
                for v in videos}
        for i, f in enumerate(as_completed(futs), 1):
            v = futs[f]
            try:
                _, wrote, exp, how = f.result()
            except Exception as e:
                print(f"  [{i}/{len(videos)}] FAIL {v.name}: {e}", file=sys.stderr)
                return 2
            total += exp
            print(f"  [{i}/{len(videos)}] {how:6} {v.relative_to(ds_root)}  "
                  f"wrote={wrote} total={exp}", flush=True)

    size = sum(p.stat().st_size for p in out_root.rglob("f*.jpg")) / 1e9
    print(f"\n{total} frames on disk, {size:.2f} GB")
    print(f"\nNow verify, then train:\n"
          f"  python tools/predecode_ffmpeg.py --dataset-root {ds_root} "
          f"--output-root {out_root} --verify\n"
          f"  LEROBOT_PREDECODED_ROOT={out_root} python tools/train_act_dark_noise.py ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Coverage for tools/wrist_view.py — the wrist-camera reader.

The tool exists to make a stream a human can inspect, so the tests are about
the ways an unreadable stream can still LOOK read: a decoder that returns
nothing without raising, a strided decode reporting its sample count as the
frame count, a frozen camera re-serving one buffer, a video and its timestamp
sidecar disagreeing on length.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrist_view import (  # noqa: E402
    FROZEN_FRAME_MAD, WRIST_KEYS, WristStream, contact_sheet, decode_frames,
    recording_stream, recording_streams, sharpness, summarise,
)

DEFAULT_CAMERA = "camera_left"


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        st = out.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(f), format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)


def _noise(n: int, h: int = 48, w: int = 64, seed: int = 0) -> list[np.ndarray]:
    """Frames with real high-frequency content, so sharpness is meaningful and
    consecutive frames genuinely differ."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _episode(tmp: Path, frames: list[np.ndarray], stamps=None,
             camera: str = DEFAULT_CAMERA) -> Path:
    ep = tmp / "episode_x"
    ep.mkdir(parents=True, exist_ok=True)
    _write_video(ep / f"{camera}-images-rgb.mp4", frames)
    if stamps is not None:
        np.save(ep / f"{camera}-rgb-timestamp.npy", np.asarray(stamps, dtype=float))
    return ep


# ── decoding ────────────────────────────────────────────────────────────────
def test_decodes_a_recording_video(tmp_path):
    ep = _episode(tmp_path, _noise(20), np.arange(20) / 30.0)
    got = list(decode_frames(ep / f"{DEFAULT_CAMERA}-images-rgb.mp4"))
    assert len(got) == 20
    assert got[0][2].shape == (48, 64, 3)


def test_stride_samples_without_changing_the_reported_frame_count(tmp_path):
    """The reported count must be the CONTAINER's, not the sample count. A
    summary that says "3 frames" for a 30-frame episode because it sampled every
    tenth is a plausible wrong number, which is the class of bug this whole
    tool exists to surface."""
    ep = _episode(tmp_path, _noise(30), np.arange(30) / 30.0)
    s = recording_stream(ep)
    assert len(list(s.frames(stride=10))) == 3
    assert summarise(s, stride=10).frames_decoded == 30


def test_a_lerobot_episode_reports_its_own_length_not_the_shared_files(tmp_path):
    """Measured against the real dataset: every yam_grasp_v2_wrist episode is a
    2.5-4.8 s slice of one 8962-frame chunk file. Reporting the container count
    would say "8962 frames" for all 69 of them -- a number that looks like a
    measurement and is the same for every row."""
    mp4 = tmp_path / "shared.mp4"
    _write_video(mp4, _noise(60))
    s = WristStream("ep0000", mp4, None, start_s=0.5, end_s=1.0,
                    source="lerobot", n_frames=15)
    assert summarise(s, stride=1).frames_decoded == 15


def test_a_timestamp_range_selects_a_slice_of_a_shared_video(tmp_path):
    """How LeRobot v3 stores episodes: many in one mp4, each a [from, to)
    timestamp range. Reading the whole file as one episode is the easy mistake."""
    mp4 = tmp_path / "shared.mp4"
    _write_video(mp4, _noise(60))
    s = WristStream("ep0000", mp4, None, start_s=0.5, end_s=1.0, source="lerobot")
    ts = [t for _, t, _ in s.frames()]
    assert ts and all(0.5 <= t < 1.0 for t in ts)
    assert len(ts) < 60


# ── health measurement ──────────────────────────────────────────────────────
def test_summary_reports_resolution_duration_and_rate(tmp_path):
    ep = _episode(tmp_path, _noise(60), np.arange(60) / 30.0)
    s = summarise(recording_stream(ep), stride=1)
    assert (s.width, s.height) == (64, 48)
    assert s.frames_decoded == 60
    assert s.stamps == 60
    assert s.fps_measured == pytest.approx(30.0, rel=0.05)
    assert s.problems == []


def test_a_frozen_camera_is_reported(tmp_path):
    """A USB webcam that stops delivering re-serves its last buffer. Every
    consumer downstream reads those as valid pictures at 30 Hz."""
    frozen = [_noise(1)[0]] * 40
    ep = _episode(tmp_path, frozen, np.arange(40) / 30.0)
    s = summarise(recording_stream(ep), stride=1)
    assert s.frozen_frac > 0.9
    assert any("identical" in p for p in s.problems)


def test_a_live_camera_is_not_reported_as_frozen(tmp_path):
    ep = _episode(tmp_path, _noise(40), np.arange(40) / 30.0)
    s = summarise(recording_stream(ep), stride=1)
    assert s.frozen_frac < 0.1
    assert not any("identical" in p for p in s.problems)


def test_a_black_stream_is_reported(tmp_path):
    ep = _episode(tmp_path, [np.zeros((48, 64, 3), np.uint8)] * 20,
                  np.arange(20) / 30.0)
    s = summarise(recording_stream(ep), stride=1)
    assert any("black" in p for p in s.problems)


def test_a_video_shorter_than_its_sidecar_is_reported(tmp_path):
    """DATA-PIPELINE 2.6: a disk-full kills the encoder thread, close() joins a
    dead thread instantly and np.save writes the partial list -- so the mp4 and
    the .npy stay mutually consistent and that camera is just silently shorter
    than the rest of the episode. Here the sidecar is the longer one, which is
    the same disagreement seen from the other side."""
    ep = _episode(tmp_path, _noise(20), np.arange(35) / 30.0)
    s = summarise(recording_stream(ep), stride=1)
    assert s.frame_stamp_mismatch == -15
    assert any("timestamp sidecar" in p for p in s.problems)


def test_non_monotonic_timestamps_are_counted(tmp_path):
    """DATA-PIPELINE 2.5. The sidecar is written straight from arrival order
    with no np.sort, and export_lerobot's nearest_index then calls
    np.searchsorted, which does not validate that its input is sorted and
    silently returns a wrong index instead of raising."""
    st = np.arange(20) / 30.0
    st[10] = st[3]                      # a pipeline restart rewound the clock
    ep = _episode(tmp_path, _noise(20), st)
    s = summarise(recording_stream(ep), stride=1)
    assert s.stamp_non_monotonic >= 1


def test_a_missing_sidecar_is_reported_not_ignored(tmp_path):
    """Two of the 30 real episodes have no camera_left timestamp sidecar. With
    no timeline, nothing that resamples by timestamp can use that episode."""
    ep = _episode(tmp_path, _noise(20), stamps=None)
    s = summarise(recording_stream(ep), stride=1)
    assert s.stamps is None
    assert any("sidecar" in p for p in s.problems)


def test_sharpness_separates_a_blurred_frame_from_a_sharp_one():
    sharp = _noise(1)[0]
    blurred = np.repeat(np.repeat(sharp[::8, ::8], 8, axis=0), 8, axis=1)
    assert sharpness(sharp) > sharpness(blurred) * 2


# ── rendering ───────────────────────────────────────────────────────────────
def test_contact_sheet_lays_out_a_grid_of_the_right_size():
    sheet = contact_sheet(_noise(7, h=48, w=64), cols=3, cell_w=32)
    assert sheet.shape[1] == 3 * 32
    assert sheet.shape[0] >= 3 * 24            # 3 rows for 7 images at 3 cols
    assert sheet.dtype == np.uint8


def test_contact_sheet_preserves_content_rather_than_blanking():
    sheet = contact_sheet(_noise(4), cols=2, cell_w=64)
    assert sheet.std() > 10, "cells look empty"


def test_contact_sheet_refuses_an_empty_list():
    with pytest.raises(ValueError):
        contact_sheet([])


# ── discovery ───────────────────────────────────────────────────────────────
def test_walking_a_tree_finds_episodes_and_skips_trash(tmp_path):
    """.trash holds episodes the operator threw away; delete is a move so it
    stays undoable, but they must never be reviewed as if they were corpus."""
    root = tmp_path / "recordings" / "20260808"
    good = root / "episode_a"
    good.mkdir(parents=True)
    _write_video(good / f"{DEFAULT_CAMERA}-images-rgb.mp4", _noise(4))
    trash = tmp_path / "recordings" / ".trash" / "20260808" / "episode_b"
    trash.mkdir(parents=True)
    _write_video(trash / f"{DEFAULT_CAMERA}-images-rgb.mp4", _noise(4))

    names = [s.name for s in recording_streams(tmp_path / "recordings")]
    assert names == ["episode_a"]


def test_a_single_episode_dir_is_accepted_directly(tmp_path):
    ep = _episode(tmp_path, _noise(4))
    assert [s.name for s in recording_streams(ep)] == [ep.name]


def test_camera_left_is_the_wrist_and_the_lerobot_keys_cover_both_arms():
    assert DEFAULT_CAMERA == "camera_left"
    assert "observation.images.wrist" in WRIST_KEYS
    assert "observation.images.wrist_left" in WRIST_KEYS
    assert "observation.images.wrist_right" in WRIST_KEYS


def test_frozen_threshold_is_above_zero_but_below_sensor_noise():
    """0.0 would only catch a byte-identical frame; a real 640x480 webcam on a
    static scene still moves ~0.3-1.0 of mean absolute difference."""
    assert 0.0 < FROZEN_FRAME_MAD < 0.3

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robots_realtime.agents.policy_learning.pixel_idm_replay_agent import (
    DEFAULT_LEFT_LIMITS_PATH,
    DEFAULT_RIGHT_LIMITS_PATH,
    command_state_from_observation,
    interpolate_action_window,
    validate_pixel_idm_window,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _image_from_obs(obs: dict[str, Any], image_key: str) -> np.ndarray | None:
    data = obs.get(image_key)
    if not isinstance(data, dict):
        return None
    images = data.get("images")
    if not isinstance(images, dict):
        return None
    img = images.get("rgb")
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _resize_pad(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - new_h) // 2
    x = (width - new_w) // 2
    out[y : y + new_h, x : x + new_w] = resized
    return out


def _wrist_mosaic_from_obs(obs: dict[str, Any], image_keys: dict[str, str]) -> np.ndarray | None:
    top = _image_from_obs(obs, image_keys["top"])
    left = _image_from_obs(obs, image_keys["left"])
    right = _image_from_obs(obs, image_keys["right"])
    if top is None or left is None or right is None:
        return None
    out = np.zeros((960, 960, 3), dtype=np.uint8)
    out[:540, :960] = _resize_pad(top, 960, 540)
    out[540:900, :480] = _resize_pad(left, 480, 360)
    out[540:900, 480:960] = _resize_pad(right, 480, 360)
    return np.ascontiguousarray(out)


def _command_from_row(row: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    left = np.asarray(row[:7], dtype=np.float32).copy()
    right = np.asarray(row[7:], dtype=np.float32).copy()
    left[-1] = np.clip(left[-1], 0.0, 1.0)
    right[-1] = np.clip(right[-1], 0.0, 1.0)
    return {"left": {"pos": left}, "right": {"pos": right}}


def _write_frame_video(path: Path, frames_rgb: list[np.ndarray], *, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("frames_rgb must not be empty")
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for frame_rgb in frames_rgb:
            if frame_rgb.shape[:2] != (h, w):
                frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video for preview: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video preview had no frames: {path}")
    return frames


def _chunk_payload(chunk: np.ndarray, start: int = 0) -> dict[str, np.ndarray]:
    remaining = chunk[min(start, len(chunk) - 1) :]
    return {
        "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
        "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
    }


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc


def _post_vllm_video_sync(
    url: str,
    *,
    prompt: str,
    input_reference: Path,
    output_path: Path,
    size: str,
    num_frames: int,
    fps: float,
    num_inference_steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
    negative_prompt: str,
    guardrails: bool,
    condition_video_keep: str,
    timeout_s: float,
) -> None:
    boundary = f"----rlteleop-cosmos-wrist-cams-{time.time_ns()}"

    def add_field(parts: list[bytes], name: str, value: str) -> None:
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    extra_params = {
        "condition_video_keep": condition_video_keep,
        "guardrails": guardrails,
        "use_duration_template": False,
        "use_resolution_template": False,
    }
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "size": size,
        "num_frames": str(int(num_frames)),
        "fps": str(float(fps)),
        "num_inference_steps": str(int(num_inference_steps)),
        "guidance_scale": str(float(guidance_scale)),
        "flow_shift": str(float(flow_shift)),
        "seed": str(int(seed)),
        "extra_params": json.dumps(extra_params),
    }
    parts: list[bytes] = []
    for name, value in data.items():
        add_field(parts, name, value)
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="input_reference"; '
                f'filename="{input_reference.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: video/mp4\r\n\r\n",
            input_reference.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    request = urllib.request.Request(
        url,
        data=b"".join(parts),
        headers={
            "Accept": "video/mp4",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc


def _video_player_command(path: Path, player_cmd: str | None) -> list[str] | None:
    path_str = str(path)
    if player_cmd:
        parts = shlex.split(player_cmd)
        if any("{path}" in part for part in parts):
            return [part.replace("{path}", path_str) for part in parts]
        return [*parts, path_str]

    mpv = shutil.which("mpv")
    if mpv:
        return [mpv, "--loop-file=inf", "--force-window=yes", path_str]

    vlc = shutil.which("vlc") or shutil.which("cvlc")
    if vlc:
        return [vlc, "--loop", path_str]

    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        return [xdg_open, path_str]

    return None


class CosmosWristCamsPixelIDMChunkAgent:
    use_joint_state_as_action = False

    def __init__(
        self,
        *,
        cosmos_url: str | None = None,
        pixel_idm_url: str | None = None,
        prompt: str | None = None,
        source_hz: float = 10.0,
        command_hz: float = 30.0,
        cosmos_video_fps: float = 10.0,
        validate_only: bool | None = None,
        image_key: str = "top_camera",
        image_keys: dict[str, str] | None = None,
        run_root: str = "/tmp/cosmos_pixel_idm_chunk_policy",
        array_key: str = "predicted_actions",
        request_timeout_s: float = 900.0,
        num_latent_conditional_frames: int = 2,
        vllm_size: str = "640x480",
        vllm_num_frames: int = 61,
        vllm_initial_num_frames: int | None = None,
        vllm_num_inference_steps: int = 35,
        vllm_guidance_scale: float = 6.0,
        vllm_flow_shift: float = 10.0,
        vllm_seed: int = 0,
        vllm_negative_prompt: str = "blurry, distorted, low quality",
        vllm_guardrails: bool = True,
        vllm_condition_video_keep: str = "first",
        open_video_plan: bool | None = None,
        video_player_cmd: str | None = None,
        left_limits_path: str = DEFAULT_LEFT_LIMITS_PATH,
        right_limits_path: str = DEFAULT_RIGHT_LIMITS_PATH,
        limit_tolerance: float = 5e-2,
        gripper_tolerance: float = 1e-6,
        gripper_clamp_tolerance: float = 2e-2,
        max_arm_frame_delta_rad: float = 0.5,
        max_gripper_frame_delta: float = 0.25,
        max_handoff_distance: float = 0.75,
    ) -> None:
        if source_hz <= 0:
            raise ValueError(f"source_hz must be > 0, got {source_hz}")
        if command_hz <= 0:
            raise ValueError(f"command_hz must be > 0, got {command_hz}")
        if request_timeout_s <= 0:
            raise ValueError(f"request_timeout_s must be > 0, got {request_timeout_s}")

        self.cosmos_url = (cosmos_url or os.environ.get("COSMOS_PIXEL_IDM_COSMOS_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.pixel_idm_url = (
            pixel_idm_url or os.environ.get("COSMOS_PIXEL_IDM_PIXEL_URL") or "http://127.0.0.1:8022"
        ).rstrip("/")
        self.prompt = prompt or os.environ.get("COSMOS_PIXEL_IDM_PROMPT")
        if not self.prompt:
            raise ValueError("CosmosWristCamsPixelIDMChunkAgent requires a prompt or COSMOS_PIXEL_IDM_PROMPT")

        self.source_hz = float(source_hz)
        self.command_hz = float(command_hz)
        self.cosmos_video_fps = float(cosmos_video_fps)
        self.validate_only = _env_bool("COSMOS_PIXEL_IDM_VALIDATE_ONLY", True) if validate_only is None else bool(validate_only)
        self.image_key = image_key
        self.image_keys = image_keys
        self.run_root = Path(run_root).expanduser()
        self.array_key = array_key
        self.request_timeout_s = float(request_timeout_s)
        self.num_latent_conditional_frames = int(num_latent_conditional_frames)
        if self.num_latent_conditional_frames not in (1, 2):
            raise ValueError(
                "num_latent_conditional_frames must be 1 or 2, "
                f"got {self.num_latent_conditional_frames}"
            )
        if vllm_num_frames <= 0:
            raise ValueError(f"vllm_num_frames must be > 0, got {vllm_num_frames}")
        if vllm_initial_num_frames is not None and vllm_initial_num_frames <= 0:
            raise ValueError(f"vllm_initial_num_frames must be > 0, got {vllm_initial_num_frames}")
        if vllm_num_inference_steps <= 0:
            raise ValueError(f"vllm_num_inference_steps must be > 0, got {vllm_num_inference_steps}")
        if vllm_condition_video_keep not in {"first", "last"}:
            raise ValueError(
                "vllm_condition_video_keep must be 'first' or 'last', "
                f"got {vllm_condition_video_keep!r}"
            )
        self.vllm_size = vllm_size
        self.vllm_num_frames = int(vllm_num_frames)
        self.vllm_initial_num_frames = None if vllm_initial_num_frames is None else int(vllm_initial_num_frames)
        self.vllm_num_inference_steps = int(vllm_num_inference_steps)
        self.vllm_guidance_scale = float(vllm_guidance_scale)
        self.vllm_flow_shift = float(vllm_flow_shift)
        self.vllm_seed = int(vllm_seed)
        self.vllm_negative_prompt = vllm_negative_prompt
        self.vllm_guardrails = bool(vllm_guardrails)
        self.vllm_condition_video_keep = vllm_condition_video_keep
        self.left_limits_path = left_limits_path
        self.right_limits_path = right_limits_path
        self.open_video_plan = (
            _env_bool("COSMOS_PIXEL_IDM_OPEN_VIDEO_PLAN", True)
            if open_video_plan is None
            else bool(open_video_plan)
        )
        self.video_player_cmd = video_player_cmd or os.environ.get("COSMOS_PIXEL_IDM_VIDEO_PLAYER")
        self.limit_tolerance = float(limit_tolerance)
        self.gripper_tolerance = float(gripper_tolerance)
        self.gripper_clamp_tolerance = float(gripper_clamp_tolerance)
        self.max_arm_frame_delta_rad = float(max_arm_frame_delta_rad)
        self.max_gripper_frame_delta = float(max_gripper_frame_delta)
        self.max_handoff_distance = float(max_handoff_distance)

        self._lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._planning = False
        self._active_chunk: np.ndarray | None = None
        self._active_display_image: np.ndarray | None = None
        self._plan_video_frames: list[np.ndarray] = []
        self._plan_video_started_at = 0.0
        self._plan_video_fps = self.cosmos_video_fps
        self._cursor = 0
        self._final_action: np.ndarray | None = None
        self._last_error: str | None = None
        self._plan_index = 0
        self._image_history: deque[np.ndarray] = deque(maxlen=4 * (self.num_latent_conditional_frames - 1) + 1)
        self._last_history_t = 0.0
        self._video_player_proc: subprocess.Popen[bytes] | None = None
        print(
            "[CosmosWristCamsPixelIDMChunkAgent] ready "
            f"validate_only={self.validate_only} source_hz={self.source_hz} command_hz={self.command_hz} "
            f"cosmos_url={self.cosmos_url} pixel_idm_url={self.pixel_idm_url} "
            f"num_frames={self.vllm_num_frames} initial_num_frames={self.vllm_initial_num_frames} "
            f"open_video_plan={self.open_video_plan}",
            flush=True,
        )

    def reset(self) -> None:
        with self._lock:
            self._active_chunk = None
            self._active_display_image = None
            self._plan_video_frames = []
            self._plan_video_started_at = 0.0
            self._cursor = 0
            self._final_action = None
            self._last_error = None
            self._latest_obs = None
            self._planning = False
            self._last_history_t = 0.0
        self._close_video_player()

    def close(self) -> None:
        self._close_video_player()

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        image = (
            _wrist_mosaic_from_obs(obs, self.image_keys)
            if self.image_keys is not None
            else _image_from_obs(obs, self.image_key)
        )
        left_state = command_state_from_observation(obs.get("left", {}))
        right_state = command_state_from_observation(obs.get("right", {}))
        if image is None or left_state is None or right_state is None:
            return {}
        paused = bool(obs.get("_paused"))

        with self._lock:
            self._latest_obs = {
                "image": image.copy(),
                "left": left_state.copy(),
                "right": right_state.copy(),
            }
            history_period_s = 1.0 / self.cosmos_video_fps
            now = time.monotonic()
            if self._last_history_t == 0.0 or (now - self._last_history_t) >= history_period_s:
                self._image_history.append(image.copy())
                self._last_history_t = now
            planning = self._planning
            active = self._active_chunk
            cursor = self._cursor
            final_action = None if self._final_action is None else self._final_action.copy()
            display_image = None if self._active_display_image is None else self._active_display_image.copy()

        if not planning and (active is None or cursor >= len(active)):
            self._start_planning()

        if active is None:
            if final_action is None or self.validate_only or paused:
                return self._meta_action(display_image, None)
            action = _command_from_row(final_action)
            action.update(self._meta_action(display_image, None))
            return action

        if self.validate_only or paused:
            return self._meta_action(display_image, active)

        with self._lock:
            active = self._active_chunk
            if active is None:
                return {}
            idx = min(self._cursor, len(active) - 1)
            row = active[idx].copy()
            self._final_action = active[-1].copy()
            if self._cursor < len(active):
                self._cursor += 1
            display_image = None if self._active_display_image is None else self._active_display_image.copy()
            next_cursor = self._cursor

        action = _command_from_row(row)
        action.update(self._meta_action(display_image, active, next_cursor))
        return action

    def _meta_action(
        self,
        display_image: np.ndarray | None,
        chunk: np.ndarray | None,
        cursor: int = 0,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if chunk is not None:
            out["_chunk"] = _chunk_payload(chunk, cursor)
        if display_image is not None:
            out["_images"] = {"top_camera": display_image}
        plan_frame = self._current_plan_video_frame()
        if plan_frame is not None:
            out.setdefault("_images", {})["cosmos_plan"] = plan_frame
        return out

    def _current_plan_video_frame(self) -> np.ndarray | None:
        with self._lock:
            frames = self._plan_video_frames
            started_at = self._plan_video_started_at
            fps = self._plan_video_fps
        if not frames:
            return None
        idx = int(max(0.0, time.monotonic() - started_at) * max(1.0, fps)) % len(frames)
        return frames[idx].copy()

    def _start_planning(self) -> None:
        with self._lock:
            if self._planning or self._latest_obs is None:
                return
            self._planning = True
        thread = threading.Thread(target=self._planning_worker, name="CosmosWristCamsPixelIDMChunkAgent_planner", daemon=True)
        thread.start()

    def _planning_worker(self) -> None:
        with self._lock:
            obs = None if self._latest_obs is None else dict(self._latest_obs)
            plan_index = self._plan_index
            self._plan_index += 1
        if obs is None:
            with self._lock:
                self._planning = False
            return

        artifact_dir = self.run_root / time.strftime("%Y%m%d_%H%M%S") / f"plan_{plan_index:04d}"
        try:
            t0 = time.monotonic()
            num_frames = self._num_frames_for_plan(plan_index)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            image_path = artifact_dir / "cosmos_input.png"
            conditioning_path = artifact_dir / "cosmos_conditioning.mp4"
            video_path = artifact_dir / "cosmos_wrist_cams_plan.mp4"
            action_path = artifact_dir / "pixel_idm_actions.npz"
            cv2.imwrite(str(image_path), cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR))
            frames_to_extract = 4 * (self.num_latent_conditional_frames - 1) + 1
            with self._lock:
                history = [frame.copy() for frame in self._image_history]
            if not history:
                history = [obs["image"].copy()]
            while len(history) < frames_to_extract:
                history.insert(0, history[0].copy())
            history = history[-frames_to_extract:]
            _write_frame_video(conditioning_path, history, fps=self.cosmos_video_fps)

            _post_vllm_video_sync(
                f"{self.cosmos_url}/v1/videos/sync",
                prompt=self.prompt,
                input_reference=conditioning_path,
                output_path=video_path,
                size=self.vllm_size,
                num_frames=num_frames,
                fps=self.cosmos_video_fps,
                num_inference_steps=self.vllm_num_inference_steps,
                guidance_scale=self.vllm_guidance_scale,
                flow_shift=self.vllm_flow_shift,
                seed=self.vllm_seed,
                negative_prompt=self.vllm_negative_prompt,
                guardrails=self.vllm_guardrails,
                condition_video_keep=self.vllm_condition_video_keep,
                timeout_s=self.request_timeout_s,
            )
            video_fps = self.cosmos_video_fps
            plan_video_frames = _read_video_frames(video_path)
            with self._lock:
                self._plan_video_frames = plan_video_frames
                self._plan_video_started_at = time.monotonic()
                self._plan_video_fps = video_fps
            self._open_video_plan(video_path)

            pixel_response = _post_json(
                f"{self.pixel_idm_url}/infer",
                {
                    "video_path": str(video_path),
                    "output_path": str(action_path),
                    "source_hz": self.source_hz,
                    "video_fps": video_fps,
                    "array_key": self.array_key,
                },
                self.request_timeout_s,
            )
            npz_path = Path(pixel_response.get("action_path") or action_path)
            source_actions = self._load_actions(npz_path)

            chunk = interpolate_action_window(source_actions, source_hz=self.source_hz, command_hz=self.command_hz)
            chunk, report = validate_pixel_idm_window(
                chunk,
                source_name=str(npz_path),
                source_shape=tuple(source_actions.shape),
                left_limits_path=self.left_limits_path,
                right_limits_path=self.right_limits_path,
                limit_tolerance=self.limit_tolerance,
                gripper_tolerance=self.gripper_tolerance,
                gripper_clamp_tolerance=self.gripper_clamp_tolerance,
                max_arm_frame_delta_rad=self.max_arm_frame_delta_rad,
                max_gripper_frame_delta=self.max_gripper_frame_delta,
                source_hz=None,
                command_hz=None,
            )
            left_distance = float(np.linalg.norm(chunk[0, :7] - obs["left"]))
            right_distance = float(np.linalg.norm(chunk[0, 7:] - obs["right"]))
            if max(left_distance, right_distance) > self.max_handoff_distance:
                raise ValueError(
                    "generated chunk first action is too far from live robot state "
                    f"(left={left_distance:.6f}, right={right_distance:.6f}, "
                    f"max={self.max_handoff_distance:.6f})"
                )
            print(report.with_handoff(left_distance, right_distance).format(), flush=True)
            if self.validate_only:
                print("[CosmosWristCamsPixelIDMChunkAgent] validate_only=true; chunk will not command hardware", flush=True)
            print(
                f"[CosmosWristCamsPixelIDMChunkAgent] plan {plan_index} ready: "
                f"{len(chunk)} command ticks in {time.monotonic() - t0:.2f}s "
                f"num_frames={num_frames} video={video_path} actions={npz_path}",
                flush=True,
            )

            with self._lock:
                self._active_chunk = chunk
                self._active_display_image = obs["image"].copy()
                self._cursor = 0
                self._last_error = None
        except Exception as exc:
            print(f"[CosmosWristCamsPixelIDMChunkAgent] planning failed: {exc}", flush=True)
            traceback.print_exc()
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._planning = False

    def _num_frames_for_plan(self, plan_index: int) -> int:
        if plan_index == 0 and self.vllm_initial_num_frames is not None:
            return self.vllm_initial_num_frames
        return self.vllm_num_frames

    def _open_video_plan(self, video_path: Path) -> None:
        if not self.open_video_plan:
            return
        if not video_path.exists():
            print(f"[CosmosWristCamsPixelIDMChunkAgent] video plan does not exist yet: {video_path}", flush=True)
            return
        command = _video_player_command(video_path, self.video_player_cmd)
        if command is None:
            print(
                "[CosmosWristCamsPixelIDMChunkAgent] no video player found; install mpv/vlc or set "
                "COSMOS_PIXEL_IDM_VIDEO_PLAYER",
                flush=True,
            )
            return

        self._close_video_player()
        try:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"[CosmosWristCamsPixelIDMChunkAgent] failed to open video plan with {command}: {exc}", flush=True)
            return

        with self._lock:
            self._video_player_proc = proc
        print(f"[CosmosWristCamsPixelIDMChunkAgent] opened video plan: {video_path}", flush=True)
        if Path(command[0]).name == "xdg-open":
            print(
                "[CosmosWristCamsPixelIDMChunkAgent] xdg-open does not provide reliable loop/close control; "
                "install mpv or vlc for that behavior",
                flush=True,
            )

    def _close_video_player(self) -> None:
        with self._lock:
            proc = self._video_player_proc
            self._video_player_proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)

    def _load_actions(self, npz_path: Path) -> np.ndarray:
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path) as data:
            if self.array_key not in data:
                raise ValueError(f"{npz_path} does not contain {self.array_key!r}; keys={list(data.keys())}")
            actions = np.asarray(data[self.array_key], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"{self.array_key!r} must have shape (T, 14) or (1, T, 14), got {actions.shape}")
        if actions.shape[0] < 2:
            raise ValueError(f"{self.array_key!r} must contain at least two actions, got {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32)

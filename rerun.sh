#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDINGS_DIR="$ROOT_DIR/recordings"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    export PATH="$ROOT_DIR/.venv/bin:$PATH"
    PYTHON_CMD=("$ROOT_DIR/.venv/bin/python")
else
    PYTHON_CMD=(uv run python)
fi

usage() {
    cat <<'EOF'
Usage:
  ./rerun.sh [--instruction TEXT] [--limit N] [--stride N] [episode-name|date/episode-name|episode-path ...]
  ./rerun.sh --set-instruction EPISODE TEXT
  ./rerun.sh --replace-instruction OLD_TEXT NEW_TEXT [--dry-run]
  ./rerun.sh --list-instructions
  ./rerun.sh --html [--instruction TEXT]

Examples:
  ./rerun.sh
  ./rerun.sh --list-instructions
  ./rerun.sh --instruction "push the box to the right"
  ./rerun.sh --instruction "push the box to the left" --limit 20
  ./rerun.sh --set-instruction 20260601/episode_183926_13afda0a "push the box to the left"
  ./rerun.sh --replace-instruction "push the box to the right" "push the box to the right with the right arm" --dry-run
  ./rerun.sh --replace-instruction "push the box to the right" "push the box to the right with the right arm"
  ./rerun.sh --html
  ./rerun.sh 20260601/episode_183926_13afda0a

Notes:
  This loads videos as Rerun AssetVideo + VideoFrameReference data. It does not
  decode every frame into Image rows.
  Use --html for the lowest-memory way to browse every video grouped by instruction.
EOF
}

episode_args=()
instruction=""
set_instruction_episode=""
set_instruction_text=""
replace_instruction_old=""
replace_instruction_new=""
dry_run=0
list_instructions=0
html_index=0
stride=1
limit=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --list-instructions)
            list_instructions=1
            shift
            ;;
        --html)
            html_index=1
            shift
            ;;
        --instruction)
            [[ $# -ge 2 ]] || { printf '%s\n' '--instruction requires text' >&2; exit 1; }
            instruction="$2"
            shift 2
            ;;
        --set-instruction)
            [[ $# -ge 3 ]] || { printf '%s\n' '--set-instruction requires EPISODE and TEXT' >&2; exit 1; }
            set_instruction_episode="$2"
            set_instruction_text="$3"
            shift 3
            ;;
        --replace-instruction)
            [[ $# -ge 3 ]] || { printf '%s\n' '--replace-instruction requires OLD_TEXT and NEW_TEXT' >&2; exit 1; }
            replace_instruction_old="$2"
            replace_instruction_new="$3"
            shift 3
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --stride)
            [[ $# -ge 2 ]] || { printf '%s\n' '--stride requires a value' >&2; exit 1; }
            stride="$2"
            shift 2
            ;;
        --limit)
            [[ $# -ge 2 ]] || { printf '%s\n' '--limit requires a value' >&2; exit 1; }
            limit="$2"
            shift 2
            ;;
        --no-mcap|--mcap)
            # Kept for compatibility with older invocations. This viewer is video-only.
            shift
            ;;
        -*)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 1
            ;;
        *)
            episode_args+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$stride" =~ ^[0-9]+$ ]] || [[ "$stride" -lt 1 ]]; then
    printf '%s\n' '--stride must be a positive integer' >&2
    exit 1
fi

if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
    printf '%s\n' '--limit must be a non-negative integer' >&2
    exit 1
fi

cd "$ROOT_DIR"
"${PYTHON_CMD[@]}" - "$RECORDINGS_DIR" "$list_instructions" "$html_index" "$instruction" "$set_instruction_episode" "$set_instruction_text" "$replace_instruction_old" "$replace_instruction_new" "$dry_run" "$stride" "$limit" "${episode_args[@]}" <<'PY'
from __future__ import annotations

import html as html_lib
import json
import shlex
import sys
import textwrap
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


recordings_root = Path(sys.argv[1]).resolve()
list_only = bool(int(sys.argv[2]))
html_index = bool(int(sys.argv[3]))
instruction_filter = sys.argv[4]
set_instruction_episode = sys.argv[5]
set_instruction_text = sys.argv[6]
replace_instruction_old = sys.argv[7]
replace_instruction_new = sys.argv[8]
dry_run = bool(int(sys.argv[9]))
stride = int(sys.argv[10])
limit = int(sys.argv[11])
episode_args = sys.argv[12:]


def episodes() -> list[Path]:
    if not recordings_root.is_dir():
        return []
    return sorted(recordings_root.glob("*/episode_*"))


def metadata(episode: Path) -> dict:
    meta_path = episode / "session_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def write_metadata(episode: Path, meta: dict) -> None:
    meta_path = episode / "session_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def instruction(episode: Path) -> str:
    value = metadata(episode).get("instruction") or ""
    return str(value).strip() or "(no instruction)"


def rel_episode(episode: Path) -> str:
    try:
        return episode.resolve().relative_to(recordings_root).as_posix()
    except ValueError:
        return episode.resolve().as_posix()


def resolve_episode(name: str) -> Path:
    candidates = [
        Path(name),
        Path.cwd() / name,
        recordings_root / name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    matches = [episode for episode in episodes() if episode.name == name]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        choices = "\n".join(f"  {rel_episode(match)}" for match in matches)
        raise SystemExit(f"Ambiguous episode name: {name}\n{choices}")
    raise SystemExit(f"Could not find episode: {name}")


def instruction_counts() -> list[tuple[str, int]]:
    counts = Counter(instruction(episode) for episode in episodes())
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def print_instruction_counts() -> None:
    counts = instruction_counts()
    if not counts:
        print(f"No episodes found under {recordings_root}", file=sys.stderr)
        return
    width = len(str(len(counts)))
    for i, (name, count) in enumerate(counts, start=1):
        print(f"{i:>{width}}. {name} ({count})")


def choose_instruction() -> str:
    counts = instruction_counts()
    if not counts:
        raise SystemExit(f"No episodes found under {recordings_root}")
    try:
        tty = open("/dev/tty", "r")
    except OSError:
        print_instruction_counts()
        raise SystemExit("\nPass --instruction TEXT to open one group.")

    print("Instructions:")
    width = len(str(len(counts)))
    for i, (name, count) in enumerate(counts, start=1):
        print(f"{i:>{width}}. {name} ({count})")
    print("\nOpen instruction number: ", end="", flush=True)
    choice = tty.readline().strip()
    tty.close()
    try:
        index = int(choice)
    except ValueError:
        raise SystemExit(f"Not a number: {choice}")
    if index < 1 or index > len(counts):
        raise SystemExit(f"Choice out of range: {choice}")
    return counts[index - 1][0]


def first_timestamp(episode: Path) -> float:
    starts: list[float] = []
    for ts_file in episode.glob("*-rgb-timestamp.npy"):
        arr = np.load(ts_file)
        if len(arr):
            starts.append(float(arr[0]))
    return min(starts) if starts else 0.0


def video_timeline_seconds(mp4: Path, episode: Path, camera: str, frame_timestamps_ns: np.ndarray, t0: float) -> np.ndarray:
    ts_file = episode / f"{camera}-rgb-timestamp.npy"
    if ts_file.exists():
        timestamps = np.load(ts_file).astype(np.float64)
        n = min(len(timestamps), len(frame_timestamps_ns))
        return timestamps[:n] - t0
    return frame_timestamps_ns.astype(np.float64) * 1e-9


def safe_entity_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def set_instruction_command(episode: Path, value: str | None = None) -> str:
    text = instruction(episode) if value is None else value
    return f"./rerun.sh --set-instruction {shlex.quote(rel_episode(episode))} {shlex.quote(text)}"


def log_text(text: str) -> rr.TextDocument:
    return rr.TextDocument(text, media_type="text/markdown")


def log_episode_videos(recording: rr.RecordingStream, episode: Path, episode_index: int) -> None:
    label = rel_episode(episode)
    entity_episode = f"episodes/{episode_index:04d}_{safe_entity_part(label)}"
    command = set_instruction_command(episode)
    recording.log(
        f"{entity_episode}/info",
        log_text(
            textwrap.dedent(
                f"""\
                # Episode

                **episode:** `{label}`

                **instruction:** `{instruction(episode)}`

                **command:** `{command}`

                **directory:** `{episode}`

                ## Copy/edit

                ```bash
                {command}
                ```
                """
            )
        ),
        static=True,
    )

    for mp4 in sorted(episode.glob("*-images-rgb.mp4")):
        camera = mp4.name.removesuffix("-images-rgb.mp4")
        entity = f"{entity_episode}/cameras/{camera}"
        video_asset = rr.AssetVideo(path=mp4)
        recording.log(entity, video_asset, static=True)

        all_frame_timestamps_ns = np.asarray(video_asset.read_frame_timestamps_nanos(), dtype=np.int64)
        if len(all_frame_timestamps_ns) == 0:
            continue

        local_seconds = video_timeline_seconds(mp4, episode, camera, all_frame_timestamps_ns, first_timestamp(episode))
        if stride > 1:
            local_seconds = local_seconds[::stride]
            frame_timestamps_ns = all_frame_timestamps_ns[::stride]
        else:
            frame_timestamps_ns = all_frame_timestamps_ns
        n = min(len(local_seconds), len(frame_timestamps_ns))
        local_seconds = local_seconds[:n]
        frame_timestamps_ns = frame_timestamps_ns[:n]
        recording.send_columns(
            entity,
            indexes=[rr.TimeColumn("episode_time", duration=local_seconds)],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )


def episode_entity(episode: Path, episode_index: int) -> str:
    label = rel_episode(episode)
    return f"episodes/{episode_index:04d}_{safe_entity_part(label)}"


def make_blueprint(group_name: str, selected: list[Path]) -> rrb.Blueprint:
    episode_tabs = []
    for i, episode in enumerate(selected[:24]):
        root = episode_entity(episode, i)
        camera_views = [
            rrb.Spatial2DView(
                origin=f"{root}/cameras/{mp4.name.removesuffix('-images-rgb.mp4')}",
                contents="$origin",
                name=mp4.name.removesuffix("-images-rgb.mp4"),
            )
            for mp4 in sorted(episode.glob("*-images-rgb.mp4"))
        ]
        episode_tabs.append(
            rrb.Vertical(
                rrb.TextDocumentView(origin=f"{root}/info", contents="$origin", name="episode"),
                rrb.Grid(*camera_views, grid_columns=2, name="cameras"),
                name=rel_episode(episode),
                row_shares=[1, 5],
            )
        )

    views = [
        rrb.TextDocumentView(origin="summary", contents="$origin", name="summary"),
        rrb.Tabs(*episode_tabs, name="episodes"),
    ]
    return rrb.Blueprint(
        rrb.Horizontal(*views, name=f"instruction/{group_name}", column_shares=[1, 4]),
        rrb.BlueprintPanel(expanded=True),
        rrb.SelectionPanel(expanded=True),
        rrb.TimePanel(expanded=True, timeline="episode_time"),
        auto_layout=False,
        auto_views=False,
        collapse_panels=False,
    )


def selected_episodes() -> tuple[str, list[Path]]:
    if episode_args:
        selected = [resolve_episode(arg) for arg in episode_args]
        return "selected episodes", selected

    selected_instruction = instruction_filter or choose_instruction()
    selected = [episode.resolve() for episode in episodes() if instruction(episode) == selected_instruction]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise SystemExit(f"No episodes found for instruction: {selected_instruction}")
    return selected_instruction, selected


def set_episode_instruction() -> None:
    episode = resolve_episode(set_instruction_episode)
    meta = metadata(episode)
    old_instruction = instruction(episode)
    meta["instruction"] = set_instruction_text
    write_metadata(episode, meta)
    print(f"{rel_episode(episode)}")
    print(f"  old: {old_instruction}")
    print(f"  new: {set_instruction_text}")


def replace_instruction() -> None:
    matches = [episode.resolve() for episode in episodes() if instruction(episode) == replace_instruction_old]
    if not matches:
        raise SystemExit(f"No episodes found for instruction: {replace_instruction_old}")

    action = "Would update" if dry_run else "Updated"
    print(f"{action} {len(matches)} episode(s):")
    for episode in matches:
        print(f"  {rel_episode(episode)}")
        if dry_run:
            continue
        meta = metadata(episode)
        meta["instruction"] = replace_instruction_new
        write_metadata(episode, meta)

    print(f"old: {replace_instruction_old}")
    print(f"new: {replace_instruction_new}")
    if dry_run:
        print("dry-run: no files changed")


def write_html_index() -> None:
    grouped: dict[str, list[Path]] = {}
    for episode in episodes():
        group = instruction(episode)
        if instruction_filter and group != instruction_filter:
            continue
        grouped.setdefault(group, []).append(episode.resolve())

    if not grouped:
        raise SystemExit("No matching episodes found.")

    output_path = Path("/tmp/rl-teleop-recordings-index.html")
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>rl-teleop recordings</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f5;color:#1d1d1b}",
        "details{margin:14px 0;padding:10px 14px;background:white;border:1px solid #ddd;border-radius:8px}",
        "summary{cursor:pointer;font-weight:650}",
        ".episode{margin:16px 0;padding-top:12px;border-top:1px solid #eee}",
        ".videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}",
        "video{width:100%;background:#111;border-radius:6px}",
        "code{font-size:12px}",
        "</style>",
        "<h1>rl-teleop recordings</h1>",
    ]

    for group, group_episodes in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        parts.append(f"<details open><summary>{html_lib.escape(group)} ({len(group_episodes)})</summary>")
        for episode in group_episodes:
            label = rel_episode(episode)
            parts.append("<section class='episode'>")
            parts.append(f"<h3>{html_lib.escape(label)}</h3>")
            parts.append(f"<code>{html_lib.escape(str(episode))}</code>")
            parts.append("<div class='videos'>")
            for mp4 in sorted(episode.glob("*-images-rgb.mp4")):
                camera = mp4.name.removesuffix("-images-rgb.mp4")
                parts.append("<figure>")
                parts.append(f"<figcaption>{html_lib.escape(camera)}</figcaption>")
                parts.append(f"<video controls preload='none' src='{mp4.resolve().as_uri()}'></video>")
                parts.append("</figure>")
            parts.append("</div></section>")
        parts.append("</details>")

    output_path.write_text("\n".join(parts) + "\n")
    print(output_path)


if list_only:
    print_instruction_counts()
    raise SystemExit(0)

if set_instruction_episode:
    set_episode_instruction()
    raise SystemExit(0)

if replace_instruction_old:
    replace_instruction()
    raise SystemExit(0)

if html_index:
    write_html_index()
    raise SystemExit(0)

group_name, selected = selected_episodes()
print(f"Opening {len(selected)} episode(s) for: {group_name}", flush=True)

rec = rr.RecordingStream("rl_teleop_videos", recording_id=uuid.uuid4())
blueprint = make_blueprint(group_name, selected)
rec.spawn(memory_limit="50%", default_blueprint=blueprint)
rec.send_blueprint(blueprint, make_active=True, make_default=True)
rec.send_recording_name(f"instruction/{group_name}")
rec.log(
    "summary",
    log_text(
        "\n".join(
            [
                "# Instruction Group",
                "",
                f"**instruction:** `{group_name}`",
                f"**episodes:** `{len(selected)}`",
                "",
                "## Change Commands",
                "",
                "Copy a command, edit the final quoted text, then rerun this viewer.",
                "",
                "```bash",
                *[set_instruction_command(episode) for episode in selected],
                "```",
                "",
                "## Episodes",
                "",
                *[rel_episode(episode) for episode in selected],
            ]
        )
    ),
    static=True,
)

for i, episode in enumerate(selected):
    print(f"{i + 1}/{len(selected)} {rel_episode(episode)}", flush=True)
    log_episode_videos(rec, episode, i)

rec.flush()
print("Loaded videos into Rerun.", flush=True)
PY

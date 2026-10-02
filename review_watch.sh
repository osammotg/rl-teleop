#!/usr/bin/env bash
# Live data-quality dashboard for kitting recording.
# On every saved episode it: (1) re-labels offline with the correct gripper refs
# (fixes the live-labeler mislabel), (2) writes a gripper-health qa.json, and
# (3) regenerates the review HTML, served over http so you watch it live.
#
# Usage:  ./review_watch.sh [recordings_dir] [http_port]
#   view on your laptop:  ssh -L 8792:localhost:8792 <rig> then open http://localhost:8792
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REC="${1:-recordings}"
PORT="${2:-8792}"
# Stable serve dir (NOT mktemp): the page survives Ctrl-C so you can keep reviewing,
# and the episode videos are reachable via a symlink into the recordings tree.
SERVE_DIR="$SCRIPT_DIR/.review"
HTML="$SERVE_DIR/index.html"
URL="http://localhost:${PORT}"
mkdir -p "$SERVE_DIR"
ln -sfn "$(cd "$REC" && pwd)" "$SERVE_DIR/recordings"   # <video src="recordings/...">

[ -f "$HTML" ] || echo "<!doctype html><meta http-equiv=refresh content=5><body style=background:#0d1117;color:#ccc;font-family:system-ui><p style=margin:40px>waiting for first episode…</p>" > "$HTML"
if ! curl -s -o /dev/null --max-time 1 "$URL/"; then
  # review_server.py, not http.server: the stdlib one ignores Range, so video
  # seeking is dead and the big camera_scan mp4s stream from byte 0 every time.
  python3 review_server.py "$SERVE_DIR" "$PORT" >/tmp/review_http.log 2>&1 &
  HTTP_PID=$!
fi
cleanup(){ kill "${HTTP_PID:-}" 2>/dev/null || true; }   # keep SERVE_DIR: page stays readable
trap cleanup EXIT

echo "live review dashboard -> ${URL}  (auto-refreshes; regenerates on each saved episode)"
echo "watching: $REC   (Ctrl-C to stop)"

# Auto-open on the rig display, same pattern as record_kitting.sh. Non-fatal;
# set REVIEW_NO_OPEN=1 to skip (e.g. when you only want the tunnel).
if [ "${REVIEW_NO_OPEN:-0}" != "1" ]; then
  export DISPLAY="${DISPLAY:-:2}"
  ( xdg-open "$URL" >/dev/null 2>&1 || firefox --new-window "$URL" >/dev/null 2>&1 ) &
fi

SIG_PREV=""
while true; do
  SIG="$(find "$REC" \( -name 'yam_left.mcap' -o -name 'annotations.json' -o -name 'qa.json' \) -printf '%p:%T@\n' 2>/dev/null | sort | md5sum)"
  if [ "$SIG" != "$SIG_PREV" ]; then
    # (1) re-label any episode whose qa.json is missing or older than its mcap.
    while IFS= read -r mcap; do
      [ -z "$mcap" ] && continue
      d="$(dirname "$mcap")"; qa="$d/qa.json"
      if [ ! -f "$qa" ] || [ "$mcap" -nt "$qa" ]; then
        echo "  [$(date +%H:%M:%S)] $(uv run python qa_label.py "$d" 2>/dev/null)"
      fi
    done < <(find "$REC" -name 'yam_left.mcap' | sort)
    # (2) regenerate the dashboard + a concise console summary.
    python3 review_corpus.py "$REC" --html "$HTML" --media-prefix recordings >/tmp/review_last.txt 2>/dev/null
    grep -E "usable \(|DEAD GRIPPER|CORRUPT mcap|MISLABEL|in target region|NOT CLEAR|CLEAR —|^    [0-9]\." /tmp/review_last.txt | sed 's/^/    /'
    echo "  [$(date +%H:%M:%S)] dashboard refreshed"
    SIG_PREV="$SIG"
  fi
  sleep 3
done

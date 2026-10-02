#!/usr/bin/env bash
# One command to start a kitting recording session:
#   1. live label backend (cockpit /state + camera JPEG bridge)
#   2. teleop session (records yam_left.mcap etc., TUI in this terminal)
#   3. auto-opens viser + the cockpit in Firefox on the rig display
#
# Prereq (needs sudo, do once per boot): bring up the left follower CAN —
#   sudo ip link set can_follow_l down && \
#   sudo ip link set can_follow_l type can bitrate 1000000 && \
#   sudo ip link set can_follow_l up
#
# Usage:  ./record_kitting.sh   [config.yaml]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/yam/yam_left_kitting_teleop.yaml}"
COCKPIT_DIR="/home/tommaso/Desktop/kitting/cockpit"
COCKPIT_PORT=8799
# Serve over http (NOT file://): the cockpit fetches /cam + /state from the cam
# server cross-origin; a file:// origin ("null") gets blocked, an http origin does not.
# Which cockpit to open. v3 is the redesigned drawing-sheet build and the only
# one with the episode archive; v1 is the older build, kept as an instant
# fallback:  COCKPIT_FILE=Buehler-Kitting-Cockpit.html ./record_kitting.sh
COCKPIT_FILE="${COCKPIT_FILE:-Buehler-Kitting-Cockpit-v3.html}"
ARCHIVE_PORT="${ARCHIVE_PORT:-8793}"     # episode archive (the cockpit's "Aufnahmen")
COCKPIT="http://localhost:${COCKPIT_PORT}/${COCKPIT_FILE}"
VISER="http://localhost:8080"
CONTROL_PORT="${CONTROL_PORT:-8792}"     # rr-session's HTTP control surface

# Which X display to open the cockpit on.
#   1. an inherited DISPLAY wins — if you launched this from a terminal on the
#      rig, that terminal already knows the right one.
#   2. otherwise take the lowest existing X socket.
# This used to be hardcoded to ":2", which does not exist on this machine (only
# :0 and :1 do), so `firefox` failed silently on every run — the script treats
# browser failures as non-fatal, so it looked like "the cockpit never opens".
if [ -z "${DISPLAY:-}" ]; then
  for _d in /tmp/.X11-unix/X*; do
    [ -e "$_d" ] || continue
    DISPLAY=":${_d##*/X}"
    break
  done
fi
export DISPLAY="${DISPLAY:-:0}"

# Preflight: refuse to start a SECOND session on top of a running one.
# The MessageBus binds 5555/5556 in a subprocess; when they are already taken
# the bind fails inside that child and the parent only ever saw "MessageBus
# failed to start within timeout", which reads like a hardware fault. It isn't
# — it means the previous session is still up. Catch it here, before anything
# else is spawned, and say so plainly.
BUS_PIDS="$(ss -tlnpH 2>/dev/null | grep -E ':(5555|5556)\s' \
            | grep -oP 'pid=\K[0-9]+' | sort -u | tr '\n' ' ')"
if [ -n "${BUS_PIDS// /}" ]; then
  cat >&2 <<EOF

  ✗ A kitting session is ALREADY RUNNING (message bus ports 5555/5556 in use).

    pid(s): ${BUS_PIDS}

    Quit it with [q] in its TUI, or:  kill ${BUS_PIDS}
    Then re-run this script.

EOF
  exit 1
fi

# 0) serve the cockpit over http so its auto-connect to the cam server works.
if ! curl -s -o /dev/null --max-time 1 "http://localhost:${COCKPIT_PORT}/"; then
  ( cd "$COCKPIT_DIR" && python3 -m http.server "$COCKPIT_PORT" --bind 127.0.0.1 ) \
      > /tmp/cockpit_http.log 2>&1 &
  HTTP_PID=$!
  echo "cockpit http server → ${COCKPIT} (log: /tmp/cockpit_http.log)"
fi

# 1) live label backend — /state + /cam JPEG from the running session's camera bus.
#    default=camera_top/rgb so every cockpit panel shows something even before the
#    exact per-camera id mapping is set.
# EPISODE_MODE — only the STARTING position of the cockpit's "Takt" switch,
# which you can flip at any point during the session (or: curl -X POST
# "localhost:8791/episodemode?m=full").
#   full  (default) one episode per BOX — you end each take yourself (→ / [1]).
#   grasp           one episode per GRASP-AND-PLACE cycle, saved hands-free the
#                   moment a placement passes the transport gate. Use this to
#                   collect grasp data for ACT: many short consistent takes beat
#                   a few long ones.  EPISODE_MODE=grasp ./record_kitting.sh
EPISODE_MODE="${EPISODE_MODE:-full}"

# A leftover live_server from a previous run holds 8791; starting a second one
# only writes "Address already in use" into the log while this script cheerfully
# prints the URL of the process that just died. Kill the stale one instead.
STALE_LIVE="$(ss -tlnpH 2>/dev/null | grep -E ':8791\s' \
              | grep -oP 'pid=\K[0-9]+' | sort -u | tr '\n' ' ')"
if [ -n "${STALE_LIVE// /}" ]; then
  echo "stale live label backend on :8791 (pid ${STALE_LIVE}) — replacing it"
  kill $STALE_LIVE 2>/dev/null || true
  for _ in $(seq 1 20); do
    ss -tlnH 2>/dev/null | grep -qE ':8791\s' || break
    sleep 0.25
  done
fi

# -u is NOT optional here: stdout is a pipe, so Python block-buffers it and the
# log stays EMPTY (0 bytes) for the whole session — including the startup banner
# and any detector error. Debugging "the scan overlay shows nothing" against a
# silent log costs far more than the unbuffered writes do.
uv run python -u -m robots_realtime.labeling.live_server --live --arm left \
    --host 0.0.0.0 --port 8791 --detect --detect-period 2.0 \
    --save-root recordings --auto-label \
    --episode-mode "$EPISODE_MODE" --control-url "http://localhost:${CONTROL_PORT}" \
    --bus-cams "default=camera_top/rgb,egocentric=camera_top/rgb,scan=camera_scan/rgb,wristL=camera_left/rgb,wristR=camera_left/rgb" \
    > /tmp/live_label.log 2>&1 &
LIVE_PID=$!
echo "live label backend → http://localhost:8791 (log: /tmp/live_label.log)"

# 1b) episode archive — what the cockpit's "▤ Aufnahmen" panel reads. Holds no
#     session state, so it can be restarted mid-take without touching the
#     recorder; skipped silently if something is already serving that port.
if ! curl -s -o /dev/null --max-time 1 "http://localhost:${ARCHIVE_PORT}/health"; then
  uv run python -m robots_realtime.labeling.episode_server \
      --save-root recordings --arm left \
      --host 0.0.0.0 --port "$ARCHIVE_PORT" --prewarm \
      > /tmp/episode_archive.log 2>&1 &
  ARCHIVE_PID=$!
fi
echo "episode archive     → http://localhost:${ARCHIVE_PORT} (log: /tmp/episode_archive.log)"

# Show the clickable cockpit link in the rr-session TUI (next to viser) and here.
COCKPIT_TELEOP="${COCKPIT}#teleop"
export RR_DASHBOARD_URL="$COCKPIT_TELEOP"
echo "cockpit → $COCKPIT_TELEOP"

# 2) open the cockpit on the RIGHT HALF of the rig screen, so the terminal you
#    launched from keeps the left half. Chrome is used in preference to Firefox
#    purely because it honours --window-position/--window-size; Firefox ignores
#    both, which is why the window used to land wherever the WM felt like.
#    No wmctrl/xdotool on this box, so geometry flags are the only lever.
#    Failures stay non-fatal — the URLs printed above are clickable regardless.
(
  for _ in $(seq 1 60); do
    curl -s -o /dev/null --max-time 1 "$VISER" && break || sleep 0.5
  done

  SW=1920; SH=1080
  if command -v xrandr >/dev/null 2>&1; then
    _geo="$(xrandr 2>/dev/null | awk '/\*/{print $1; exit}')"
    case "$_geo" in
      [0-9]*x[0-9]*) SW="${_geo%x*}"; SH="${_geo#*x}" ;;
    esac
  fi
  HALF=$(( SW / 2 ))

  if command -v google-chrome >/dev/null 2>&1; then
    google-chrome --new-window \
      --window-position="${HALF},0" --window-size="${HALF},${SH}" \
      "$COCKPIT_TELEOP" >/dev/null 2>&1 &
  elif command -v firefox >/dev/null 2>&1; then
    firefox --new-window "$COCKPIT_TELEOP" >/dev/null 2>&1 &
  fi
) &

cleanup() { kill "$LIVE_PID" 2>/dev/null || true; kill "${HTTP_PID:-}" 2>/dev/null || true;
            kill "${ARCHIVE_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

cat <<EOF

  ┌─ kitting session ─────────────────────────────────────────────────
  │  display   ${DISPLAY}   (auto-detected; export DISPLAY to override)
  │  cockpit   ${COCKPIT_TELEOP}
  │  control   http://localhost:${CONTROL_PORT}   ← the cockpit drives this
  │  labels    http://localhost:8791
  │  archive   http://localhost:${ARCHIVE_PORT}   ← "▤ Aufnahmen" in the cockpit
  │
  │  The cockpit's REC button and the keys below now drive the SAME
  │  recorder. Press [r] here and the cockpit updates; click REC there
  │  and this session starts recording. No refresh needed either way.
  │
  │  [r] start · [1] save · [d] discard · [space] pause · [q] quit
  │  [→] next take (start, or save-and-stop) · [←] redo this take
  └───────────────────────────────────────────────────────────────────

EOF

# 3) teleop session (TUI here; the cockpit is a peer on --control-port)
RS2_USE_RSUSB_BACKEND=true exec uv run rr-session "$CONFIG" --control-port "$CONTROL_PORT"

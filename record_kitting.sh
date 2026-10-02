#!/usr/bin/env bash
# One command to start a kitting recording session:
#   1. live label backend (cockpit /state + camera JPEG bridge)
#   2. teleop session (records yam_left.mcap etc., TUI in this terminal)
#   3. auto-opens viser + the cockpit in Firefox on the rig display
#
# Prereq (needs sudo, do once per boot): bring up the follower CAN for the arm
# you are recording — can_follow_l for left, can_follow_r for right —
#   sudo ip link set can_follow_<l|r> down && \
#   sudo ip link set can_follow_<l|r> type can bitrate 1000000 && \
#   sudo ip link set can_follow_<l|r> up
# Check with `ip -br link show`; the interface must read UP, not DOWN.
#
# Usage:  ./record_kitting.sh   [config.yaml]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/yam/yam_left_kitting_teleop.yaml}"

# Which arm this config drives. Until 2026-08-10 this script passed `--arm left`
# to BOTH label backends unconditionally, so handing it the right-arm config
# recorded yam_right.mcap while the labeller subscribed to yam_left/joint_state
# and the archive listed left episodes: a silently mislabelled dataset rather
# than an error. Derived from the config's RobotNode name, overridable.
#   ARM=right ./record_kitting.sh configs/yam/yam_right_kitting_teleop.yaml
if [ -z "${ARM:-}" ]; then
  if grep -qE '^\s*name:\s*yam_right\s*$' "$CONFIG"; then ARM=right; else ARM=left; fi
fi
case "$ARM" in
  left|right) ;;
  *) echo "✗ ARM must be 'left' or 'right', got '${ARM}'" >&2; exit 1 ;;
esac

COCKPIT_DIR="/home/tommaso/Desktop/kitting-v2/cockpit"
COCKPIT_PORT=8799
# Serve over http (NOT file://): the cockpit fetches /cam + /state from the cam
# server cross-origin; a file:// origin ("null") gets blocked, an http origin does not.
# Which cockpit to open. v3 is the redesigned drawing-sheet build and the only
# one with the episode archive; v1 is the older build, kept as an instant
# fallback:  COCKPIT_FILE=Buehler-Kitting-Cockpit.html ./record_kitting.sh
COCKPIT_FILE="${COCKPIT_FILE:-Buehler-Kitting-Cockpit-v3.html}"
ARCHIVE_PORT="${ARCHIVE_PORT:-8793}"     # episode archive (the cockpit's "Aufnahmen")
COCKPIT="http://localhost:${COCKPIT_PORT}/${COCKPIT_FILE}"
# Viser port follows the config: the right session uses 8081 because 8080 is the
# left session's and they collide. Waiting on the wrong one costs a silent 30 s
# stall before the cockpit opens.
if [ "${ARM}" = right ]; then VISER="http://localhost:8081"; else VISER="http://localhost:8080"; fi
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

# Preflight: the follower CAN interface must be UP, or the RobotNode fails to
# open the motor chain and the session dies with a driver error rather than
# saying which interface is down. It is a per-boot sudo step, deliberately not
# scripted here: bringing up a channel energises a brakeless arm.
CAN_IF="can_follow_${ARM:0:1}"
if ! ip -br link show "$CAN_IF" 2>/dev/null | grep -qw UP; then
  cat >&2 <<EOF

  ✗ ${CAN_IF} is not UP — the ${ARM} follower cannot be opened.

    Support the arm, then:
      sudo ip link set ${CAN_IF} down
      sudo ip link set ${CAN_IF} type can bitrate 1000000
      sudo ip link set ${CAN_IF} up

    Verify with:  ip -br link show ${CAN_IF}

EOF
  exit 1
fi

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

# Preflight: every camera in the config must actually be present.
#
# A camera node that cannot open its device dies inside setup(). That death used
# to be invisible — the traceback went to a per-node log under /tmp/rr_logs_*/,
# the session carried on without it, and the TUI stayed green because
# NodeStatus.alive was a constant. One real instance ran eleven minutes with a
# defunct camera_top while three cockpit panels pointed at it.
#
# The supervisor now reports that loudly at runtime, but the cheapest place to
# catch a missing camera is here: before anything is spawned, before the arm is
# energised, while there is still a terminal to print to. The check OPENS
# NOTHING, so it cannot steal a RealSense from anything.
#
# It is a hard gate. Override with FORCE_CAMERAS=1 if you genuinely mean to
# record without one — the episode will be marked degraded either way.
if ! ./.venv/bin/python3 tools/preflight_cameras.py "$CONFIG"; then
  if [ "${FORCE_CAMERAS:-0}" != "1" ]; then
    cat >&2 <<'EOF'
    Refusing to start. Re-run with FORCE_CAMERAS=1 to record anyway — episodes
    recorded without a camera are stamped "degraded": true in session_meta.json,
    so they can be filtered out later, but they are not clean takes.

EOF
    exit 1
  fi
  echo "  FORCE_CAMERAS=1 — starting anyway; episodes will be marked degraded" >&2
fi

# Post-start stream audit. THE PREFLIGHT ABOVE ONLY PROVES A DEVICE EXISTS; only
# frames whose CONTENT differs, counted off the bus, prove a camera is actually
# delivering. That check needs the session up, and this script exec()s into the
# session at the end — so it runs in the background and writes its verdict where
# the operator can find it. The cockpit shows the same truth live, per panel.
(
  sleep 30
  ./.venv/bin/python3 tools/check_streams.py --secs 8 > /tmp/check_streams.log 2>&1
  if [ $? -ne 0 ]; then
    {
      echo
      echo "  ######################################################################"
      echo "  #  STREAM CHECK FAILED $(date '+%H:%M:%S') — one or more cameras are NOT"
      echo "  #  delivering distinct frames on the bus. DO NOT RECORD until this is"
      echo "  #  resolved; check the cockpit panels and /tmp/check_streams.log."
      echo "  ######################################################################"
      echo
    } | tee -a /tmp/check_streams.log > /dev/tty 2>/dev/null || true
  fi
) &

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
# The detector, the auto-labeller and the grasp episode-mode gate are all
# calibrated in the LEFT arm's base frame (results/calibration.json, keepout.json,
# tray_box.json) and the grasp witness measures AUC 0.4525 — below chance — so on
# the right arm they would produce a confidently wrong overlay and confidently
# wrong labels. Off for right; left keeps exactly the flags it always had.
# There is also no camera_right node yet (see BLOCKER 2 in the right config), so
# the right session ships NO wrist panel rather than a black one pointed at a
# topic nobody publishes — a black panel reads as "camera present, scene dark".
if [ "$ARM" = right ]; then
  # No scan= entry: the D435i is unplugged (see the right config). A bus-cam
  # alias pointing at a topic no node publishes serves a black JPEG, which the
  # cockpit cannot tell from a dark scene.
  # wristR IS mapped now — camera_right was enabled 2026-08-10 at usb-0:1.1.
  LABEL_FLAGS=()
  # ALL FOUR panels are mapped as of 2026-08-11. The cockpit asks for exactly
  # these ids — top, scan, wristL, wristR, egocentric — and every one now resolves
  # to a DISTINCT camera: the D435i was replugged and camera_left was added to the
  # right config. Verified on the bus at 15/30/30/30 Hz, all frames distinct.
  # If a camera is ever removed, delete its alias here in the SAME edit that
  # disables the node. An alias with no publisher 503s, which is honest; an alias
  # pointed at the WRONG camera shows a plausible lie, which is how this cockpit
  # once displayed the top view in three panels at once.
  BUS_CAMS="default=camera_top/rgb,egocentric=camera_top/rgb,top=camera_top/rgb,scan=camera_scan/rgb,wristL=camera_left/rgb,wristR=camera_right/rgb"
else
  LABEL_FLAGS=(--detect --detect-period 2.0 --auto-label)
  BUS_CAMS="default=camera_top/rgb,egocentric=camera_top/rgb,scan=camera_scan/rgb,wristL=camera_left/rgb,wristR=camera_left/rgb"
fi

uv run python -u -m robots_realtime.labeling.live_server --live --arm "$ARM" \
    --host 0.0.0.0 --port 8791 "${LABEL_FLAGS[@]}" \
    --save-root recordings \
    --episode-mode "$EPISODE_MODE" --control-url "http://localhost:${CONTROL_PORT}" \
    --bus-cams "$BUS_CAMS" \
    > /tmp/live_label.log 2>&1 &
LIVE_PID=$!
echo "live label backend → http://localhost:8791 (log: /tmp/live_label.log)"

# 1b) episode archive — what the cockpit's "▤ Aufnahmen" panel reads. Holds no
#     session state, so it can be restarted mid-take without touching the
#     recorder; skipped silently if something is already serving that port.
if ! curl -s -o /dev/null --max-time 1 "http://localhost:${ARCHIVE_PORT}/health"; then
  uv run python -m robots_realtime.labeling.episode_server \
      --save-root recordings --arm "$ARM" \
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
  │  arm       ${ARM}   (${CAN_IF} up · records yam_${ARM}.mcap)
  │  config    ${CONFIG}
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

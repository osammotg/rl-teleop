#!/usr/bin/env bash
# Dev launcher for the kitting cockpit WITHOUT the robot/rr-session.
# Publishes the REAL cameras to the ZMQ bus, runs the live label backend + scan-cam
# packet detector, and serves the cockpit over http. Idempotent: re-run to restart.
#
#   Start:   bash dev_cockpit.sh
#   Watch:   tail -f /tmp/dev_cam.log /tmp/dev_live.log /tmp/dev_http.log
#   Stop:    bash dev_cockpit.sh stop
#
# NOTE: this holds the D455 + D435i + wrist cameras directly. Stop it before running
# the full ./record_kitting.sh (rr-session opens the same cameras).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RS2_USE_RSUSB_BACKEND=true
COCKPIT_DIR="/home/tommaso/Desktop/kitting/cockpit"
URL="http://localhost:8799/Buehler-Kitting-Cockpit.html#teleop"

echo "stopping any previous instances…"
pkill -f 'scripts/dev_cam_publisher.py' 2>/dev/null || true
pkill -f 'labeling.live_server'         2>/dev/null || true
pkill -f 'http.server 8799'             2>/dev/null || true
sleep 2
[ "${1:-}" = "stop" ] && { echo "stopped."; exit 0; }

# 1) real cameras -> ZMQ bus (starts the broker + D455 camera_top + D435i camera_scan + wrist camera_left)
setsid uv run python scripts/dev_cam_publisher.py > /tmp/dev_cam.log 2>&1 &
echo "  [1/3] cameras -> bus        (log: /tmp/dev_cam.log)"
for _ in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':5555' && break; sleep 0.5; done

# 2) live label backend + packet detector (pick-this-next box)
setsid uv run python -m robots_realtime.labeling.live_server --live --arm left \
    --host 0.0.0.0 --port 8791 --detect --detect-period 2.0 \
    --bus-cams "default=camera_top/rgb,egocentric=camera_top/rgb,scan=camera_scan/rgb,wristL=camera_left/rgb,wristR=camera_left/rgb" \
    > /tmp/dev_live.log 2>&1 &
echo "  [2/3] live backend :8791    (log: /tmp/dev_live.log)"

# 3) cockpit over http
setsid bash -c "cd '$COCKPIT_DIR' && exec python3 -m http.server 8799 --bind 127.0.0.1" > /tmp/dev_http.log 2>&1 &
echo "  [3/3] cockpit http :8799    (log: /tmp/dev_http.log)"

sleep 4
echo
echo "  ports: $(ss -ltn 2>/dev/null | grep -oE ':(5555|8791|8799)' | sort -u | tr '\n' ' ')"
echo "  scan cam frames: $(grep -c camera_scan /tmp/dev_cam.log 2>/dev/null) log lines"
echo
echo "  COCKPIT →  $URL"
echo "  follow  →  tail -f /tmp/dev_cam.log /tmp/dev_live.log /tmp/dev_http.log"
echo "  stop    →  bash dev_cockpit.sh stop"

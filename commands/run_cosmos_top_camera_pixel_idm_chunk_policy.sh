#!/usr/bin/env bash
set -euo pipefail

PROMPT=""
EXECUTE=0
COSMOS_URL="http://127.0.0.1:8021"
PIXEL_IDM_URL="http://127.0.0.1:8022"
BITRATE="${CAN_BITRATE:-1000000}"
RES_TIER="${COSMOS_PIXEL_IDM_RES_TIER:-480}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --res)
      RES_TIER="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --prompt \"task prompt\" [--res 224|480] [--execute]"
      echo "Cosmos URL is hardcoded to ${COSMOS_URL}; use commands/tunnel_cosmos_ec2.sh for EC2."
      echo "Pixel-IDM URL is hardcoded to ${PIXEL_IDM_URL}."
      echo "Default is MuJoCo sim playback; --execute starts the SocketCAN hardware config."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "--prompt is required" >&2
  exit 2
fi

case "$RES_TIER" in
  224)
    export COSMOS_PIXEL_IDM_COSMOS_WIDTH=384
    export COSMOS_PIXEL_IDM_COSMOS_HEIGHT=224
    ;;
  480)
    export COSMOS_PIXEL_IDM_COSMOS_WIDTH=832
    export COSMOS_PIXEL_IDM_COSMOS_HEIGHT=480
    ;;
  *)
    echo "--res must be 224 or 480, got: $RES_TIER" >&2
    exit 2
    ;;
esac

export COSMOS_PIXEL_IDM_PROMPT="$PROMPT"
export COSMOS_PIXEL_IDM_COSMOS_URL="$COSMOS_URL"
export COSMOS_PIXEL_IDM_PIXEL_URL="$PIXEL_IDM_URL"
if [[ "$EXECUTE" == "1" ]]; then
  export COSMOS_PIXEL_IDM_VALIDATE_ONLY=0
  CONFIG="configs/yam/yam_bimanual_cosmos_top_camera_pixel_idm_chunk_policy.yaml"
else
  export COSMOS_PIXEL_IDM_VALIDATE_ONLY=1
  CONFIG="configs/yam/yam_bimanual_cosmos_top_camera_pixel_idm_chunk_policy_sim.yaml"
fi

cd "$(dirname "$0")/.."

check_followers() {
  for iface in can_follow_l can_follow_r; do
    if ! ip link show dev "$iface" >/dev/null 2>&1; then
      echo "ERROR: $iface not found. Run ./custom/init-mapping.sh and replug CANables." >&2
      exit 1
    fi

    state="$(ip -brief link show dev "$iface" | awk '{print $2}')"
    if [[ "$state" != "UP" ]]; then
      echo "Bringing up $iface at ${BITRATE} bit/s..."
      sudo ip link set dev "$iface" down 2>/dev/null || true
      sudo ip link set dev "$iface" type can bitrate "$BITRATE"
      sudo ip link set dev "$iface" up
    fi
  done
}

.venv/bin/python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

for name, base_url in {
    "Cosmos": os.environ["COSMOS_PIXEL_IDM_COSMOS_URL"],
    "Pixel-IDM": os.environ["COSMOS_PIXEL_IDM_PIXEL_URL"],
}.items():
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            body = response.read().decode("utf-8").strip()
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"{name} server health check failed at {url}: {exc}", file=sys.stderr)
        print("Start the model servers first, then rerun this command.", file=sys.stderr)
        raise SystemExit(1) from exc
    if name == "Cosmos" and not body:
        continue
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        print(f"{name} server health check failed at {url}: {exc}", file=sys.stderr)
        print("Start the model servers first, then rerun this command.", file=sys.stderr)
        raise SystemExit(1) from exc
    if payload.get("ok") is not True:
        print(f"{name} server at {url} returned unhealthy payload: {payload}", file=sys.stderr)
        raise SystemExit(1)
PY

if [[ "$CONFIG" == *"_sim.yaml" ]]; then
  echo "Launching MuJoCo sim config: $CONFIG"
else
  check_followers
  echo "CAN interfaces:"
  ip -brief link show dev can_follow_l
  ip -brief link show dev can_follow_r
  echo "Launching HARDWARE SocketCAN config: $CONFIG"
fi

exec .venv/bin/rr-session "$CONFIG"

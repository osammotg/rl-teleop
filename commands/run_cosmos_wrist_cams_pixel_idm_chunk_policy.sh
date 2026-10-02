#!/usr/bin/env bash
set -euo pipefail

PROMPT=""
EXECUTE=0
COSMOS_URL="${COSMOS_PIXEL_IDM_COSMOS_URL:-http://127.0.0.1:8021}"
PIXEL_IDM_URL="${COSMOS_PIXEL_IDM_PIXEL_URL:-http://127.0.0.1:8022}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --cosmos-url)
      COSMOS_URL="${2:-}"
      shift 2
      ;;
    --pixel-idm-url)
      PIXEL_IDM_URL="${2:-}"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --prompt \"task prompt\" [--cosmos-url URL] [--pixel-idm-url URL] [--execute]"
      echo "Cosmos defaults to the local EC2 tunnel at http://127.0.0.1:8021."
      echo "Default is validate-only MuJoCo dry-run; --execute starts the SocketCAN hardware config."
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

export COSMOS_PIXEL_IDM_PROMPT="$PROMPT"
export COSMOS_PIXEL_IDM_COSMOS_URL="$COSMOS_URL"
export COSMOS_PIXEL_IDM_PIXEL_URL="$PIXEL_IDM_URL"
if [[ "$EXECUTE" == "1" ]]; then
  export COSMOS_PIXEL_IDM_VALIDATE_ONLY=0
  CONFIG="configs/yam/yam_bimanual_cosmos_wrist_cams_pixel_idm_chunk_policy.yaml"
else
  export COSMOS_PIXEL_IDM_VALIDATE_ONLY=1
  CONFIG="configs/yam/yam_bimanual_cosmos_wrist_cams_pixel_idm_chunk_policy_sim.yaml"
fi

cd "$(dirname "$0")/.."

.venv/bin/python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

checks = {
    "Cosmos": (os.environ["COSMOS_PIXEL_IDM_COSMOS_URL"], "/v1/models", False),
    "Pixel-IDM": (os.environ["COSMOS_PIXEL_IDM_PIXEL_URL"], "/health", True),
}

for name, (base_url, path, expect_ok) in checks.items():
    url = base_url.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"{name} server health check failed at {url}: {exc}", file=sys.stderr)
        print("Start the model servers first, then rerun this command.", file=sys.stderr)
        raise SystemExit(1) from exc
    if expect_ok and payload.get("ok") is not True:
        print(f"{name} server at {url} returned unhealthy payload: {payload}", file=sys.stderr)
        raise SystemExit(1)
PY

exec .venv/bin/rr-session "$CONFIG"

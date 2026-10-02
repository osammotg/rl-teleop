#!/usr/bin/env bash
# Print which /dev/video* is camera_left vs camera_right (requires enrolled references).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REFERENCES="${REFERENCES:-.local/references}"
MAPPING="${MAPPING:-.local/mapping.json}"
PRODUCT_FILTER="${PRODUCT_FILTER:-Innomaker}"

if [[ ! -f "$REFERENCES/reference_set.json" ]]; then
  echo "No enrolled references at $REFERENCES" >&2
  echo "" >&2
  echo "Run once:" >&2
  echo "  cd $ROOT" >&2
  echo "  uv run cv-camera-map enumerate --json" >&2
  echo "  uv run cv-camera-map capture-reference \\" >&2
  echo "    --roles camera_left,camera_right \\" >&2
  echo "    --role-device camera_left=/dev/video0 \\" >&2
  echo "    --role-device camera_right=/dev/video8 \\" >&2
  echo "    --output .local/references" >&2
  exit 1
fi

mkdir -p "$(dirname "$MAPPING")"

EXTRA_ARGS=()
if [[ -n "$PRODUCT_FILTER" ]]; then
  EXTRA_ARGS+=(--product-filter "$PRODUCT_FILTER")
fi

set +e
uv run cv-camera-map identify \
  --references "$REFERENCES" \
  --json \
  --output "$MAPPING" \
  "${EXTRA_ARGS[@]}" \
  >/dev/null
identify_exit=$?
set -e

exec uv run python - "$MAPPING" "$identify_exit" <<'PY'
import json
import sys
from pathlib import Path

mapping_path = Path(sys.argv[1])
identify_exit = int(sys.argv[2])

data = json.loads(mapping_path.read_text())
assignments = data.get("assignments") or {}
ambiguous = data.get("ambiguous", True)
reason = data.get("failure_reason")

by_path: dict[str, str] = {}
for role, entry in assignments.items():
    path = entry.get("device_path")
    if path:
        by_path[path] = role

products = {
    c.get("device_path"): c.get("product")
    for c in data.get("source_candidates") or []
}

print()
print("Wrist camera mapping")
print("====================")

if ambiguous or identify_exit != 0:
    print("STATUS: FAILED (do not trust /dev/videoN guesses)")
    if reason:
        print(f"REASON: {reason}")
    print()
    print("Tips:")
    print("  - Unplug extra capture devices or set PRODUCT_FILTER=Innomaker")
    print("  - Re-run capture-reference if the desk scene changed")
    print(f"  - Full JSON: {mapping_path}")
    sys.exit(1)

for role in ("camera_left", "camera_right"):
    entry = assignments.get(role, {})
    path = entry.get("device_path", "?")
    conf = entry.get("confidence")
    method = entry.get("method", "?")
    product = products.get(path) or ""
    vid = path.replace("/dev/video", "video") if path.startswith("/dev/video") else path
    extra = f"  ({product})" if product else ""
    conf_s = f"  confidence={conf:.2f}" if conf is not None else ""
    print(f"{vid:10} → {role:12}  [{method}]{conf_s}{extra}")

print()
print("By role:")
for role in ("camera_left", "camera_right"):
    path = assignments[role]["device_path"]
    vid = path.replace("/dev/", "")
    print(f"  {role:12} → {path}  ({vid})")

print()
print(f"Artifact: {mapping_path}")
PY

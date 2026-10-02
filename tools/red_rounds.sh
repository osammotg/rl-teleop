#!/usr/bin/env bash
# RED rounds — the stop condition from HANDOFF-CAMERA-HARDENING.md §5.5.
#
# One ROUND attacks the whole catalogue: every fault in §5.1, every invented
# fault, the process-level faults from §5.2, and the "power off" case from §5.3.
# The run stops when three CONSECUTIVE rounds find no violation.
#
# Everything here is safe to run at any time EXCEPT the optional hardware stage:
# the hermetic tier uses fakes, the soak uses synthetic cameras on its own
# ports (5595/5596), and the cold-start check kills only its own process group.
# Nothing
# opens a real camera, touches the arm, or binds a live port.
#
#   tools/red_rounds.sh              # 3 rounds, the stop condition
#   tools/red_rounds.sh 5            # 5 rounds
#   WITH_HARDWARE=1 tools/red_rounds.sh   # also soak the real cameras
#                                         # (refuses if a session is running)
#
# Exit 0 only if every round passed.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 2

ROUNDS="${1:-3}"
PY=./.venv/bin/python3
LOG_DIR="${LOG_DIR:-/tmp/red_rounds}"
mkdir -p "$LOG_DIR"

# Its own port pair, distinct from every other soak in this tree. A round runner
# that collides with a soak someone left running reports a port conflict as a
# product failure, which is a lie about the thing it exists to measure.
SOAK_PUB="${SOAK_PUB:-5595}"
SOAK_SUB="${SOAK_SUB:-5596}"

fail_total=0

for r in $(seq 1 "$ROUNDS"); do
  echo
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  RED ROUND $r/$ROUNDS   $(date '+%H:%M:%S')"
  echo "════════════════════════════════════════════════════════════════════════"
  round_fail=0

  echo
  echo "── stage 1: the hermetic catalogue (§5.1 + invented) ──────────────────"
  # A generous timeout, not none: a hung pytest is indistinguishable from a
  # working one for hours, and this runs unattended.
  if timeout 900 $PY -m pytest -q --no-header -p no:cacheprovider \
        tests/sensors/cameras/ \
        tests/runtime/test_session_health.py \
        tests/labeling/test_camera_bridge.py \
        tests/labeling/test_camera_bridge_staleness.py \
        > "$LOG_DIR/round${r}-tierA.log" 2>&1; then
    echo "   PASS  $(tail -1 "$LOG_DIR/round${r}-tierA.log")"
  else
    echo "   FAIL  see $LOG_DIR/round${r}-tierA.log"
    grep -E '^FAILED|^ERROR' "$LOG_DIR/round${r}-tierA.log" | head -20
    round_fail=1
  fi

  echo
  echo "── stage 2: process-level faults, synthetic cameras (§5.2) ────────────"
  if timeout 900 $PY tools/camera_soak.py \
        --config configs/yam/cameras_only_soak_fake.yaml \
        --duration 300 --fault-period 32 --faults --warmup 8 \
        --pub-port "$SOAK_PUB" --sub-port "$SOAK_SUB" \
        > "$LOG_DIR/round${r}-soak.log" 2>&1; then
    echo "   PASS  $(grep -E '^SOAK REPORT' -A1 "$LOG_DIR/round${r}-soak.log" | tail -1)"
    grep -E '^SOAK REPORT' "$LOG_DIR/round${r}-soak.log"
  else
    echo "   FAIL  see $LOG_DIR/round${r}-soak.log"
    grep -E 'VIOLATION' "$LOG_DIR/round${r}-soak.log" | head -20
    round_fail=1
  fi

  echo
  echo "── stage 3: the power-off case (§5.3) ─────────────────────────────────"
  if timeout 600 $PY tools/cold_start_check.py \
        --config configs/yam/cameras_only_soak_fake.yaml \
        --pub-port 5585 --sub-port 5586 --warmup 8 \
        > "$LOG_DIR/round${r}-cold.log" 2>&1; then
    echo "   PASS  $(tail -1 "$LOG_DIR/round${r}-cold.log")"
  else
    echo "   FAIL  see $LOG_DIR/round${r}-cold.log"
    tail -8 "$LOG_DIR/round${r}-cold.log"
    round_fail=1
  fi

  if [ "${WITH_HARDWARE:-0}" = "1" ]; then
    echo
    echo "── stage 4: the real cameras (Mode A only; the guard enforces it) ─────"
    if timeout 900 $PY tools/camera_soak.py \
          --config configs/yam/cameras_only_soak.yaml \
          --duration 300 --fault-period 45 --faults --warmup 15 \
          --pub-port 5575 --sub-port 5576 \
          > "$LOG_DIR/round${r}-hw.log" 2>&1; then
      grep -E '^SOAK REPORT|^VERDICT' "$LOG_DIR/round${r}-hw.log"
    else
      echo "   FAIL (or refused) — see $LOG_DIR/round${r}-hw.log"
      tail -6 "$LOG_DIR/round${r}-hw.log"
      round_fail=1
    fi
  fi

  if [ "$round_fail" -eq 0 ]; then
    echo
    echo "  ROUND $r: CLEAN"
  else
    echo
    echo "  ROUND $r: VIOLATIONS FOUND — the streak resets"
    fail_total=$((fail_total + 1))
  fi
done

echo
echo "════════════════════════════════════════════════════════════════════════"
if [ "$fail_total" -eq 0 ]; then
  echo "  $ROUNDS CONSECUTIVE CLEAN ROUNDS — §5.5 stop condition satisfied"
  exit 0
fi
echo "  $fail_total of $ROUNDS rounds found violations. Logs in $LOG_DIR"
exit 1

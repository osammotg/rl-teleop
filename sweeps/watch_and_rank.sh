#!/usr/bin/env bash
# Re-rank the sweep every time a run completes, so the table is current whenever
# somebody looks at it.
#
# Polls results.jsonl for a new terminal record rather than re-ranking on a
# timer: ranking loads every checkpoint onto the GPU, and doing that while eight
# training runs are queued would steal throughput from the thing being measured.
#
# Writes sweeps/ranking.json (overwritten, always current) and appends a stamped
# table to sweeps/ranking-history.txt so a regression between rounds is visible
# rather than silently replaced.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
SW=sweeps
export LEROBOT_PREDECODED_ROOT="$HOME/.cache/lerobot-predecoded/yam_grasp_v2_wrist"

last=0
idle=0
while :; do
  n=$( [ -f "$SW/results.jsonl" ] && wc -l < "$SW/results.jsonl" || echo 0 )

  if [ "$n" -gt "$last" ]; then
    last=$n; idle=0
    echo "[$(date '+%H:%M:%S')] $n runs terminal -- re-ranking" \
      | tee -a "$SW/ranking-history.txt"
    ./.venv/bin/python "$SW/rank_sweep.py" 2>&1 \
      | grep -vE "^Loading weights" \
      | tee -a "$SW/ranking-history.txt"
  else
    idle=$((idle + 1))
  fi

  # Stop once the driver is gone AND nothing new has appeared for ~10 minutes.
  # Both conditions matter: the driver can exit while a final rank is pending,
  # and a stalled driver should not look like a finished one.
  if ! pgrep -f "run_sweep.py --queue $SW/queue.json" >/dev/null 2>&1; then
    if [ "$idle" -ge 10 ]; then
      echo "[$(date '+%H:%M:%S')] driver gone, no new results for ~10 min -- final rank" \
        | tee -a "$SW/ranking-history.txt"
      ./.venv/bin/python "$SW/rank_sweep.py" 2>&1 \
        | grep -vE "^Loading weights" \
        | tee -a "$SW/ranking-history.txt"
      echo "[$(date '+%H:%M:%S')] watcher done." | tee -a "$SW/ranking-history.txt"
      exit 0
    fi
  fi
  sleep 60
done

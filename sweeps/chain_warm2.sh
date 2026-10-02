#!/usr/bin/env bash
# Wait for the round-1 warm driver to finish, then run round 2. Chained rather
# than launched concurrently: a third simultaneous training would slow the two
# already in flight more than it gains.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "run_sweep.py --queue sweeps/queue-warm20k.json" >/dev/null; do sleep 30; done
exec ./.venv/bin/python sweeps/run_sweep.py --queue sweeps/queue-warm2.json \
     --results sweeps/results-warm.jsonl

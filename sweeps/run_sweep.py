#!/usr/bin/env python3
"""Sequential ACT sweep driver with a disk guard.

Runs the configs in `queue.json` one at a time, prunes checkpoints as it goes,
and appends one line per run to `results.jsonl`. Resumable: a run whose id is
already in `results.jsonl` with a terminal status is skipped, so the driver can
be killed and relaunched without losing or repeating work.

WHY A DRIVER AND NOT A SHELL LOOP
---------------------------------
Two reasons, both learned from this repo's history.

1. **Disk.** The box is at 93% with ~62 GB free and each ACT run writes 3-6 GB of
   checkpoints. A naive `for cfg in ...; do train; done` fills the disk somewhere
   in the middle of the night, and what breaks is not the sweep -- it is the
   production recording tree that shares the volume. So free space is checked
   before every run and again between checkpoints, and the sweep stops rather
   than degrading.

2. **This codebase's defining defect is status signals that cannot report
   failure** (AUDIT.md: eleven of them). A shell loop reports the exit code of
   the last command. This driver records, per run: the exit code, whether a
   checkpoint was actually written, its size, the wall time, and the last lines
   of stderr. A run that "succeeded" having written no checkpoint is a FAILURE
   here and is recorded as one.

USAGE
-----
    cd ~/Desktop/kitting-v2/rl-teleop
    .venv/bin/python sweeps/run_sweep.py --queue sweeps/queue.json

    # inspect without running
    .venv/bin/python sweeps/run_sweep.py --queue sweeps/queue.json --dry-run

Intended to be launched inside tmux so it survives a disconnect:
    tmux new-session -d -s sweep '.../python sweeps/run_sweep.py ...'
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # rl-teleop/
SWEEPS = REPO / "sweeps"
TRAIN_ENTRY = REPO / "tools" / "train_act_dark_noise.py"
PY = REPO / ".venv" / "bin" / "python"

# Sweep output goes HERE, not into `rl-teleop/outputs/`.
#
# In this working copy `outputs/` is a SYMLINK to the live production tree at
# ~/Desktop/kitting, where a sort_server has had a checkpoint resident for days.
# Writing sweep runs through that symlink would put experimental checkpoints
# into the directory production reads from, on a volume that is already 93%
# full. So the sweep owns a real directory inside the working copy instead.
OUT_ROOT = SWEEPS / "outputs"

# The decode path. Without BOTH of these, training does not run slowly -- it dies
# at the first batch, because torchcodec cannot load on this rig (libavutil is
# missing for every FFmpeg version it supports). See train_act_dark_noise.py.
PREDECODED_ROOT = Path.home() / ".cache" / "lerobot-predecoded"
FAST_PATCH = Path.home() / "Desktop" / "lab" / "lerobot-fast" / "predecoded_patch.py"

# Disk thresholds, in GB of free space on the volume holding outputs/.
DISK_MIN_TO_START = 25.0   # refuse to begin a run below this
DISK_MIN_TO_CONTINUE = 12.0  # abort the sweep entirely below this


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def preflight() -> list[str]:
    """Return a list of blocking problems. Empty list means good to go."""
    problems = []
    if not PY.exists():
        problems.append(f"interpreter missing: {PY}")
    if not TRAIN_ENTRY.exists():
        problems.append(f"training entrypoint missing: {TRAIN_ENTRY}")
    if not FAST_PATCH.exists():
        problems.append(
            f"predecoded patch missing: {FAST_PATCH}\n"
            "  Without it torchcodec is reached and training dies at the first batch.")
    if not PREDECODED_ROOT.exists():
        problems.append(f"predecoded cache missing: {PREDECODED_ROOT}")
    f = free_gb(REPO)
    if f < DISK_MIN_TO_START:
        problems.append(f"only {f:.1f} GB free, need {DISK_MIN_TO_START} GB to start")
    return problems


def load_done(results_path: Path) -> set[str]:
    """Ids of runs already carried to a terminal state."""
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in {"ok", "failed", "skipped"}:
                done.add(rec["id"])
    return done


def checkpoints_of(out_dir: Path) -> list[Path]:
    """Real checkpoint directories, oldest first. Excludes the `last` symlink.

    LeRobot writes numbered directories plus a `last` SYMLINK pointing at the
    newest one. `Path.is_dir()` follows symlinks and returns True for it, so a
    naive listing yields ['000500', 'last'] -- and because 'last' sorts after
    any digit string, a keep-newest-N prune deletes the real checkpoint and
    leaves a dangling link behind.

    That is not hypothetical: it happened here on the first two runs. The
    directory was left holding nothing but `last -> 000500`, and the driver
    cheerfully recorded `n_checkpoints=2, status=ok` over an empty tree -- the
    exact failure mode this file exists to avoid. Hence: symlinks excluded, and
    a name that must be all digits.
    """
    ck = out_dir / "checkpoints"
    if not ck.is_dir():
        return []
    return sorted(
        (d for d in ck.iterdir()
         if d.is_dir() and not d.is_symlink() and d.name.isdigit()),
        key=lambda p: int(p.name),
    )


def prune_checkpoints(out_dir: Path, keep: int) -> tuple[int, float]:
    """Keep the newest `keep` checkpoints. Returns (n_removed, gb_freed).

    The last checkpoint is what gets deployed and the ones before it are what
    let you A/B a training curve on the robot -- which matters here more than
    usual, because validation loss is known not to predict rollout success on
    this rig, so the only real ranking is on-robot and you need candidates.
    """
    cks = checkpoints_of(out_dir)
    if len(cks) <= keep:
        return 0, 0.0
    doomed = cks[:-keep] if keep > 0 else cks
    freed = 0.0
    for d in doomed:
        try:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(d)
        except OSError as exc:
            print(f"    prune failed for {d.name}: {exc}", flush=True)
    return len(doomed), freed


def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def run_one(spec: dict, results_path: Path, log_dir: Path) -> dict:
    """Run a single training config to completion. Never raises."""
    run_id = spec["id"]
    out_dir = OUT_ROOT / run_id
    log_path = log_dir / f"{run_id}.log"
    keep = int(spec.get("keep_checkpoints", 2))

    # LEROBOT_PREDECODED_ROOT is PER-DATASET, not a shared root. predecoded_patch.py
    # resolves `$ROOT/videos/<key>/chunk-000/...`, while the cache on this box is laid
    # out as `~/.cache/lerobot-predecoded/<dataset>/videos/<key>/...`. Pointing at the
    # shared parent yields "Predecoded cache missing" and the run dies at the first
    # batch -- there is no fallback, because torchcodec cannot load here at all.
    dataset = next((a.split("=", 1)[1] for a in spec["args"]
                    if a.startswith("--dataset.repo_id=")), None)
    predecoded = PREDECODED_ROOT / dataset.split("/")[-1] if dataset else PREDECODED_ROOT

    env = os.environ.copy()
    env["LEROBOT_PREDECODED_ROOT"] = str(predecoded)
    env.update(spec.get("env", {}))
    rec_predecoded = str(predecoded)

    cmd = [str(PY), str(TRAIN_ENTRY), f"--output_dir={out_dir}", *spec["args"]]

    rec = {
        "id": run_id,
        "started_at": utcnow(),
        "cmd": " ".join(cmd),
        "predecoded_root": rec_predecoded,
        "note": spec.get("note", ""),
    }
    print(f"\n=== {run_id} ===", flush=True)
    print(f"    {' '.join(cmd)}", flush=True)
    print(f"    log -> {log_path}", flush=True)

    t0 = time.time()
    try:
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=spec.get("timeout_s", 86400))
        rec["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        rec["exit_code"] = -1
        rec["error"] = "timeout"
    except Exception as exc:                      # noqa: BLE001 - must never kill the sweep
        rec["exit_code"] = -2
        rec["error"] = f"{type(exc).__name__}: {exc}"

    rec["wall_s"] = round(time.time() - t0, 1)

    # The honest part: an exit code of 0 is NOT success. A checkpoint must exist.
    cks = checkpoints_of(out_dir)
    rec["n_checkpoints"] = len(cks)
    rec["last_checkpoint"] = cks[-1].name if cks else None

    if rec["exit_code"] == 0 and cks:
        rec["status"] = "ok"
    else:
        rec["status"] = "failed"
        if rec["exit_code"] == 0 and not cks:
            rec["error"] = "exit 0 but no checkpoint was written"
        if log_path.exists():
            tail = log_path.read_text(errors="replace").splitlines()[-25:]
            rec["log_tail"] = "\n".join(tail)

    n_pruned, gb_freed = prune_checkpoints(out_dir, keep)
    rec["pruned"] = n_pruned
    rec["gb_freed_by_prune"] = round(gb_freed, 2)
    rec["size_gb"] = round(dir_size_gb(out_dir), 2)

    # Re-check AFTER pruning, and re-point `last`. A prune that removed the
    # checkpoint `last` refers to would otherwise leave a dangling symlink that
    # every downstream loader follows into a FileNotFoundError.
    surviving = checkpoints_of(out_dir)
    rec["n_checkpoints_after_prune"] = len(surviving)
    last_link = out_dir / "checkpoints" / "last"
    if surviving:
        if last_link.is_symlink() and not last_link.resolve().exists():
            last_link.unlink()
            last_link.symlink_to(surviving[-1].name)
            rec["relinked_last"] = surviving[-1].name
    if rec["status"] == "ok" and not surviving:
        rec["status"] = "failed"
        rec["error"] = "pruning left no checkpoint behind"
    rec["free_gb_after"] = round(free_gb(REPO), 1)
    rec["finished_at"] = utcnow()

    with results_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"    status={rec['status']} wall={rec['wall_s']}s "
          f"ckpts={rec['n_checkpoints']} size={rec['size_gb']}GB "
          f"free={rec['free_gb_after']}GB", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, default=SWEEPS / "queue.json")
    ap.add_argument("--results", type=Path, default=SWEEPS / "results.jsonl")
    ap.add_argument("--log-dir", type=Path, default=SWEEPS / "logs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run, touch nothing")
    args = ap.parse_args()

    problems = preflight()
    if problems and not args.dry_run:
        print("PREFLIGHT FAILED -- refusing to start:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    for p in problems:
        print(f"(dry-run) would block on: {p}")

    if not args.queue.exists():
        print(f"no queue at {args.queue}; nothing to do.", file=sys.stderr)
        return 1

    queue = json.loads(args.queue.read_text())
    runs = queue["runs"] if isinstance(queue, dict) else queue
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    done = load_done(args.results)
    pending = [r for r in runs if r["id"] not in done]

    print(f"queue: {len(runs)} runs, {len(done)} already terminal, {len(pending)} to go")
    print(f"free disk: {free_gb(REPO):.1f} GB")

    if args.dry_run:
        for r in pending:
            print(f"  would run {r['id']}: {' '.join(r['args'])}")
        return 0

    for spec in pending:
        f = free_gb(REPO)
        if f < DISK_MIN_TO_CONTINUE:
            print(f"\nSTOPPING: only {f:.1f} GB free, below the "
                  f"{DISK_MIN_TO_CONTINUE} GB floor. Remaining runs left queued.",
                  flush=True)
            break
        if f < DISK_MIN_TO_START:
            print(f"\nSTOPPING before {spec['id']}: {f:.1f} GB free is under the "
                  f"{DISK_MIN_TO_START} GB start threshold.", flush=True)
            break
        run_one(spec, args.results, args.log_dir)

    print(f"\nsweep driver done. free disk: {free_gb(REPO):.1f} GB")
    print(f"results: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

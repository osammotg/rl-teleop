#!/usr/bin/env python3
"""Rank every checkpoint the sweep produced, on the HELD-OUT recordings only.

WHY THIS EXISTS SEPARATELY FROM grasp_proxy.py
----------------------------------------------
`grasp_proxy.py`'s `score()` already takes a `groups` filter, but its __main__
block does not pass one -- so running it directly scores on every episode,
including the ones the checkpoint trained on. That is exactly the case its own
docstring says has NO RESOLUTION: `close_rate` saturates at 1.000 and
`approach_l1` degenerates into tracking the training loss. Run that way it will
happily print a confident-looking table in which every checkpoint is excellent.

This wrapper exists so that cannot happen by accident. It reads the holdout
recordings out of `split_group3.json` -- the same file the sweep trained
against -- and refuses to score anything on data that checkpoint saw.

RANKING CAVEAT, which must survive into whatever reads this
-----------------------------------------------------------
This is a proxy, not a rollout. Its one validation (RESEARCH.md S4.7) is that
it ranks the DEPLOYED checkpoint above the from-scratch one on held-out grasps
7/7 paired, while the training loss ranks them the other way round -- i.e. it
agrees with the robot where the loss disagrees. That is real evidence, and it
is n=7 on the hardest band of the workspace. It is not a substitute for putting
the top two or three candidates on the arm.

Usage:
    cd ~/Desktop/kitting-v2/rl-teleop
    LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_v2_wrist \
      .venv/bin/python sweeps/rank_sweep.py
    # or point it at specific checkpoints
    ... sweeps/rank_sweep.py sweeps/outputs/s_kl1/checkpoints/020000/pretrained_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SWEEPS = Path(__file__).resolve().parent
sys.path.insert(0, str(SWEEPS))

from grasp_proxy import (  # noqa: E402
    LeRobotDataset, REPO, episode_to_recording, score,
)

SPLIT = SWEEPS / "split_group3.json"
OUT_ROOT = SWEEPS / "outputs"


def discover() -> list[Path]:
    """Every real checkpoint under sweeps/outputs, newest step per run first.

    Excludes the `last` symlink -- following it would score the same weights
    twice under two names and make a run look like two agreeing results.
    """
    found = []
    for run in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
        cks = sorted(
            (d for d in (run / "checkpoints").iterdir()
             if d.is_dir() and not d.is_symlink() and d.name.isdigit()),
            key=lambda p: int(p.name),
        ) if (run / "checkpoints").is_dir() else []
        found += [c / "pretrained_model" for c in cks
                  if (c / "pretrained_model").is_dir()]
    return found


def main() -> int:
    if not SPLIT.exists():
        print(f"FATAL: {SPLIT} missing -- cannot establish what was held out, "
              f"and scoring on seen data is worse than not scoring.", file=sys.stderr)
        return 2
    split = json.loads(SPLIT.read_text())
    holdout = set(split["holdout_recordings"])
    print(f"holdout recordings ({len(holdout)}): {sorted(holdout)}")

    ckpts = [Path(a) for a in sys.argv[1:]] or discover()
    if not ckpts:
        print("no checkpoints found yet -- the sweep is probably still running.")
        return 0

    ds = LeRobotDataset(REPO)
    e2r = episode_to_recording(ds)

    # Re-key the recording labels from the SPLIT FILE, not from the dataset.
    #
    # grasp_proxy.episode_to_recording() derives the recording from dataset
    # metadata, and on yam_grasp_v2_wrist that collapses all 69 episodes onto a
    # SINGLE recording -- so filtering by `groups` matches either everything or
    # nothing, and the holdout silently selects zero episodes. Scored that way
    # the ranking is not merely wrong, it is empty while looking fine.
    #
    # split_group3.json carries its own episode -> recording map, which is what
    # the grouping was actually built from, so that is the authority here.
    split_map = {int(k): v for k, v in split["episode_to_recording"].items()}
    missing = set(e2r) - set(split_map)
    if missing:
        print(f"FATAL: {len(missing)} episodes absent from the split's "
              f"episode_to_recording map; refusing to guess their group.",
              file=sys.stderr)
        return 2
    e2r = {ep: (split_map[ep], gi, res) for ep, (_, gi, res) in e2r.items()}

    n_eval = sum(1 for rec, _, _ in e2r.values() if rec in holdout)
    print(f"{len(e2r)} episodes total, {n_eval} in the holdout")
    if n_eval == 0:
        print("FATAL: the holdout selects zero episodes -- nothing can be ranked.",
              file=sys.stderr)
        return 2
    print()

    rows = []
    for ck in ckpts:
        try:
            r = score(str(ck), ds, e2r, groups=holdout)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {ck}: FAILED {type(exc).__name__}: {exc}")
            continue
        # A score computed over zero episodes is not a score.
        if r["n"] == 0:
            print(f"  {ck}: no holdout episodes scored -- refusing to rank")
            continue
        r["ckpt"] = str(ck.relative_to(SWEEPS)) if SWEEPS in ck.parents else str(ck)
        r["run"] = ck.parts[-4] if len(ck.parts) >= 4 else "?"
        rows.append(r)
        print(f"  scored {r['run']:<18} n={r['n']:<3} score={r['score']:.4f}")

    rows.sort(key=lambda r: -r["score"])
    print(f"\n{'run':<20}{'n':>4}{'close_rate':>12}{'close_dt_s':>12}"
          f"{'approach_l1':>13}{'score':>10}  flag")
    print("-" * 79)
    for r in rows:
        flag = "" if r["close_rate"] >= 1.0 else "  <-- DID NOT ALWAYS CLOSE"
        print(f"{r['run']:<20}{r['n']:>4}{r['close_rate']:>12.3f}"
              f"{r['close_dt_med_s']:>12.3f}{r['approach_l1_rad']:>13.5f}"
              f"{r['score']:>10.4f}{flag}")

    # Surface close_rate separately and loudly, because the composite score hides
    # it. Measured in this sweep: chunk_size=50 scored 0.7503 against a seed band
    # floor of 0.7510 -- inconclusive on score alone -- while its close_rate was
    # 0.944 against 1.000 for every baseline checkpoint. The categorical drop in
    # the thing that actually matters was nearly buried by the blend.
    #
    # Worse, approach_l1 moved the OPPOSITE way: shrinking the chunk produced a
    # visibly smoother approach that grasped less often. A ranking that reads
    # only the composite would have promoted it.
    #
    # This codebase's recurring failure is a single number that cannot say "no"
    # (a witness saturating at 1.000, a val loss that anti-correlates with
    # success). A blended score is one of those. So close_rate gets its own line.
    bad = [r for r in rows if r["close_rate"] < 1.0]
    if bad:
        print(f"\n!! {len(bad)} checkpoint(s) failed to close the jaws on some held-out "
              f"grasps.\n   Rank these BELOW anything at close_rate 1.000 regardless of "
              f"score -- a policy\n   that does not close is not a better policy with a "
              f"worse number, it is not a policy.")
        for r in bad:
            missed = round((1.0 - r["close_rate"]) * r["n"])
            print(f"     {r['run']:<20} close_rate {r['close_rate']:.3f} "
                  f"({missed} of {r['n']} held-out grasps)")

    out = SWEEPS / "ranking.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print("\nThese are CANDIDATES. The proxy agrees with the robot where the loss "
          "does not,\nbut its validation is n=7. Put the top two or three on the arm "
          "before believing\nany of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

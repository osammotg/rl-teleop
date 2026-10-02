#!/usr/bin/env python3
"""Why did the live auto-advance stop advancing?

Replays recorded episodes through the SAME OnlineGripSegmenter + transport gate
the live cockpit uses, and prints the gate's decision for every release. No
robot, no cameras — pure replay of yam_<arm>.mcap.

A release only advances the kit pointer when ALL THREE hold (live.py:213):

    outcome == "success"  AND  lifted is not False  AND  dxy_m >= MIN_TRANSPORT_M

so when the cockpit stops stepping forward, exactly one of those is failing and
this tells you which, with the measured numbers.

Usage:
    uv run python tools/diagnose_live_gate.py                      # all episodes
    uv run python tools/diagnose_live_gate.py recordings/20260726  # one day
    uv run python tools/diagnose_live_gate.py --arm left --transport 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.live import OnlineGripSegmenter
from robots_realtime.labeling.mcap_io import read_positions
from robots_realtime.labeling.segmentation import transport_ok

try:
    from robots_realtime.labeling.fk import ForwardKinematics
except Exception:                                              # pragma: no cover
    ForwardKinematics = None


def episode_dirs(root: Path) -> list[Path]:
    if (root / "yam_left.mcap").exists() or (root / "yam_right.mcap").exists():
        return [root]
    # Skip <root>/.trash — episodes the operator threw away in the cockpit
    # archive still live there (delete is a move, so it can be undone), and a
    # diagnostic that re-includes them would report on data nobody wants.
    return sorted(p for p in root.rglob("episode_*")
                  if p.is_dir() and ".trash" not in p.parts)


def analyse(ep: Path, arm: str, urdf: str | None, min_transport: float) -> dict | None:
    mcap = ep / f"yam_{arm}.mcap"
    if not mcap.exists():
        return {"ep": ep.name, "error": f"no yam_{arm}.mcap"}

    try:
        times, pos = read_positions(mcap, f"yam_{arm}")
    except Exception as e:
        return {"ep": ep.name, "error": f"read failed: {e}"}
    if times.size == 0:
        return {"ep": ep.name, "error": "empty timeline"}

    grip = pos[:, C.GRIPPER_JOINT_INDEX]
    ee = None
    fk_err = None
    if urdf and ForwardKinematics is not None:
        try:
            ee = ForwardKinematics(urdf).ee_positions(pos[:, : C.N_ARM_JOINTS])
        except Exception as e:
            fk_err = str(e)

    seg = OnlineGripSegmenter()
    releases, closes = [], 0
    for i in range(times.size):
        ev = seg.push(float(times[i]), float(grip[i]),
                      None if ee is None else ee[i])
        if ev is None:
            continue
        if ev.kind == "close":
            closes += 1
        else:
            moved = transport_ok(ev.dxy_m, min_transport)
            releases.append({
                "t": round(ev.t, 2),
                "outcome": getattr(ev, "outcome", None),
                "lifted": getattr(ev, "lifted", None),
                "dxy_m": None if ev.dxy_m is None else round(float(ev.dxy_m), 4),
                "moved": moved,
                "advanced": (getattr(ev, "outcome", None) == "success"
                             and getattr(ev, "lifted", None) is not False
                             and moved),
            })

    return {
        "ep": ep.name,
        "dur_s": round(float(times[-1] - times[0]), 1),
        "samples": int(times.size),
        "grip_range": (round(float(grip.min()), 4), round(float(grip.max()), 4)),
        "fk": "ok" if ee is not None else f"MISSING ({fk_err or 'no urdf'})",
        "closes": closes,
        "releases": releases,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="recordings")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--urdf", default="urdf/yam.urdf")
    ap.add_argument("--transport", type=float, default=C.MIN_TRANSPORT_M)
    a = ap.parse_args(argv)

    eps = episode_dirs(Path(a.root))
    if not eps:
        print(f"no episodes under {a.root}")
        return 1

    print(f"gate: outcome==success AND lifted!=False AND dxy>={a.transport}m "
          f"(MIN_TRANSPORT_M={C.MIN_TRANSPORT_M}, MIN_LIFT_M={C.MIN_LIFT_M})\n")

    tot_rel = tot_adv = 0
    blocked_by = {"outcome": 0, "lifted": 0, "transport": 0, "no_pose": 0}

    for ep in eps:
        r = analyse(ep, a.arm, a.urdf, a.transport)
        if r is None:
            continue
        if "error" in r:
            print(f"{r['ep']:<34} !! {r['error']}")
            continue

        adv = sum(1 for x in r["releases"] if x["advanced"])
        tot_rel += len(r["releases"])
        tot_adv += adv
        flag = "" if adv else "   <-- NEVER ADVANCED"
        print(f"{r['ep']:<34} {r['dur_s']:>6.1f}s  grip{r['grip_range']}  "
              f"fk={r['fk']}  closes={r['closes']}  releases={len(r['releases'])}  "
              f"advanced={adv}{flag}")

        for x in r["releases"]:
            if x["advanced"]:
                continue
            if x["dxy_m"] is None:
                why = "no EE pose (gate cannot run — fails OPEN live)"; blocked_by["no_pose"] += 1
            elif x["outcome"] != "success":
                why = f"outcome={x['outcome']}"; blocked_by["outcome"] += 1
            elif x["lifted"] is False:
                why = "lifted=False (never rose MIN_LIFT_M)"; blocked_by["lifted"] += 1
            else:
                why = (f"only moved {x['dxy_m']}m < {a.transport}m"); blocked_by["transport"] += 1
            print(f"      t={x['t']:<8} blocked: {why}")

    print(f"\ntotal: {tot_adv}/{tot_rel} releases advanced the pointer")
    if tot_rel:
        print("blocked by: " + ", ".join(f"{k}={v}" for k, v in blocked_by.items() if v))
        if blocked_by["transport"] and blocked_by["transport"] >= tot_rel * 0.5:
            print("\n  → the TRANSPORT THRESHOLD is the dominant blocker. Either the "
                  f"operator's carry is shorter than {a.transport}m, or FK/URDF is "
                  "reporting the wrong scale. Re-run with --transport 0.05 to compare.")
        if blocked_by["no_pose"]:
            print("\n  → releases with NO EE POSE: FK is not loading. Live, this fails "
                  "OPEN (advances on every re-grasp) and sets gate_off in the cockpit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

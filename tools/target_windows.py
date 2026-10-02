#!/usr/bin/env python3
"""THE TARGET SIDECAR FOR export_lerobot.py: which packet, as two normalised numbers.

Reads every <episode>/target_cycles.json the recording loop wrote and turns it into the
three floats that get appended to `observation.state`:

    state[7] target_u     = 2*cx/W - 1    in [-1, 1]
    state[8] target_v     = 2*cy/H - 1    in [-1, 1]
    state[9] target_valid = +1 supplied, -1 masked/unknown  (u = v = 0 when -1)

WHY NORMALISED TOP-FRAME PIXELS AND NOT METRES. Every metric route on this rig is
additive error and the two largest terms are unresolved: pixel->base projection
intersects the work plane, so a packet lying on another localises up to 26 mm wrong
against a 73 mm packet (wrist_ocr/targets.py:174-178), and kitbox's own overlay check
puts the modelled bin 119 mm from the detected one -- 1.4x the homography's p90
(docs/reference/kitbox-camera-overlay.md:128-145). The top camera does not move. A
normalised (u, v) in its frame designates a packet completely, with NO calibration at
all: no homography, no plane assumption, no base offset, no P_TOOL. Every one of those
error sources is bypassed rather than estimated.

WHY THE LABEL IS ALWAYS `moved_n`, NEVER `assigned_n`. On a cycle where the operator did
not take the packet he was assigned, conditioning on the assignment would be a WRONG
demonstration -- pointing the channel at a packet the arm demonstrably did not go to.
That is the exact confound this corpus exists to break, so `assigned_n` is used only to
COUNT non-redundant mass (see target_corpus_audit.py); the geometry always comes from
what actually moved. Cycles where which_moved() abstained (`moved_n: null`) carry
target_valid = -1 rather than a guess.

THE FRAME SIZE IS NEVER ASSUMED. Candidate coordinates are in the BUS frame -- measured
480x270 on 2026-09-21 -- while the homography and every doc about the top camera are in
1280x720. Normalising against the wrong one puts every target off by 2.667x, and it
would train perfectly happily. So W and H come from a real PNG on disk or the cycle is
REFUSED; there is no default.

MASKING IS PER-WINDOW, NEVER PER-FRAME. The goal is constant over a cycle by
construction, so per-frame flicker would be a train/deploy mismatch the model can detect
and exploit. p_m = 0.5 is the validated instance (NoMaD, act-place-conditioning-20260916
S10). The seed and the fraction are returned in the manifest so a dataset card can state
exactly which windows were blind.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pathlib
import sys

TARGET_DIMS = ("target_u", "target_v", "target_valid")
SEGBENCH_SCRATCH = (pathlib.Path(__file__).resolve().parents[2]
                    / "yam-pick-pipeline" / "perception" / "segbench" / "scenarios")

# DUPLICATE-MASK DEDUPE, and why the number is 10.
#
# MEASURED 2026-09-21 over every candidate pair the recording loop logged that day (81
# pairs, 8 cycles): the centroid-separation distribution is cleanly bimodal with a GAP
# between 5.2 px and 18.8 px. All seven pairs at <= 5.2 px also have one bbox fully
# CONTAINING the other -- SAM2 returning a packet and a sub-region of the same packet as
# two separate candidates. The smallest separation between two genuinely distinct
# packets is 18.8 px; a packet is ~30 px across in this 480x270 bus frame (73 mm at
# ~2.4 mm/px). So the threshold comes from the measured gap, not from a guess, and it is
# set at the conservative end of it.
#
# THIS IS NOT COSMETIC. A duplicate pair breaks which_moved() outright: when the packet
# is taken, BOTH co-located candidates lose their neighbour by nearly the same distance,
# so the margin rule `(far - runner) >= thresh/2` can never be met and the cycle abstains
# with moved_n = null. Measured on 2026-09-21: two of the three diagnosable abstentions
# were caused by exactly this (margins of 3.8 px and 0.4 px between candidates 5.0 px and
# 1.1 px apart), NOT by the operator re-staging. That is the dominant loss of trainable
# cycles, and it is recoverable offline from what is already on disk.
DUP_SEP_PX = float(os.environ.get("TARGET_DUP_SEP_PX", "10.0"))


def _frame_size(ep: pathlib.Path, cyc: dict):
    """(W, H, provenance) or (None, None, why). Mirrors target_corpus_audit.frame_size.

    Deliberately duplicated rather than imported: that module lives in
    training_experiment/ and runs in a different venv, and an import across the two
    would make an export depend on a tree it has no other reason to see. If these two
    resolvers ever disagree, a human decides and both get a dated comment."""
    try:
        from PIL import Image                                   # noqa: PLC0415
    except ImportError:
        return None, None, "no PIL"
    k, ph = cyc.get("k"), cyc.get("phase") or "?"
    tries = [(ep / "segmentation" / f"c{k:03d}_{ph}_pre.png", "episode/segmentation")]
    if cyc.get("slug"):
        tries.append((SEGBENCH_SCRATCH / str(cyc["slug"]) / "top.png",
                      f"segbench/{cyc['slug']}"))
    for path, why in tries:
        try:
            if path.exists():
                with Image.open(path) as im:
                    return int(im.size[0]), int(im.size[1]), why
        except Exception:                                       # noqa: BLE001
            continue
    return None, None, "no pre-frame png on disk"


def to_uv(cx: float, cy: float, W: int, H: int) -> tuple[float, float] | None:
    """(u, v) in [-1, 1] for a candidate centroid, or None if it falls outside the frame.

    THE ONE DEFINITION. The export (this module) and the inference-time target supply
    (yam-pick-pipeline/target_supply.py) must agree character for character: a second copy
    is a second chance to normalise against 1280x720 instead of the 480x270 bus frame,
    which is a silent 2.667x error that trains and deploys without complaining.

    Returns None rather than clamping. A centroid outside the frame means the frame was
    resolved wrong, not that the packet moved -- clamping would turn a resolver bug into a
    plausible-looking target pointing at the frame edge."""
    if not W or not H:
        return None
    u = 2.0 * float(cx) / float(W) - 1.0
    v = 2.0 * float(cy) / float(H) - 1.0
    if not (-1.0 <= u <= 1.0 and -1.0 <= v <= 1.0):
        return None
    return u, v


def dedupe_candidates(cands: list, sep_px: float = DUP_SEP_PX) -> list[int]:
    """-> the INDICES of `cands` to keep: drop any candidate whose centroid is within
    `sep_px` of an EARLIER (higher-ranked) one.

    Indices, not candidates, because the caller has to index the parallel `moved_px`
    list the recording loop logged alongside them. Rank order is preserved and the
    survivor is always the higher-ranked mask of a duplicate pair, so `n == 1` can never
    be dropped in favour of its own sub-region."""
    keep: list[int] = []
    for i, c in enumerate(cands):
        if any(math.hypot(c["cx"] - cands[j]["cx"], c["cy"] - cands[j]["cy"]) < sep_px
               for j in keep):
            continue
        keep.append(i)
    return keep


def recompute_moved(cands: list, moved_px: list, sep_px: float = DUP_SEP_PX,
                    thresh_px: float = 25.0):
    """Re-run which_moved()'s decision on the DEDUPED candidate set.

    The rule is copied verbatim from wrist_ocr/targets.py:196-198 rather than imported:
    that module is live, the recording-loop author edits it during sessions, and an
    import would let an edit there silently change a label already written into a
    training set. If the two ever drift, a human decides and this comment gets a date.

        thresh = 40 mm / mm_per_px, or 25 px when mm_per_px is unknown (it always is,
        here -- the loop calls which_moved without one)
        moved iff  far >= thresh  AND  (far - runner) >= thresh * 0.5

    -> (n or None, reason). Only the candidate set changes; not one distance is
    recomputed, so this cannot invent movement that was not measured at record time."""
    if not cands or not moved_px or len(moved_px) != len(cands):
        return None, "no usable candidate/distance pair"
    keep = dedupe_candidates(cands, sep_px)
    if len(keep) < 1:
        return None, "nothing survived dedupe"
    d = sorted(((moved_px[i], i) for i in keep))
    far, i = d[-1]
    runner = d[-2][0] if len(d) > 1 else 0.0
    if far >= thresh_px and (far - runner) >= thresh_px * 0.5:
        return cands[i]["n"], (f"far={far} (n={cands[i]['n']}) runner={runner} "
                               f"over {len(keep)}/{len(cands)} deduped candidates")
    return None, (f"margin {far - runner:.1f} < {thresh_px * 0.5} "
                  f"over {len(keep)}/{len(cands)} deduped candidates")


def _masked(ep_name: str, k: int, seed: int, p_mask: float) -> bool:
    """Deterministic per-(episode, cycle) mask. Hash rather than random.random() so a
    re-export with the same seed reproduces the same blind windows exactly, without
    depending on iteration order."""
    if p_mask <= 0:
        return False
    h = hashlib.sha256(f"{seed}:{ep_name}:{k}".encode()).digest()
    return (int.from_bytes(h[:8], "big") / 2 ** 64) < p_mask


def load_targets(root, p_mask: float = 0.0, seed: int = 0,
                 phases=("isolate", "pick"), dedupe: bool = True) -> dict:
    """-> {"targets": {episode_name: [(t_start, t_end, u, v, valid), ...]},
           "manifest": {...}}

    `root` is a recordings day dir (or any parent). Episodes with no
    target_cycles.json simply do not appear -- it is export_lerobot's job to REFUSE
    loudly when an episode it was asked to export is absent from this map, because a
    silently-unconditioned episode inside a conditioned dataset is the failure mode
    that costs a whole training run."""
    root = pathlib.Path(root)
    out: dict[str, list] = {}
    stats = {"episodes": 0, "cycles": 0, "supplied": 0, "masked": 0,
             "no_moved_n": 0, "no_frame": 0, "no_candidate": 0, "bad": 0,
             "phase_skipped": 0, "frame_sources": {}}
    for tc in sorted(glob.glob(str(root / "**" / "target_cycles.json"), recursive=True)):
        ep = pathlib.Path(tc).parent
        try:
            doc = json.load(open(tc))
        except Exception:                                       # noqa: BLE001
            continue
        rows = []
        stats["episodes"] += 1
        for c in doc.get("cycles") or []:
            t0, t1 = c.get("t_start"), c.get("t_end")
            if t0 is None or t1 is None or not (t1 > t0):
                continue
            stats["cycles"] += 1
            if (c.get("phase") or "") not in phases:
                stats["phase_skipped"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            if c.get("bad"):
                stats["bad"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            mn = c.get("moved_n")
            if mn is None and dedupe:
                # Recover the cycles a duplicate SAM2 mask made which_moved() abstain
                # on. Nothing is invented: the distances were measured at record time
                # and only the candidate SET changes. Measured 2026-09-21: this turns
                # 2 of 6 null cycles into labelled ones and re-labels none.
                mn, why = recompute_moved(c.get("candidates") or [],
                                          c.get("moved_px") or [])
                if mn is not None:
                    stats["recovered_by_dedupe"] = stats.get("recovered_by_dedupe", 0) + 1
            if mn is None:
                stats["no_moved_n"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            hit = next((a for a in (c.get("candidates") or []) if a.get("n") == mn), None)
            if hit is None:
                stats["no_candidate"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            W, H, prov = _frame_size(ep, c)
            if not W or not H:
                stats["no_frame"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            stats["frame_sources"][prov] = stats["frame_sources"].get(prov, 0) + 1
            if _masked(ep.name, int(c.get("k") or 0), seed, p_mask):
                stats["masked"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            uv = to_uv(hit["cx"], hit["cy"], W, H)
            if uv is None:
                # The resolver picked the wrong picture; see to_uv's own note.
                stats["no_frame"] += 1
                rows.append((float(t0), float(t1), 0.0, 0.0, -1.0))
                continue
            u, v = uv
            stats["supplied"] += 1
            rows.append((float(t0), float(t1), u, v, 1.0))
        if rows:
            out[ep.name] = sorted(rows, key=lambda r: r[0])
    return {"targets": out,
            "manifest": {"root": str(root), "p_mask": p_mask, "seed": seed,
                         "phases": list(phases), "dims": list(TARGET_DIMS), **stats}}


def lookup(rows: list, t: float) -> tuple[float, float, float]:
    """(u, v, valid) for one instant, by TIME CONTAINMENT -- never by index.

    target_cycles.json's `k` and phases.json's `seg` numbering are produced by different
    processes and are not guaranteed to agree, so joining them positionally would
    silently mislabel whole windows. Outside every cycle -> (0, 0, -1)."""
    for t0, t1, u, v, valid in rows:
        if t0 <= t <= t1:
            return (u, v, valid)
    return (0.0, 0.0, -1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--p-mask", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true",
                    help="do NOT recover cycles which_moved abstained on because of "
                         "duplicate SAM2 masks (see DUP_SEP_PX)")
    ap.add_argument("--out", help="write the target map + manifest here")
    a = ap.parse_args()
    doc = load_targets(a.root, p_mask=a.p_mask, seed=a.seed, dedupe=not a.no_dedupe)
    m = doc["manifest"]
    print(json.dumps(m, indent=2))
    for ep, rows in sorted(doc["targets"].items()):
        sup = sum(1 for r in rows if r[4] > 0)
        print(f"  {ep}: {len(rows)} cycles, {sup} with a target")
    if a.out:
        json.dump(doc, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0 if m["supplied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

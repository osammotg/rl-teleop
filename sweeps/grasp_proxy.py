#!/usr/bin/env python3
"""Offline proxy score for an ACT grasp checkpoint. NOT val loss.

Ranks checkpoints by whether they REPRODUCE THE GRASP, evaluated on the frames
where the grasp actually happens, with a holdout grouped by source recording.

Why not loss: see sweeps/RESEARCH.md S4. Two reasons in one line each --
(a) LeRobot's ACT l1 term divides by B*chunk_size*dim including the padded
    positions, so the number is scaled by the supervised fraction and is not
    comparable across chunk_size or across datasets with different episode
    lengths; (b) 96% of the frames in a window are approach/lift, so a policy
    that never closes its jaws pays ~4% of the loss for it.

VALIDATED, once, on real held-out data -- see RESEARCH.md S4.7. The zone filter
dropped 8 grasps from yam_grasp_v1 when building yam_grasp_v2_wrist, so those 8
are unseen by every v2-line checkpoint (indices in sweeps/holdout_v1_indices.json).
Scored there, this metric ranks the DEPLOYED ft50k/006000 above v2-scratch-50000
on 7/7 episodes for both close_dt and approach_l1 -- while the training loss ranks
them the other way round (0.0509 vs 0.0392). Generalisation gap 1.43x vs 2.31x.

It has NO resolution when run on data the checkpoint trained on: close_rate
saturates at 1.000 and approach_l1 just tracks the loss (RESEARCH.md S4.4, S4.5).
So: only ever score a checkpoint on recordings withheld from its training run.

Usage:
    LEROBOT_PREDECODED_ROOT=~/.cache/lerobot-predecoded/yam_grasp_v2_wrist \
    .venv/bin/python sweeps/grasp_proxy.py CKPT [CKPT ...]
"""
import os, sys, json, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FAST = Path.home() / "Desktop/lab/lerobot-fast"
if os.environ.get("LEROBOT_PREDECODED_ROOT"):
    sys.path.insert(0, str(FAST)); import predecoded_patch  # noqa

import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

REPO = os.environ.get("PROXY_REPO", "ETHRC/yam_grasp_v2_wrist")
FPS = 30
CLOSE = 0.45          # C.GRIPPER_CLOSE_ENTER, the threshold that labelled the corpus
DEMO_GRASPS = Path(__file__).resolve().parents[2] / "yam-pick-pipeline/results/demo_grasps.json"


def episode_to_recording(ds):
    """Reconstruct the manifest AUDIT S2 says does not exist.

    Matches each exported episode to its demo_grasps key by nearest-neighbour on
    the 6 arm joints at the annotated grasp instant. Measured residual over
    yam_grasp_v2_wrist: max 0.004 rad, 69/69 matched to DISTINCT keys.
    """
    g = json.load(open(DEMO_GRASPS))
    Q = np.array([x["q"] for x in g]); keys = [x["key"] for x in g]
    S = ds.hf_dataset.with_format("numpy")["observation.state"]
    ei = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"])
    out = {}
    for e in np.unique(ei):
        s = S[ei == e][:, :6]
        d = np.linalg.norm(s[:, None, :] - Q[None, :, :], axis=2)
        i, j = np.unravel_index(d.argmin(), d.shape)
        out[int(e)] = (keys[j].split("#")[0], int(i), float(d[i, j]))
    return out


def close_frame(a_grip, close=CLOSE):
    """First index where the demonstrated gripper action crosses below `close`."""
    b = np.where(a_grip < close)[0]
    return int(b[0]) if len(b) else None


@torch.no_grad()
def score(ckpt, ds, e2r, groups=None, device="cuda", lead_s=1.0):
    """-> dict of proxy components. Higher `score` is better.

    For every held-out episode, query the policy ONCE at `lead_s` before the
    demonstrated gripper close, from the real observation, and read the whole
    predicted chunk. Three things are measured, in falsifiability order:

      close_rate   fraction of episodes whose predicted chunk closes the jaws
                   at all. A checkpoint that scores 0 here cannot grasp, and no
                   loss number says so.
      close_dt     |predicted close step - demonstrated close step| / fps, over
                   the episodes that DO close. Timing, not magnitude: the
                   gripper channel is min-max normalised PER RECORDING
                   (export_lerobot.py:418-419) so absolute widths are not
                   comparable across episodes, but crossings are.
      approach_l1  mean per-step L1 on the 6 arm joints over the VALID
                   (non-padded) part of the chunk only, in radians. Reported
                   per step so it is comparable across chunk_size.
    """
    pol = ACTPolicy.from_pretrained(ckpt).to(device).eval()
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)

    ei = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"])
    A = ds.hf_dataset.with_format("numpy")["action"]
    lead = int(round(lead_s * FPS))
    closes, dts, l1s, n = 0, [], [], 0
    for e, (rec, gi, _res) in sorted(e2r.items()):
        if groups is not None and rec not in groups:
            continue
        idx = np.where(ei == e)[0]
        a = A[idx]
        cf = close_frame(a[:, 6])
        if cf is None or cf - lead < 0:
            continue
        q = int(idx[cf - lead])
        item = ds[q]
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()
                 if isinstance(v, torch.Tensor) and k.startswith("observation")}
        batch = pre(batch)
        chunk = pol.predict_action_chunk(batch)          # (1, chunk, 7) normalised
        # Unnormalise by hand rather than through the postprocessor pipeline: the
        # pipeline's converters are shaped for a single (7,) action, not a whole
        # (chunk, 7) block, and MEAN_STD is the only mapping in play here.
        ch = chunk[0].cpu().numpy()
        st = ds.meta.stats["action"]
        ch = ch * st["std"] + st["mean"]
        n += 1
        pc = close_frame(ch[:, 6])
        if pc is not None:
            closes += 1
            dts.append(abs(pc - lead) / FPS)
        valid = min(ch.shape[0], len(a) - (cf - lead))
        l1s.append(float(np.abs(ch[:valid, :6] - a[cf - lead:cf - lead + valid, :6]).mean()))
    close_rate = closes / max(n, 1)
    med_dt = float(np.median(dts)) if dts else float("nan")
    ap = float(np.mean(l1s)) if l1s else float("nan")
    # single number: closing at all dominates; timing next; smoothness last
    s = close_rate - 0.5 * (0.0 if np.isnan(med_dt) else min(med_dt, 1.0)) - 2.0 * ap
    return {"n": n, "close_rate": close_rate, "close_dt_med_s": med_dt,
            "approach_l1_rad": ap, "score": s}


if __name__ == "__main__":
    ds = LeRobotDataset(REPO)
    e2r = episode_to_recording(ds)
    recs = sorted({r for r, _, _ in e2r.values()})
    print(f"{len(e2r)} episodes over {len(recs)} recordings")
    for ck in sys.argv[1:]:
        print(ck, score(ck, ds, e2r))

#!/usr/bin/env python3
"""Make a fine-tunable checkpoint that expects FEWER cameras than it was trained on.

WHY THIS IS SOUND, and not a hack. Verified against the 50k checkpoint: ACT has
**zero camera-keyed tensors**.

    model.backbone.*                    100 tensors, ONE shared ResNet18
    model.encoder_img_feat_input_proj   (512, 512, 1, 1), shared
    encoder_cam_feat_pos_embed          absent -- sinusoidal, computed not learned
    encoder_1d_feature_pos_embed        (2, 512) -- latent + state, not cameras

Every camera is pushed through the same backbone in a loop
(modeling_act.py:473) and its position encoding is generated from the feature
map. So camera count changes only the NUMBER OF VISION TOKENS entering the
transformer encoder, never a weight shape. Dropping a camera is therefore a
config edit, not a surgery on the network.

WHY THE CONFIG MUST BE EDITED AT ALL. `--policy.path` makes lerobot load the
policy config FROM the checkpoint (configs/train.py:86), and make_policy only
re-derives features from the dataset `if not cfg.input_features`
(policies/factory.py:471). Since the checkpoint's config already lists both
cameras, a wrist-only dataset would be fed to a policy still expecting
observation.images.top. This writes a copy whose input_features match the
dataset you are about to fine-tune on.

The normalizer safetensors is copied UNCHANGED. It carries stats for the
dropped camera too, which are simply never looked up -- Normalize iterates the
config's features, not the stat file's keys. Rewriting it would risk corrupting
the stats that DO matter (observation.state, action, the surviving camera).

Usage:
    python tools/rebase_checkpoint_cameras.py \
        --src outputs/train/act_grasp_dark_noise_20260728/checkpoints/050000/pretrained_model \
        --dst outputs/pretrained/act_50k_wristonly \
        --keep observation.images.wrist
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def rebase(src: Path, dst: Path, keep: list[str]) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    feats = cfg.get("input_features") or {}

    visual = {k for k, v in feats.items() if v.get("type") == "VISUAL"}
    missing = [k for k in keep if k not in feats]
    if missing:
        raise SystemExit(
            f"checkpoint has no input feature(s) {missing}.\n"
            f"it has: {sorted(feats)}")

    dropped = sorted(visual - set(keep))
    if not dropped:
        print(f"nothing to drop -- {src} already expects exactly {sorted(visual)}")

    # Non-visual inputs (observation.state) are ALWAYS kept. Dropping the joints
    # would be a different model, not a camera rebase, and nothing here checks
    # that the weights would still fit.
    cfg["input_features"] = {
        k: v for k, v in feats.items()
        if v.get("type") != "VISUAL" or k in keep
    }

    if dst.exists():
        raise SystemExit(f"{dst} already exists; refusing to overwrite a checkpoint")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    # train_config.json is provenance, not load-bearing for --policy.path, but a
    # stale copy claiming two cameras is exactly the kind of thing someone reads
    # later and believes.
    tc_path = dst / "train_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if isinstance(tc.get("policy"), dict):
            tc["policy"]["input_features"] = cfg["input_features"]
        tc["_rebased_from"] = str(src)
        tc["_rebased_dropped"] = dropped
        tc_path.write_text(json.dumps(tc, indent=2))

    return {"kept": sorted(cfg["input_features"]), "dropped": dropped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="pretrained_model dir")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--keep", nargs="+", required=True,
                    help="visual input features to KEEP, e.g. observation.images.wrist")
    a = ap.parse_args(argv)

    out = rebase(a.src.resolve(), a.dst.resolve(), a.keep)
    print(f"kept    : {out['kept']}")
    print(f"dropped : {out['dropped'] or 'nothing'}")
    print(f"\n  {a.dst}")
    print("\nfine-tune with:\n"
          f"  --policy.path={a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

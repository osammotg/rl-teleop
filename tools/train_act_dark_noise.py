#!/usr/bin/env python3
"""Train ACT on the exported grasp windows with the `dark_noise` augmentation.

Stock `lerobot-train` cannot express dark_noise: three of its seven transforms
(GaussianNoise, RandomAutocontrast, GaussianBlur) are unknown to
`make_transform_from_config`, which raises ValueError on them. `tools/act_aug`
registers those types and holds the recipe; this script is the entry point that
installs them before the dataset is built, then hands off to LeRobot's own
training loop untouched.

Everything else is LeRobot's: parsing, policy construction, checkpointing, the
step loop. The only injection is `cfg.dataset.image_transforms`.

WHY NOT JUST PASS IT ON THE CLI: the recipe is a 7-entry nested dict of weights,
types and kwargs. Expressed as draccus overrides it is ~30 flags with no place
to record where the numbers came from, and one silent typo (a weight on the
wrong transform) is invisible in the loss curve and expensive on the robot.

WHY n_action_steps IS LEFT AT 100: it is read only by `ACTPolicy.select_action`
(modeling_act.py:97,117) to size the action queue. `chunk_size` is what shapes
the network (decoder_pos_embed, encoder tokens). So the 100-vs-16 question is a
DEPLOY-time knob and needs no retraining -- set it at rollout and A/B it there.
Training it away now would just cost a run.

Usage (all lerobot-train flags work):
    uv run python tools/train_act_dark_noise.py \
        --dataset.repo_id=ETHRC/yam_grasp_v1 \
        --policy.type=act --policy.device=cuda \
        --steps=50000 --save_freq=5000 --batch_size=8 \
        --output_dir=outputs/train/act_grasp_dark_noise
"""
# NB: no `from __future__ import annotations` here. draccus resolves the config
# dataclass from main()'s type annotation via inspect; with PEP 563 the
# annotation is the STRING "TrainPipelineConfig" and parsing dies with
# "must be called with a dataclass type or instance".
import os  # noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── fast data path ──────────────────────────────────────────────────────────
# LEROBOT_PREDECODED_ROOT on its own does NOTHING. The env var is only read by
# lerobot-fast/predecoded_patch.py, which must be IMPORTED to monkey-patch
# `lerobot.datasets.video_utils.decode_video_frames` before the dataset is
# built. Setting the variable and not importing the patch silently trains at
# full decode cost -- or, on this rig, does not train at all: torchcodec cannot
# load (libavutil missing for every FFmpeg version it supports), so the stock
# decode path raises at the first batch. The patch is what makes video reading
# work here, not just what makes it fast.
#
# torchcodec is imported LAZILY inside video_utils.get_decoder(), so installing
# the patch first means that import is never reached.
FAST = Path.home() / "Desktop/lab/lerobot-fast"
if os.environ.get("LEROBOT_PREDECODED_ROOT"):
    if not (FAST / "predecoded_patch.py").exists():
        raise SystemExit(
            f"LEROBOT_PREDECODED_ROOT is set but {FAST}/predecoded_patch.py is missing.\n"
            "Refusing to run: without the patch this trains from the mp4s at full\n"
            "decode cost, or dies on torchcodec. Unset the variable to opt out.")
    sys.path.insert(0, str(FAST))
    import predecoded_patch  # noqa: F401,E402  side-effect import
    print(f"predecoded cache: {os.environ['LEROBOT_PREDECODED_ROOT']}", flush=True)

from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.scripts.lerobot_train import train as _train  # noqa: E402

from tools.act_aug import dark_noise_config, register_transforms  # noqa: E402


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    # Must happen before make_dataset() builds ImageTransforms, which resolves
    # make_transform_from_config by module-global name at call time.
    register_transforms()
    cfg.dataset.image_transforms = dark_noise_config()

    tfs = cfg.dataset.image_transforms
    print("=" * 66)
    print("augmentation: dark_noise "
          f"(max {tfs.max_num_transforms} of {len(tfs.tfs)} per sample, "
          f"random_order={tfs.random_order})")
    for name, t in tfs.tfs.items():
        print(f"  {name:20} w={t.weight:<4} {t.type:<19} {t.kwargs}")
    print("=" * 66, flush=True)

    # __wrapped__ is the undecorated train(); calling `_train` directly would
    # re-parse argv and discard the config we just modified.
    _train.__wrapped__(cfg)


if __name__ == "__main__":
    main()

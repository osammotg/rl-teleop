#!/usr/bin/env python3
"""The `dark_noise` image augmentation recipe, restored for stock LeRobot.

WHY THIS FILE EXISTS. `dark_noise` was defined in the `lerobot-edit-scripts`
fork, which was lost in the 2026-05-16 migration off the `ethrc` account. Only
its config YAML survives, inside
`~/Desktop/_archive/handoff-2026-05-16/patches/lerobot-edit-scripts-commits/0005-*.patch`.
That YAML names three transforms stock LeRobot does not know:

    GaussianNoise · RandomAutocontrast · GaussianBlur

`lerobot.datasets.transforms.make_transform_from_config` supports exactly four
types (Identity, ColorJitter, SharpnessJitter, RandomAffine) and raises
ValueError on anything else. So without this module you cannot run dark_noise --
you can only run a brightness/contrast approximation of it, and the gaussian
noise is the second-heaviest term in the recipe (weight 1.2, behind brightness
at 1.4). Dropping it would change what the name means.

All three transforms exist in torchvision 0.22.1. This module registers them
with LeRobot rather than reimplementing them.

THE UINT8 TRAP. `v2.GaussianNoise` raises `ValueError: Input tensor is expected
to be in float dtype` on uint8 input. LeRobot's image dtype depends on the
decoder backend and has changed across versions, so a run could get most of the
way through training and then die on the first batch that arrives as uint8.
`_SafeGaussianNoise` converts to float, applies, and converts back, so the
transform is correct on either dtype instead of correct-or-fatal.

MEASURED PROVENANCE of the weights and ranges below: they are copied verbatim
from `job_p4_aug_sweep_T4_dark_noise.yaml`, the last iteration of the recipe,
which the operator's own eval notes in `scripts/inference.sh` mark as
"WORKS WELL - daylight tested".

Usage:
    from tools.act_aug import register_transforms, dark_noise_config
    register_transforms()                    # idempotent
    cfg.dataset.image_transforms = dark_noise_config()
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2

from lerobot.datasets import transforms as _lr
from lerobot.datasets.transforms import ImageTransformConfig, ImageTransformsConfig


class _SafeGaussianNoise(torch.nn.Module):
    """v2.GaussianNoise that tolerates uint8 by round-tripping through float.

    sigma is always interpreted in [0,1] units (the scale the recipe was tuned
    in), so a uint8 image gets the same perturbation a float one would.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True):
        super().__init__()
        self.inner = v2.GaussianNoise(mean=mean, sigma=sigma, clip=clip)

    def forward(self, img):
        if not torch.is_floating_point(img):
            orig = img.dtype
            info = torch.iinfo(orig)
            out = self.inner(img.to(torch.float32).div(info.max))
            return out.mul(info.max).round().clamp(info.min, info.max).to(orig)
        return self.inner(img)

    def __repr__(self) -> str:
        return f"_SafeGaussianNoise({self.inner})"


# type name -> builder. Anything already handled by stock LeRobot is absent here.
_EXTRA = {
    "GaussianNoise": lambda **kw: _SafeGaussianNoise(**kw),
    "RandomAutocontrast": lambda **kw: v2.RandomAutocontrast(**kw),
    "GaussianBlur": lambda **kw: v2.GaussianBlur(**kw),
}

_ORIGINAL = None


def register_transforms() -> None:
    """Teach LeRobot the three extra transform types. Idempotent.

    Patches the module-level name that `ImageTransforms.__init__` resolves at
    call time (transforms.py:244), so this works without touching site-packages.
    """
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    _ORIGINAL = _lr.make_transform_from_config

    def make_transform_from_config(cfg: ImageTransformConfig):
        builder = _EXTRA.get(cfg.type)
        if builder is not None:
            return builder(**cfg.kwargs)
        return _ORIGINAL(cfg)

    _lr.make_transform_from_config = make_transform_from_config


# ── the recipe ──────────────────────────────────────────────────────────────
# Verbatim from job_p4_aug_sweep_T4_dark_noise.yaml. Do not "tidy" these numbers:
# they are the ones behind the checkpoint the operator marked as working.
DARK_NOISE_TFS = {
    "brightness":         (1.4, "ColorJitter",        {"brightness": (0.2, 1.1)}),
    "contrast":           (1.0, "ColorJitter",        {"contrast": (0.5, 1.5)}),
    "saturation":         (0.3, "ColorJitter",        {"saturation": (0.2, 1.3)}),
    "hue":                (0.2, "ColorJitter",        {"hue": (-0.03, 0.03)}),
    "gaussian_noise":     (1.2, "GaussianNoise",      {"mean": 0.0, "sigma": 0.06, "clip": True}),
    "random_autocontrast": (0.5, "RandomAutocontrast", {"p": 0.35}),
    "gaussian_blur":      (0.3, "GaussianBlur",       {"kernel_size": 3, "sigma": (0.1, 1.0)}),
}


def dark_noise_config() -> ImageTransformsConfig:
    """The dark_noise ImageTransformsConfig, ready to assign to cfg.dataset."""
    return ImageTransformsConfig(
        enable=True,
        max_num_transforms=3,   # at most 3 of the 7 fire per sample
        random_order=True,
        tfs={name: ImageTransformConfig(weight=w, type=t, kwargs=k)
             for name, (w, t, k) in DARK_NOISE_TFS.items()},
    )


def _selftest() -> int:
    """Prove the recipe actually runs on both dtypes before a 50k-step run."""
    register_transforms()
    register_transforms()                      # idempotent
    tf = _lr.ImageTransforms(dark_noise_config())
    ok = True
    for dtype, img in (("float32", torch.rand(3, 64, 64)),
                       ("uint8", (torch.rand(3, 64, 64) * 255).to(torch.uint8))):
        outs = {tuple(tf(img).shape) for _ in range(40)}
        same = torch.equal(tf(img), tf(img))
        print(f"  {dtype:8} shapes={outs}  varies={not same}")
        if outs != {(3, 64, 64)}:
            print(f"  FAIL: {dtype} changed shape"); ok = False
        if tf(img).dtype != img.dtype:
            print(f"  FAIL: {dtype} dtype not preserved"); ok = False
    # every declared transform must build
    for name, (w, t, k) in DARK_NOISE_TFS.items():
        try:
            _lr.make_transform_from_config(ImageTransformConfig(weight=w, type=t, kwargs=k))
        except Exception as e:
            print(f"  FAIL: {name} ({t}) -> {e}"); ok = False
    print("  all 7 transforms build" if ok else "  BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

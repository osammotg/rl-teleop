# TODOS

Deferred work, with enough context that someone picking it up in three months
knows why it mattered. A TODO without reasoning is worse than no TODO: it creates
confidence that the idea was captured while the thinking behind it is gone.

---

## Stop telling people to run `uv run pytest`

**What.** Update the run instructions in test docstrings and docs to
`.venv/bin/python -m pytest ...` instead of `uv run pytest ...`.

**Why.** `uv run` re-syncs the environment from `uv.lock` before running
anything. Until 2026-07-29 that swapped a working cu128 torch for the cu126 build
PyPI ships, which has no kernels for the rig's sm_120 GPU (RTX 5090). Fourteen
packages, silently, in under a second.

**Status: the root cause is FIXED.** `pyproject.toml` now pins torch and
torchvision to an explicit `pytorch-cu128` index, and `uv lock` was regenerated.
Verified after the fix: `uv run python -c "import torch"` reports
`2.7.1+cu128`, a GPU matmul succeeds, and only 1 package is reinstalled instead
of 14. So `uv run` is safe again *on this machine's lockfile*.

**What is left.** Two loose ends:

1. `tests/runtime/test_control_server.py:10` (and likely other files) still say
   `Run: uv run pytest ...`. That instruction is now safe, but it was the vector,
   and it is worth deciding deliberately whether `uv run` is the blessed entry
   point or whether the venv python is. Pick one and make the docs agree.
2. The pin assumes no one on this project needs cu126. cu128 requires driver
   >= 525 and works on Ampere and Ada as well as Blackwell, so this should be
   safe, but it has only been verified on the 5090 box. If a non-Blackwell
   machine ever fails to install, make the index choice conditional on the
   platform rather than reverting the pin -- reverting silently breaks the 5090
   again, and the symptom appears much later as "the policy is bad".

**Pros of finishing it.** Removes the last trace of a footgun that cost a live
session and manifests as a model-quality problem rather than an install problem.

**Cons.** Touching the lockfile affects everyone on the repo; the docs sweep is
tedious and low-drama.

**Depends on.** Confirming which GPUs other contributors actually run.

**Context.** The rig HANDOFF documents this as "footgun 5" and gives the manual
recovery command:
```
uv pip install --python .venv/bin/python --reinstall-package torch \
  --reinstall-package torchvision torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

---

## gstack tooling silently no-ops without `bun`

**What.** `bun` is not installed on this box, so any gstack helper that shells
out to it fails closed.

**Why it matters.** `~/.claude/skills/gstack/bin/gstack-review-log` validates its
input with `bun -e "JSON.parse(...)"`. With no `bun`, it rejects *every* payload
with "invalid JSON, skipping" -- including valid JSON. The review dashboard then
shows no reviews were ever run. The failure blames your input for a missing
dependency, which is the worst possible error message.

**Workaround in use.** Append the JSONL entry directly to
`~/.gstack/projects/<slug>/<branch>-reviews.jsonl`, which is exactly what the
script's line 18 does after validation.

**Fix options.** Install `bun` (the gstack setup script wants it anyway, with a
checksum-verified installer), or patch the validator to fall back to `python -m
json.tool` when `bun` is absent.

**Pros.** Review history stops silently vanishing, and `gstack/browse` becomes
usable (it needs a `bun` build too).

**Cons.** Installing a new runtime on the rig box for tooling that is not part of
the robot stack.

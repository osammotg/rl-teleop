# cv-camera-mapping

Persistent OpenCV camera role mapping for bimanual wrist cameras.

## Why this exists

Linux assigns volatile `/dev/videoN` paths. Innomaker U20CAM units often share duplicate serial `SN0001`, so **serial-only mapping cannot distinguish `camera_left` from `camera_right`**.

This package:

1. Enumerates capture-capable V4L2 nodes (filters metadata siblings).
2. Enrolls golden reference views per role.
3. Identifies roles using deterministic **AKAZE + BF matching + RANSAC fundamental matrix** when serials are ambiguous.

**USB path is port-stable, not camera-identity-stable** — replugging into another port changes `ID_PATH` but not which physical camera it is.

**Serial burning / firmware flashing is unsupported and rejected** — unofficial UVC descriptor writes are high risk.

**Runtime VLM/LLM assignment is excluded** — matching runs offline via this CLI only.

## v1 scope

Exactly two roles: `camera_left` and `camera_right`.

## Setup

```bash
cd lib/cv-camera-mapping
uv sync --extra dev
```

## Commands

```bash
# List capture-capable devices
uv run cv-camera-map enumerate --json

# Enroll references (explicit role→device required)
uv run cv-camera-map capture-reference \
  --roles camera_left,camera_right \
  --role-device camera_left=/dev/video0 \
  --role-device camera_right=/dev/video8 \
  --output .local/references

# Identify after replug
uv run cv-camera-map identify \
  --references .local/references \
  --json \
  --output .local/mapping.json

# Validate artifact
uv run cv-camera-map validate-mapping --mapping .local/mapping.json

# Optional hardware QA (skips if <2 cameras)
uv run cv-camera-map live-qa --output .test-output/live --evidence ../../.omo/evidence/cv-camera-mapping/live-qa.json

# Tests
uv run pytest
uv run ruff check .
```

## Main repo integration

Produce `mapping.json` offline, then in YAML (example):

```yaml
driver: OpenCVCamera
mapping_role: camera_left
mapping_artifact: /path/to/lib/cv-camera-mapping/.local/mapping.json
```

Existing `device_path: /dev/video-left` configs are unchanged.

Resolver precedence:

1. Explicit `device_path` when mapping fields omitted
2. Unique serial (only if unique among candidates)
3. USB path — port-stable only
4. Visual fingerprint for duplicate serials
5. Fail closed on ambiguity

## Failure reasons

`duplicate_serial`, `insufficient_candidates`, `too_many_candidates`, `insufficient_features`, `insufficient_geometric_inliers`, `score_below_threshold`, `score_margin_below_threshold`, `mapping_artifact_missing`, `reference_artifact_invalid`, `role_device_mapping_required`

## Troubleshooting (black mat / low texture)

Mask/downweight dark mat regions automatically. Prefer background and periphery features; avoid specular hotspots; keep a stable enrollment pose; add a small landmark if the scene is too uniform.

Visual fingerprinting **fails closed** when confidence or score margin is insufficient — it will not guess.

## Local artifacts (gitignored)

- `.local/` — enrollment references
- `.test-output/` — CLI test output

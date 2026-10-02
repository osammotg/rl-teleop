# Draft: Inference

## Requirements (confirmed)
- Add inference to this project.
- Investigate whether there already is a code path for inference.
- Original request mentioned `lerobot-record` / LR-compatible policies, but user clarified `.pt`/`.pts` artifacts are produced from fine-tuned NVIDIA Cosmos plus a custom IDM, so LeRobot may not need to be involved.
- Reuse the same dashboard as teleop.
- Reuse as many teleop code paths/configs as possible, especially arms configuration.

## Technical Decisions
- Launch shape: Plan should prioritize existing `rr-session <yaml>` policy deployment path and explain how to deploy a `.pts`/checkpoint through YAML; optional wrapper only if needed later.
- Recording output: no output required for inference rollouts; do not plan LeRobotDataset export or RR recording changes for initial scope.
- Test strategy: TDD adapter tests with fake LR policy first, then implementation and dry-run/sim QA.
- Revised adapter direction: likely a generic `PolicyAgent` adapter for PyTorch/Cosmos+custom IDM checkpoints, not a LeRobot-specific adapter.
- User accepted recommendation to drop LeRobot from v1 unless actual LeRobot checkpoints/dataset output are needed later.
- Plan default: Cosmos/IDM `.pt` inference through existing `rr-session` YAML; design the adapter with an explicit loader/mapping contract so a future LeRobot backend can be added without changing session/TUI/arms paths.
- Pending: exact checkpoint loading contract, model callable signature, observation/action schema, and whether an optional LeRobot-compatible shim remains in scope.

## Research Findings
- README describes a unified pipeline for teleoperation, data collection, and policy deployment, but marks “Test + verify policy deploy pipeline” as TODO.
- README quickstart uses `uv run rr-session <config.yaml>` for teleop sessions and shows the existing Rich TUI/dashboard.
- No OpenSpec or Spec Kit directories found.
- Detailed code-path and test/config exploration is in progress via background research agents.
- Existing policy/inference-shaped path: `robots_realtime/agents/agent.py` defines the base `PolicyAgent`; `robots_realtime/agents/policy_learning/diffusion_policy_agent.py` shows learned-policy agent patterns with `load_model` and `act`.
- Existing runtime bridge: `robots_realtime/runtime/agent_node.py` wraps agents on the ZMQ bus and supports `subscriber_driven` loop mode, which research identified as likely reusable for inference.
- Existing session/config path: `robots_realtime/runtime/config.py`, `robots_realtime/runtime/session.py`, and `robots_realtime/rr_session_cli.py` load YAML session configs and run the dashboard/TUI.
- Existing recording path: `robots_realtime/runtime/recording.py` has `McapWriter` and `AsyncMp4Writer`; LeRobot dataset writing/format mapping appears missing.
- Existing reusable arms/config area: `configs/yam/` contains current robot/leader/sim configs to preserve rather than duplicate.
- Gap: no dedicated `lerobot-record`/LR-compatible inference CLI was found; likely plan needs either a wrapper command or session mode that reuses `rr-session` internals.

## Open Questions
- What exactly is the `.pts` artifact: LeRobot `from_pretrained` checkpoint/repo, TorchScript, plain PyTorch state dict, or project-specific serialized policy?
- What observations/actions does the `.pts` policy expect: state vector order, image keys/sizes, task text, and action vector split/order for left/right arms?
- Exact checkpoint loading and callable signature remain discoverable only from the Cosmos/IDM artifact code/docs; plan should include a first task to formalize this contract with a fake-policy fixture before implementation.

## Scope Boundaries
- INCLUDE: inference planning, LR-compatible policy loading through `lerobot-record`, teleop dashboard reuse, teleop arms/config reuse.
- EXCLUDE: training new policies unless explicitly requested.

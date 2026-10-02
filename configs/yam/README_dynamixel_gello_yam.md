# Dynamixel GELLO YAM Teleop

This configuration drives the existing bimanual YAM followers from Dynamixel-backed GELLO leader arms.

## Hardware paths

- Left leader serial: `/dev/leader-left`
- Right leader serial: `/dev/leader-right`
- Left follower CAN: `can_follow_l`
- Right follower CAN: `can_follow_r`

The Dynamixel leaders are read-only in this integration. The agent does not enable torque, set modes, write goal positions, configure IDs, or write calibration/EEPROM registers.

## Configuration

Use:

```bash
configs/yam/yam_bimanual_dynamixel_gello_teleop.yaml
```

The config keeps the runtime topology compatible with the Feetech GELLO config:

- `gello_left` publishes `gello_left/joint_pos`
- `gello_right` publishes `gello_right/joint_pos`
- `yam_left` subscribes to `gello_left/joint_pos`
- `yam_right` subscribes to `gello_right/joint_pos`

## Validation commands

Verify direct leader reads:

```bash
uv run python -m robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent \
  --port /dev/leader-left \
  --motor-ids 1,2,3,4,5,6,7 \
  --discover \
  --samples 5 \
  --output .omo/evidence/task-5-left-read.json

uv run python -m robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent \
  --port /dev/leader-right \
  --motor-ids 8,9,10,11,12,13,14 \
  --discover \
  --samples 5 \
  --output .omo/evidence/task-5-right-read.json
```

Verify topic-shaped leader output without commanding followers:

```bash
./teleop.sh --dry-run-leaders
```

This writes `.omo/evidence/task-7-dry-run-topics.json` and should show finite length-7 vectors for both `gello_left/joint_pos` and `gello_right/joint_pos`.

Run live teleop:

```bash
./teleop.sh --no-tui
```

or:

```bash
uv run rr-session configs/yam/yam_bimanual_dynamixel_gello_teleop.yaml --no-tui
```

## Known failure signatures

- `FeetechMotorsBus motor check failed`: the old Feetech config is being used instead of the Dynamixel config.
- `Failed to open Dynamixel port /dev/leader-left` or `/dev/leader-right`: leader symlinks are missing or permissions are wrong.
- `Dynamixel read failed ... comm_result=-3001`: a leader motor did not return a status packet.
- `Dynamixel read failed ... comm_result=-3002`: transient corrupted read; the reader retries these.
- `fail to communicate with the motor ... can_follow_l` or `can_follow_r`: follower CAN/hardware issue, not a Dynamixel leader issue.

## Evidence artifacts

- Dependency import: `.omo/evidence/task-2-dynamixel-import.txt`
- Feetech import regression: `.omo/evidence/task-2-feetech-regression.txt`
- Unit tests: `.omo/evidence/task-4-agent-tests.txt`
- No-write scan: `.omo/evidence/task-4-no-write-scan.txt`
- Direct reads: `.omo/evidence/task-5-left-read.json`, `.omo/evidence/task-5-right-read.json`
- Dry-run topics: `.omo/evidence/task-7-dry-run-topics.json`
- Live smoke attempt: `.omo/evidence/task-8-live-smoke.log`

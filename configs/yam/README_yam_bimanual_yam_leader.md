# YAM Bimanual YAM Leader Teleoperation Setup

This configuration enables bimanual teleoperation using YAM leader arms (with teaching handles) to control YAM follower arms.

## Hardware Setup

### Required Hardware
- 2x YAM Leader Arms (with teaching handles)
- 2x YAM Follower Arms
- 4x CAN interfaces (CANable or similar)

### CAN Interface Configuration

Based on your system, the CAN interfaces should be mapped as follows:

```bash
/sys/class/net/can_follow_l   -> Left follower arm
/sys/class/net/can_follow_r   -> Right follower arm
/sys/class/net/can_tleader_l  -> Left leader arm (teaching handle)
/sys/class/net/can_tleader_r  -> Right leader arm (teaching handle)
```

Verify your CAN interfaces are up and running:

```bash
ls -l /sys/class/net/can*
ip a | grep can
```

All interfaces should show `state UP`.

## Configuration Files

### Robot Configs

**Leader Arms:**
- `robot_configs/yam/leader_left.yaml` - Left YAM leader with teaching handle
- `robot_configs/yam/leader_right.yaml` - Right YAM leader with teaching handle

**Follower Arms:**
- `robot_configs/yam/left.yaml` - Left YAM follower (existing)
- `robot_configs/yam/right.yaml` - Right YAM follower (existing)

### System Config

**Main config:**
- `configs/yam/yam_bimanual_yam_leader.yaml` - Full bimanual system setup

## How It Works

### Architecture

```
┌─────────────────┐      ┌─────────────────┐
│ YAM Leader Left │─────▶│ YAM Follower L  │
│  (can_tleader_l)│      │  (can_follow_l) │
└─────────────────┘      └─────────────────┘
   Teaching Handle          Gripper Arm

┌─────────────────┐      ┌─────────────────┐
│ YAM Leader Right│─────▶│ YAM Follower R  │
│  (can_tleader_r)│      │  (can_follow_r) │
└─────────────────┘      └─────────────────┘
   Teaching Handle          Gripper Arm
```

### Teaching Handle Controls

Each YAM leader arm has a teaching handle with:
- **Trigger**: Controls the follower gripper (pull to close, release to open)
- **Top Button**: Press to enable/disable arm synchronization
- **Second Button**: User-programmable (currently unused)

### Agent Implementation

The `YamLeaderAgent` class:
1. Reads joint positions from the leader arm via i2rt's `MotorChainRobot`
2. Reads the teaching handle encoder for:
   - Trigger position → gripper command
   - Button states → enable/disable control
3. Publishes joint positions to the follower robot node
4. Applies coordinate transformations (signs, offsets)

## Usage

### 1. Setup CAN Interfaces

Before launching the system, ensure all CAN interfaces are up and running:

```bash
# Use the provided script
./scripts/setup_can_yam_bimanual_leader.sh
```

Or manually:
```bash
sudo ip link set can_tleader_l type can bitrate 1000000 && sudo ip link set can_tleader_l up
sudo ip link set can_tleader_r type can bitrate 1000000 && sudo ip link set can_tleader_r up
sudo ip link set can_follow_l type can bitrate 1000000 && sudo ip link set can_follow_l up
sudo ip link set can_follow_r type can bitrate 1000000 && sudo ip link set can_follow_r up
```

### 2. Launch the System

```bash
python -m robots_realtime.main configs/yam/yam_bimanual_yam_leader.yaml
```

### Operation

1. **Initial State**: Both leader arms start DISABLED
2. **Enable Control**: Press the top button on either teaching handle to enable that arm
3. **Move the Arms**: Physically move the leader arm; the follower will mirror
4. **Control Gripper**: Pull the trigger to close the gripper, release to open
5. **Disable Control**: Press the top button again to disable synchronization

### Safety Notes

- Always ensure you have a clear workspace before enabling
- The enable/disable button prevents accidental follower movement
- Each arm can be enabled/disabled independently
- Emergency stop: Press the button to disable at any time

## Troubleshooting

### CAN Interface Issues

**Problem**: CAN interfaces not found or "Network is down"
```bash
# Check if interfaces exist
ip link show can_tleader_l
ip link show can_tleader_r
ip link show can_follow_l
ip link show can_follow_r
```

**Solution**:

1. **Interfaces not found**: Ensure all CANable devices are plugged in and configured with persistent IDs. See i2rt documentation: `dependencies/i2rt/docs/set_persist_id_socket_can.md`

2. **Network is down**: The CAN interfaces need to be brought up. Run:
```bash
# Bring up all CAN interfaces
sudo ip link set can_tleader_l type can bitrate 1000000
sudo ip link set can_tleader_l up

sudo ip link set can_tleader_r type can bitrate 1000000
sudo ip link set can_tleader_r up

sudo ip link set can_follow_l type can bitrate 1000000
sudo ip link set can_follow_l up

sudo ip link set can_follow_r type can bitrate 1000000
sudo ip link set can_follow_r up
```

3. **Verify all are UP**:
```bash
ip a | grep can
# All interfaces should show "state UP"
```

**Tip**: Create a script to bring up all CAN interfaces automatically:
```bash
#!/bin/bash
# save as setup_can.sh
for iface in can_tleader_l can_tleader_r can_follow_l can_follow_r; do
    sudo ip link set $iface type can bitrate 1000000
    sudo ip link set $iface up
done
echo "All CAN interfaces are UP"
```

### No Encoder Data Warning

**Problem**: Log shows "No encoder data from teaching handle"

**Possible causes**:
1. Teaching handle encoder not configured in robot config
2. Encoder ID mismatch (should be 0x50E)
3. CAN bus communication issue

**Solution**:
- Verify `get_same_bus_device_driver` is set in the robot config
- Test encoder reading: `python dependencies/i2rt/scripts/read_encoder.py --channel can_tleader_l`

### Arms Not Responding

**Problem**: Moving leader arm doesn't move follower

**Checklist**:
1. Press the top button on the teaching handle to enable
2. Check log for "YamLeaderAgent: ENABLED" message
3. Verify follower robot node is running (check logs)
4. Check CAN interface is UP: `ip link show can_follow_l`

### Coordinate Transformation Issues

**Problem**: Follower moves in wrong direction or has offset

**Solution**: Adjust `joint_signs` and `joint_offsets` in the agent config:
```yaml
agent_kwargs:
  joint_signs: [1, 1, 1, 1, 1, 1]  # Use -1 to flip a joint
  joint_offsets: [0, 0, 0, 0, 0, 0]  # Radians to add
```

## Bilateral Control (Haptic Feedback)

Bilateral control is **enabled by default** with `bilateral_kp: 0.3`. This provides haptic feedback to the operator, allowing you to feel:
- Contact forces when the follower touches objects
- Collisions and obstacles
- Weight and inertia of manipulated objects

### How It Works

The follower's joint positions are read from observations and commanded back to the leader motors as position targets. The motor controller generates a restoring torque proportional to:

```
τ ∝ Kp × (q_follower − q_leader)
```

When the follower encounters resistance, the position error increases, creating a force you feel on the leader arm.

### Tuning Bilateral Stiffness

Adjust `bilateral_kp` in the config (range: 0.0 to 1.0):

```yaml
agent_kwargs:
  bilateral_kp: 0.3  # Default: moderate haptic feedback
```

- **0.0**: Disables bilateral control (pass-through teleoperation, no haptic feedback)
- **0.1-0.2**: Light haptic feedback (subtle)
- **0.3-0.4**: Moderate haptic feedback (recommended)
- **0.5-0.7**: Strong haptic feedback
- **0.8-1.0**: Very strong (may feel stiff or jerky)

### Warmup Period

The `warmup_steps: 5` parameter prevents jarring forces at startup:
- For the first 5 control steps, bilateral feedback is disabled
- This allows the follower to converge to the leader's initial position
- After warmup, bilateral control engages smoothly

Increase warmup_steps if you experience a jerk when enabling the arms.

## Customization

### Gripper Range

Adjust the gripper range in the agent config:

```yaml
agent_kwargs:
  gripper_open_pos: 0.0    # Encoder value for fully open
  gripper_close_pos: 1.0   # Encoder value for fully closed
```

### Enable Button

Change which button enables/disables:

```yaml
agent_kwargs:
  enable_button_index: 0  # 0 for top button, 1 for second button
```

### Disable Gripper

To use only the 6-DOF arm without gripper:

```yaml
agent_kwargs:
  include_gripper: false
```

## References

- **i2rt YAM Leader Documentation**: `dependencies/i2rt/docs/products/yam-leader.md`
- **i2rt Bimanual Setup**: `dependencies/i2rt/examples/bimanual_lead_follower/README.md`
- **Teaching Handle Specs**: See encoder reading in i2rt docs
- **Existing Gello Config**: `configs/yam/yam_bimanual_gello_teleop.yaml` (similar structure)

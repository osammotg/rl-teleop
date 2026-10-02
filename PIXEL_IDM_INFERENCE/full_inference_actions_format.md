# Full Inference Action Output

`notebooks/full_inference_check.ipynb` writes:

```text
artifacts/full/full_inference_actions.npz
```

The file is a compressed NumPy archive containing model outputs for the sampled
windows used by the notebook's full-run inference check.

## Arrays

```python
import numpy as np

data = np.load("artifacts/full/full_inference_actions.npz")

sample_indices = data["sample_indices"]      # int64, shape (N,)
predicted = data["predicted_actions"]        # float32/float64, shape (N, T, 14)
target = data["target_actions"]              # float32, shape (N, T, 14)
```

For the current notebook settings:

```text
N = 10 sampled windows
T = 8 frames per window
```

The axes of `predicted_actions` and `target_actions` are:

```text
[sample window, frame in window, action dimension]
```

## Action Dimensions

Each 14-value action row is an absolute robot state:

```text
0   left/j0
1   left/j1
2   left/j2
3   left/j3
4   left/j4
5   left/j5
6   left/gripper
7   right/j0
8   right/j1
9   right/j2
10  right/j3
11  right/j4
12  right/j5
13  right/gripper
```

These are absolute per-frame joint/gripper positions, not deltas, velocities, or
relative commands. The training targets come from `/yam_left/joint_state` and
`/yam_right/joint_state`, linearly interpolated onto camera frame timestamps.

`predicted_actions` has already been de-normalized into the same space as
`target_actions`.

## Using For Replay

Choose one sampled window and step through its predicted absolute states:

```python
import numpy as np

data = np.load("artifacts/full/full_inference_actions.npz")
actions = data["predicted_actions"][0]  # shape (8, 14)

for state in actions:
    left_joints = state[:6]
    left_gripper = state[6]
    right_joints = state[7:13]
    right_gripper = state[13]

    # Send these as absolute setpoints to the corresponding robot interfaces.
```

The notebook inference is evaluated at the dataset `eval_fps` from the checkpoint
config, currently 10 Hz. If your robot control loop runs faster than this, hold
or interpolate between consecutive absolute states in your replay layer.

Use `target_actions` only for comparison against held-out real YAM states. For
generated video replay, use `predicted_actions`.

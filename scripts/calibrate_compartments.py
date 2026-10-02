"""One-time box calibration → recordings/compartments.json.

The offline labeler needs to know WHERE the 7 box compartments are in the robot base
frame so it can say which compartment each release landed in. This captures that by
having you jog the gripper to the CENTER of each compartment; it reads the arm's
end-effector XY via forward-kinematics and writes axis-aligned rects.

Run it while a teleop session (record_kitting.sh / rr-session) is live so yam_left is
publishing on the bus:

    uv run python scripts/calibrate_compartments.py --arm left        # 7 compartments

For each compartment it prompts you to move the gripper over its center and press
Enter. Rect half-size is derived from the spacing between the captured centers so the
compartments tile without gaps. Writes recordings/compartments.json — record_kitting's
watcher then copies it into every episode automatically.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.fk import ForwardKinematics


def _ee_xy(fk, sub, topic):
    env = sub.get_latest(topic)
    if not env:
        return None
    data = env.get("data") or {}
    jp = data.get("joint_pos")            # numpy array on the bus — avoid `or` (ambiguous truth)
    if jp is None:
        jp = data.get("position")
    if jp is None:
        return None
    arm = np.asarray(jp, float)[: C.N_ARM_JOINTS][None, :]
    xyz = fk.ee_positions(arm)[0]
    return float(xyz[0]), float(xyz[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate the 7 box compartments (robot-assisted).")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--n", type=int, default=7, help="number of compartments")
    ap.add_argument("--urdf", default="urdf/yam.urdf")
    ap.add_argument("--out", default="recordings/compartments.json")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
    from robots_realtime.runtime.transport.subscriber import Subscriber

    topic = f"yam_{args.arm}/joint_state"
    sub = Subscriber([topic], host=args.host, port=DEFAULT_SUB_PORT)
    fk = ForwardKinematics(args.urdf)

    if _ee_xy(fk, sub, topic) is None:
        print(f"No data on {topic}. Start the teleop session first (record_kitting.sh).")
        return 1

    print(f"Calibrating {args.n} compartments. Jog the gripper to each CENTER, then press Enter.")
    centers = []
    for cid in range(1, args.n + 1):
        input(f"  compartment {cid}: move gripper to its center, then press Enter... ")
        xy = _ee_xy(fk, sub, topic)
        if xy is None:
            print("    no reading — is the arm live? aborting."); return 1
        centers.append(xy)
        print(f"    captured c{cid} at x={xy[0]:.3f} y={xy[1]:.3f}")

    centers = np.asarray(centers)
    # half-size = ~0.45 * nearest-neighbor spacing → rects tile with a small gap
    def nn(i):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d[i] = np.inf
        return d.min()
    half = 0.45 * float(np.median([nn(i) for i in range(len(centers))]))

    comps = [{"id": i + 1,
              "x_min": round(cx - half, 4), "x_max": round(cx + half, 4),
              "y_min": round(cy - half, 4), "y_max": round(cy + half, 4)}
             for i, (cx, cy) in enumerate(centers)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # load_compartments reads data["compartments"], so wrap the list (not a bare list).
    out.write_text(json.dumps({"frame": "robot_base", "compartments": comps}, indent=2))
    print(f"\nwrote {out}  (half-size {half*1000:.0f} mm). New recordings will classify places.")
    print("Backfill existing episodes:  for d in recordings/*/episode_*; do cp "
          f"{out} \"$d/\"; uv run python -m robots_realtime.labeling.label_episode \"$d\" --arm {args.arm}; done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

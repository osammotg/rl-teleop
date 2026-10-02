#!/usr/bin/env python3
"""Label ONE episode with the correct physical gripper refs and emit qa.json.

Run in the project env (uv run). Fixes the live-labeler mislabel (it omits
gripper refs) and records recording-health so the review can tell a dead
gripper channel apart from a genuine operator abort.

  uv run python qa_label.py recordings/<date>/episode_xxx
"""
import json, os, sys
import numpy as np
from robots_realtime.labeling.mcap_io import read_positions
from robots_realtime.labeling.label_episode import label_episode_dir
from robots_realtime.labeling import constants as C

# Physical gripper limits for this YAM rig (raw joint units): open ~1.0, closed ~0.0.
OPEN_REF, CLOSED_REF = 1.0, 0.0

def run(d, arm="left"):
    qa = {"episode": os.path.basename(d), "arm": arm}
    mcap = os.path.join(d, f"yam_{arm}.mcap")
    try:
        t, pos = read_positions(mcap, f"yam_{arm}")
        g = pos[:, C.GRIPPER_JOINT_INDEX]
        norm = np.clip((g - CLOSED_REF) / (OPEN_REF - CLOSED_REF), 0.0, 1.0)
        qa["mcap_ok"] = True
        qa["n_samples"] = int(t.size)
        qa["gripper_min_norm"] = round(float(norm.min()), 4)
        # Did the gripper ever close enough to attempt a grasp? (< close-enter threshold)
        qa["gripper_actuated"] = bool(norm.min() < C.GRIPPER_CLOSE_ENTER)
    except Exception as e:
        qa["mcap_ok"] = False
        qa["error"] = str(e)[:200]

    if qa.get("mcap_ok"):
        try:
            ann = label_episode_dir(d, arm=arm,
                                    gripper_open_ref=OPEN_REF, gripper_closed_ref=CLOSED_REF,
                                    min_transport_m=C.MIN_TRANSPORT_M, geometric_targets=True)
            qa["outcome"] = ann.episode_meta.outcome
            qa["grasps"] = len(ann.grasp_attempts)
            qa["places"] = len(ann.place_events)
        except Exception as e:
            qa["label_error"] = str(e)[:200]

    json.dump(qa, open(os.path.join(d, "qa.json"), "w"), indent=2)
    return qa

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: qa_label.py <episode_dir> [arm]"); raise SystemExit(2)
    arm = sys.argv[2] if len(sys.argv) > 2 else "left"
    q = run(sys.argv[1], arm)
    health = ("CORRUPT" if not q.get("mcap_ok")
              else "DEAD-GRIPPER" if q.get("gripper_actuated") is False
              else "ok")
    print(f"{q['episode']}: health={health} grasps={q.get('grasps')} outcome={q.get('outcome')}")

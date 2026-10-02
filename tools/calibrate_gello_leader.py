#!/usr/bin/env python3
"""Read a GELLO leader and solve its joint_offsets_ticks. NEVER commands a follower.

WHY THIS EXISTS. `configs/yam/yam_right_kitting_teleop.yaml` carries right-leader
offsets copied verbatim from `yam_bimanual_dynamixel_gello_teleop.yaml` with a
comment saying they were not re-derived and to "re-run the leader calibration"
if the leader has been re-assembled. There was no such tool in the repo.

SAFETY. This opens the Dynamixel serial port only. It never enables torque, never
writes EEPROM, and never opens a CAN channel, so the follower cannot move no
matter what this prints. Run it with the teleop session STOPPED — the leader
serial port cannot be held by two readers at once.

THE MAPPING, from dynamixel_gello_leader_agent._map_arm_ticks (the one authority;
if that function changes, this file is wrong until updated to match):

    calibrated = (raw_ticks + offsets_ticks) * scales
    rad        = wrap_pi(calibrated * 2*pi/4096 - pi)
    reported   = signs * rad + offsets_rad

Solving it for offsets_ticks at a pose whose true angles you know:

    offsets_ticks = ((q_true - offsets_rad)/signs + pi) / (2*pi/4096) / scales - raw_ticks

`scales` is +/-1 in every config on this rig, so dividing by it is exact. The
result is wrapped into (-4096, 4096] because ticks are modulo one revolution and
an unwrapped offset would read correctly yet look absurd in the YAML.

USAGE
    # watch what the leader reports under the CURRENT config (no changes made)
    tools/calibrate_gello_leader.py --config configs/yam/yam_right_kitting_teleop.yaml --watch

    # solve: hold the leader at the reference pose, then
    tools/calibrate_gello_leader.py --config configs/yam/yam_right_kitting_teleop.yaml --solve
    # ...defaults to a zero reference pose; pass --target-deg for anything else.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

NUM_ARM_JOINTS = 6
TICKS_PER_REV = 4096
TICKS_TO_RAD = (2.0 * np.pi) / TICKS_PER_REV


def load_leader_kwargs(config_path: Path, agent_name: str | None) -> dict:
    """Pull the leader AgentNode's kwargs out of a session config.

    Reading the real config rather than taking numbers on the command line is
    deliberate: the whole failure mode being fixed here is a calibration that
    drifted from the file that actually runs.
    """
    cfg = yaml.safe_load(config_path.read_text())
    agents = [n for n in cfg.get("nodes", []) if n.get("type") == "AgentNode"]
    if not agents:
        raise SystemExit(f"no AgentNode in {config_path}")
    if agent_name:
        agents = [n for n in agents if n.get("name") == agent_name]
        if not agents:
            raise SystemExit(f"no AgentNode named {agent_name!r} in {config_path}")
    if len(agents) > 1:
        names = ", ".join(str(n.get("name")) for n in agents)
        raise SystemExit(f"{config_path} has several leaders ({names}); pass --agent")
    node = agents[0]
    kw = dict(node.get("agent_kwargs", {}))
    kw["_node_name"] = node.get("name")
    return kw


class LeaderMoved(RuntimeError):
    """The leader was not stationary while being sampled.

    THE WHOLE POINT. A solve is only meaningful if the arm is physically held at
    the reference pose for the duration of the read. On 2026-08-10 two reads
    thirty seconds apart differed by 12 deg on joint 0 and 8 deg on joint 5, and
    both produced confident-looking offsets that would have baked the operator's
    hand motion into the config permanently. A drifting read must fail, not
    average: the mean of a moving arm is a pose it never occupied.
    """


def read_raw_ticks(port: str, motor_ids: list[int], baudrate: int, samples: int,
                   max_drift_deg: float | None = None) -> np.ndarray:
    """Median of `samples` sync reads, in raw encoder ticks.

    Median rather than mean: a corrupted Dynamixel read (comm_result -3002) is
    retried inside the reader, but a single survivor outlier would drag a mean
    and silently bias every offset derived from it.

    With `max_drift_deg`, raises LeaderMoved if any arm joint's peak-to-peak
    spread across the samples exceeds it.
    """
    from robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent import (
        _DynamixelPositionReader,
    )

    reader = _DynamixelPositionReader(
        port=port, baudrate=baudrate, motor_ids=motor_ids,
        protocol_version=2.0, present_position_addr=132,
        present_position_len=4, num_read_retries=3,
    )
    try:
        rows = []
        for _ in range(samples):
            rows.append(np.asarray(reader.get_positions(), dtype=np.float64))
            time.sleep(0.02)
        stack = np.vstack(rows)
        if max_drift_deg is not None:
            ptp_ticks = stack[:, :NUM_ARM_JOINTS].ptp(axis=0)
            ptp_deg = np.degrees(ptp_ticks * TICKS_TO_RAD)
            if float(ptp_deg.max()) > max_drift_deg:
                worst = int(np.argmax(ptp_deg))
                raise LeaderMoved(
                    f"leader moved while sampling: joint {worst} swept "
                    f"{ptp_deg[worst]:.2f} deg over {samples} reads "
                    f"(limit {max_drift_deg:.2f}). Per-joint spread (deg): "
                    "[" + " ".join(f"{x:.2f}" for x in ptp_deg) + "]. "
                    "Hold the arm still at the reference pose and re-run."
                )
        return np.median(stack, axis=0)
    finally:
        reader.close()


def ticks_to_reported(raw: np.ndarray, offsets_ticks, scales, signs, offsets_rad) -> np.ndarray:
    """Exactly what the agent would publish for these ticks. Mirrors _map_arm_ticks."""
    calibrated = (raw[:NUM_ARM_JOINTS] + np.asarray(offsets_ticks, float)) * np.asarray(scales, float)
    rad = calibrated * TICKS_TO_RAD - np.pi
    rad = (rad + np.pi) % (2.0 * np.pi) - np.pi
    return np.asarray(signs, float) * rad + np.asarray(offsets_rad, float)


def solve_offsets(raw: np.ndarray, q_true: np.ndarray, scales, signs, offsets_rad) -> np.ndarray:
    """Offsets that make the leader report `q_true` at this raw pose."""
    scales = np.asarray(scales, float)
    signs = np.asarray(signs, float)
    rad = (np.asarray(q_true, float) - np.asarray(offsets_rad, float)) / signs
    calibrated = (rad + np.pi) / TICKS_TO_RAD
    off = calibrated / scales - raw[:NUM_ARM_JOINTS]
    # Ticks are modulo one revolution; keep the printed number human-sized.
    return (off + TICKS_PER_REV / 2.0) % TICKS_PER_REV - TICKS_PER_REV / 2.0


def fmt_deg(v: np.ndarray) -> str:
    return "[" + " ".join(f"{x:7.2f}" for x in np.degrees(v)) + "]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--agent", default=None, help="AgentNode name when the config has several")
    ap.add_argument("--watch", action="store_true", help="stream what the leader reports")
    ap.add_argument("--solve", action="store_true", help="snapshot and emit new offsets")
    ap.add_argument("--target-deg", default=None,
                    help="reference pose, 6 comma-separated degrees (default: all zeros)")
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--max-drift-deg", type=float, default=1.0,
                    help="refuse to solve if any joint sweeps more than this while sampling")
    args = ap.parse_args(argv)

    if not (args.watch or args.solve):
        ap.error("pass --watch or --solve")

    kw = load_leader_kwargs(args.config, args.agent)
    port = kw["port"]
    motor_ids = list(kw["motor_ids"])
    baud = int(kw.get("baudrate", 1_000_000))
    offsets_ticks = kw.get("joint_offsets_ticks", [0.0] * NUM_ARM_JOINTS)
    scales = kw.get("joint_scales", [1.0] * NUM_ARM_JOINTS)
    signs = kw.get("joint_signs", [1.0] * NUM_ARM_JOINTS)
    offsets_rad = kw.get("joint_offsets_rad", [0.0] * NUM_ARM_JOINTS)

    print(f"leader   : {kw['_node_name']}  {port}  ids={motor_ids}")
    print(f"offsets  : {list(offsets_ticks)}")
    print(f"scales   : {list(scales)}   signs: {list(signs)}")
    print()

    if args.watch:
        print("  raw ticks                                    reported angles (deg)")
        try:
            while True:
                raw = read_raw_ticks(port, motor_ids, baud, 3)
                rep = ticks_to_reported(raw, offsets_ticks, scales, signs, offsets_rad)
                ticks_s = "[" + " ".join(f"{x:6.0f}" for x in raw[:NUM_ARM_JOINTS]) + "]"
                print(f"\r  {ticks_s}   {fmt_deg(rep)}", end="", flush=True)
                time.sleep(1.0 / max(args.hz, 0.1))
        except KeyboardInterrupt:
            print("\nstopped")
        return 0

    # --solve
    if args.target_deg:
        parts = [p for p in args.target_deg.replace(",", " ").split() if p]
        if len(parts) != NUM_ARM_JOINTS:
            ap.error(f"--target-deg needs {NUM_ARM_JOINTS} values, got {len(parts)}")
        q_true = np.radians(np.array([float(p) for p in parts]))
    else:
        q_true = np.zeros(NUM_ARM_JOINTS)

    try:
        raw = read_raw_ticks(port, motor_ids, baud, args.samples,
                             max_drift_deg=args.max_drift_deg)
    except LeaderMoved as exc:
        print(f"REFUSED: {exc}")
        return 2
    before = ticks_to_reported(raw, offsets_ticks, scales, signs, offsets_rad)
    new_off = solve_offsets(raw, q_true, scales, signs, offsets_rad)
    after = ticks_to_reported(raw, new_off, scales, signs, offsets_rad)

    print(f"raw ticks     : [{' '.join(f'{x:6.0f}' for x in raw[:NUM_ARM_JOINTS])}]")
    print(f"reference pose: {fmt_deg(q_true)}")
    print(f"reported BEFORE: {fmt_deg(before)}")
    print(f"reported AFTER : {fmt_deg(after)}   <- must equal the reference pose")
    print(f"change per joint (deg): {fmt_deg(before - q_true)}")
    print()

    # Round to whole ticks: the config carries integers, and a fractional offset
    # would not survive a round-trip through the YAML anyway. One tick is 0.088 deg.
    rounded = np.round(new_off).astype(int)
    resid = ticks_to_reported(raw, rounded, scales, signs, offsets_rad) - q_true
    print("      joint_offsets_ticks: [" + ", ".join(str(int(x)) for x in rounded) + "]")
    print(f"(rounding residual, deg: {fmt_deg(resid)})")
    print()
    print(f"Paste that line into {args.config} under the leader's agent_kwargs,")
    print("then restart the session and check the follower tracks before recording.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

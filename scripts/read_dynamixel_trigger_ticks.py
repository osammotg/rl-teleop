#!/usr/bin/env python3
"""Read raw Dynamixel GELLO trigger ticks for gripper calibration."""

from __future__ import annotations

import argparse
import time

import numpy as np

from robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent import _DynamixelPositionReader


_SIDE_DEFAULTS = {
    "left": ("/dev/leader-left", 7),
    "right": ("/dev/leader-right", 14),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print raw trigger ticks while you release and squeeze the GELLO trigger. "
            "Use the released value as gripper_open_ticks and the fully squeezed value "
            "as gripper_closed_ticks."
        )
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--port", help="Override serial port, e.g. /dev/leader-left")
    parser.add_argument("--motor-id", type=int, help="Override trigger motor id")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--period-s", type=float, default=0.05)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--num-read-retries", type=int, default=0)
    args = parser.parse_args()

    default_port, default_motor_id = _SIDE_DEFAULTS[args.side]
    port = args.port or default_port
    motor_id = args.motor_id if args.motor_id is not None else default_motor_id

    reader = _DynamixelPositionReader(
        port=port,
        baudrate=args.baudrate,
        motor_ids=[motor_id],
        protocol_version=2.0,
        present_position_addr=132,
        present_position_len=4,
        num_read_retries=args.num_read_retries,
    )

    values: list[int] = []
    errors = 0
    print(f"Reading {args.side} trigger on {port}, motor id {motor_id}")
    print("Release and fully squeeze the trigger a few times.")
    print("Press Ctrl-C to stop early.\n")
    print(f"{'sample':>6} {'tick':>8} {'min':>8} {'max':>8} {'range':>8}")

    try:
        for sample in range(1, args.samples + 1):
            try:
                tick = int(reader.get_positions()[0])
                values.append(tick)
                tick_min = min(values)
                tick_max = max(values)
                print(f"{sample:6d} {tick:8d} {tick_min:8d} {tick_max:8d} {tick_max - tick_min:8d}")
            except RuntimeError as exc:
                errors += 1
                print(f"{sample:6d} read error: {exc}")
            time.sleep(args.period_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.close()

    if not values:
        print("\nNo successful reads.")
        raise SystemExit(1)

    arr = np.asarray(values, dtype=np.int64)
    print("\nSummary")
    print(f"  successful_reads: {len(values)}")
    print(f"  errors:           {errors}")
    print(f"  min_tick:         {int(arr.min())}")
    print(f"  max_tick:         {int(arr.max())}")
    print(f"  median_tick:      {int(np.median(arr))}")
    print("\nConfig guidance")
    print("  The released trigger value is gripper_open_ticks.")
    print("  The fully squeezed trigger value is gripper_closed_ticks.")
    print("  If you moved through the full trigger travel during this run, the two")
    print("  endpoints should be close to min_tick and max_tick; assign them based on")
    print("  which endpoint you observed while released vs squeezed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refuse to start a session whose cameras are not physically there.

WHY. A camera node that cannot open its device dies inside ``setup()``. Until
2026-08-11 that death was invisible: the traceback went to a per-node log under
``/tmp/rr_logs_*/``, the session carried on without it, ``NodeStatus.alive``
was a constant True, and the TUI stayed green. One real instance ran for eleven
minutes with a defunct camera_top and three cockpit panels pointing at it.

The supervisor now reports that loudly at runtime — but the cheapest place to
catch a missing camera is before anything starts, while there is still a
terminal to print to and nobody has energised an arm.

This checks PRESENCE, not liveness. It opens nothing (so it is safe to run with
a session up, and it cannot steal a RealSense) and it is not a substitute for
``tools/check_streams.py``, which is the only honest test that frames are
actually being delivered.

    tools/preflight_cameras.py configs/yam/yam_right_kitting_teleop.yaml

Exit 0 = every configured camera is present. Exit 1 = something is missing.
Exit 2 = the config could not be read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def usb_speed(device_path: str) -> str | None:
    """Negotiated link speed in Mbit for a /dev/v4l/by-path node, if knowable."""
    try:
        real = os.path.realpath(device_path)
        node = os.path.basename(real)
        dev_link = os.path.realpath(f"/sys/class/video4linux/{node}/device")
        # walk up to the USB device directory that carries `speed`
        cur = dev_link
        for _ in range(6):
            speed_file = os.path.join(cur, "speed")
            if os.path.exists(speed_file):
                with open(speed_file, encoding="utf-8") as f:
                    return f.read().strip()
            cur = os.path.dirname(cur)
    except Exception:
        pass
    return None


def realsense_serials() -> tuple[list[str], str | None]:
    """Enumerate connected RealSense serials without opening any of them."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return [], "pyrealsense2 is not installed in this venv"
    try:
        return [
            d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()
        ], None
    except Exception as exc:                                       # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    try:
        import yaml
        cfg = yaml.safe_load(a.config.read_text(encoding="utf-8")) or {}
    except Exception as exc:                                       # noqa: BLE001
        print(f"preflight: cannot read {a.config}: {exc}", file=sys.stderr)
        return 2

    cams = [n for n in (cfg.get("nodes") or [])
            if isinstance(n, dict) and n.get("type") == "CameraNode"]
    if not cams:
        if not a.quiet:
            print("preflight: no CameraNode in this config — nothing to check")
        return 0

    serials, rs_err = realsense_serials()
    problems: list[str] = []
    lines: list[str] = []

    for cam in cams:
        name = cam.get("name", "?")
        path = cam.get("device_path")
        dev_id = cam.get("device_id")

        if path:
            if not os.path.exists(path):
                problems.append(
                    f"{name}: {path} does NOT exist.\n"
                    f"      /dev/videoN numbers change on every replug, which is why the "
                    f"config uses a by-path node — but the PORT changes too if the camera "
                    f"was moved. List what is actually there:\n"
                    f"        ls -l /dev/v4l/by-path/\n"
                    f"      and confirm which port is which arm by FRAME CONTENT, never by "
                    f"assuming the old mapping held (both wrist cameras report serial SN0001)."
                )
                continue
            speed = usb_speed(path)
            note = f"  link {speed}M" if speed else ""
            lines.append(f"  ok  {name:14s} {os.path.realpath(path)}{note}")
            # 1280x720@30 does not fit in USB 2.0's 480 Mbit; asking for it there
            # gives "Couldn't resolve requests" and the node dies at startup.
            res = cam.get("resolution") or []
            fps = cam.get("fps") or 0
            if speed == "480" and len(res) == 2 and int(res[0]) >= 1280 and int(fps) > 15:
                problems.append(
                    f"{name}: asking for {res[0]}x{res[1]}@{fps} on a USB 2.0 link (480M). "
                    f"That profile does not exist at this speed; the node will die at "
                    f"startup with 'Couldn't resolve requests'. Drop fps to 15, or give the "
                    f"camera a USB 3 path."
                )

        elif dev_id:
            if rs_err:
                problems.append(f"{name}: cannot enumerate RealSense devices — {rs_err}")
            elif str(dev_id) not in serials:
                problems.append(
                    f"{name}: RealSense serial {dev_id} is NOT connected.\n"
                    f"      Connected serials: {serials or 'NONE'}\n"
                    f"      If lsusb and rs-enumerate-devices both show nothing and a replug "
                    f"produces no dmesg event at all, the USB CONTROLLER is dead, not the "
                    f"camera — see CLAUDE.md for the one command that fixes it (needs a real "
                    f"terminal for sudo)."
                )
            else:
                lines.append(f"  ok  {name:14s} RealSense {dev_id}")
        else:
            lines.append(f"  --  {name:14s} no device_path or device_id (synthetic?)")

    if not a.quiet:
        for line in lines:
            print(line)

    if problems:
        print("\n  ✗ CAMERA PREFLIGHT FAILED\n", file=sys.stderr)
        for p in problems:
            print(f"    · {p}\n", file=sys.stderr)
        print("    Nothing has been started. Fix the above, then re-run.\n", file=sys.stderr)
        return 1

    if not a.quiet:
        print(f"  camera preflight ok — {len(cams)} camera(s) present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

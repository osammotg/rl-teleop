#!/usr/bin/env python3
"""Opt-in USB recovery helper for Intel RealSense cameras.

This script only unbinds/rebinds the USB device when --execute is passed.
By default it prints the recovery path as a dry run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SYSFS_USB_DEVICES = Path("/sys/bus/usb/devices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Last-resort USB unbind/bind recovery helper for RealSense cameras."
    )
    parser.add_argument("serial", help="USB serial number of the RealSense camera")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write to sysfs driver unbind/bind files. Default is dry-run.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def find_usb_device(serial: str) -> Path:
    matches: list[Path] = []
    for device in sorted(SYSFS_USB_DEVICES.iterdir()):
        if read_text(device / "serial") == serial:
            matches.append(device)

    if not matches:
        raise RuntimeError(
            f"No USB device with serial {serial!r} was found under {SYSFS_USB_DEVICES}.\n"
            "Check the serial number and confirm the camera is visible with lsusb."
        )
    if len(matches) > 1:
        found = ", ".join(str(path.name) for path in matches)
        raise RuntimeError(f"Serial {serial!r} matched multiple USB devices: {found}")
    return matches[0]


def driver_dir(device: Path) -> Path:
    driver = device / "driver"
    if not driver.exists():
        raise RuntimeError(
            f"USB device {device.name} has no bound driver. Nothing to unbind/bind."
        )
    try:
        return driver.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Unable to resolve driver for USB device {device.name}: {exc}") from exc


def show_device(device: Path, driver: Path) -> None:
    busnum = read_text(device / "busnum") or "unknown"
    devnum = read_text(device / "devnum") or "unknown"
    product = read_text(device / "product") or "unknown product"
    manufacturer = read_text(device / "manufacturer") or "unknown manufacturer"

    print(f"USB device: {device.name}")
    print(f"Serial: {read_text(device / 'serial') or 'unknown'}")
    print(f"Bus/device: {busnum}/{devnum}")
    print(f"Driver: {driver.name}")
    print(f"Product: {manufacturer} {product}")


def write_sysfs(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="utf-8")
    except PermissionError as exc:
        raise RuntimeError(
            f"Permission denied writing {path}.\n"
            "Run this script with sufficient privileges, for example:\n"
            f"  sudo {sys.executable} {Path(__file__).resolve()} <serial> --execute\n"
            "Dry-run mode is available without privileges by omitting --execute."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed writing {path}: {exc}") from exc


def check_execute_allowed(unbind: Path, bind: Path) -> None:
    missing = [str(path) for path in (unbind, bind) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Driver recovery files are missing: " + ", ".join(missing)
        )

    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise RuntimeError(
            "--execute requires privileges to write USB driver sysfs files.\n"
            "Re-run with sudo, for example:\n"
            f"  sudo {sys.executable} {Path(__file__).resolve()} <serial> --execute\n"
            "No action was taken."
        )


def recover(device: Path, driver: Path, execute: bool) -> None:
    unbind = driver / "unbind"
    bind = driver / "bind"

    print("Recovery path:")
    print(f"  write {device.name!r} to {unbind}")
    print(f"  write {device.name!r} to {bind}")

    if not execute:
        print("Dry-run only. Pass --execute to perform the unbind/bind reset.")
        return

    check_execute_allowed(unbind, bind)
    print("Executing USB unbind/bind reset...")
    write_sysfs(unbind, device.name)
    time.sleep(1.0)
    write_sysfs(bind, device.name)
    print("Done.")


def main() -> int:
    args = parse_args()
    try:
        device = find_usb_device(args.serial)
        driver = driver_dir(device)
        show_device(device, driver)
        recover(device, driver, args.execute)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

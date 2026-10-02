#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDEV_DIR="/etc/udev/rules.d"

sudo cp "$SCRIPT_DIR/90-yam-can.rules" "$UDEV_DIR/90-yam-can.rules"
sudo cp "$SCRIPT_DIR/99-yam-ft232h-leaders.rules" "$UDEV_DIR/99-yam-ft232h-leaders.rules"

sudo udevadm control --reload-rules
sudo systemctl restart systemd-udevd
sudo udevadm trigger

echo "Installed YAM udev mapping rules."
echo "Replug CAN/FT232H devices if names do not update immediately."

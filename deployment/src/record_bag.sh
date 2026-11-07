#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 BAG_NAME (start camera and optional teleop separately)" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/../topomaps/bags"
exec ros2 bag record /usb_cam/image_raw -o "$script_dir/../topomaps/bags/$1"

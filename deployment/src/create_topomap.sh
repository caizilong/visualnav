#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MAP_NAME ROS2_BAG_DIRECTORY [create_topomap.py arguments]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
map_name="$1"
bag_path="$2"
shift 2
python3 "$script_dir/create_topomap.py" --dir "$map_name" "$@" &
map_pid=$!
cleanup() { kill -INT "$map_pid" 2>/dev/null || true; wait "$map_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
# Let the subscriber join DDS discovery before replaying the first image.
sleep 2
ros2 bag play "$bag_path" --rate 1.5
wait "$map_pid"

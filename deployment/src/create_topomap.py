"""Save a ROS 2 RGB image stream as consecutively numbered topomap nodes."""

import argparse
from pathlib import Path
import shutil
import time

from sensor_msgs.msg import Image, Joy
from deployment_config import DEPLOYMENT_ROOT
from observation_buffer import ObservationBuffer
from ros2_runtime import ROS2Runtime, application_args


def main(args):
    if args.dt <= 0:
        raise ValueError("dt must be positive")
    directory = DEPLOYMENT_ROOT / "topomaps" / "images" / args.dir
    if directory.exists() and any(directory.iterdir()):
        if not args.overwrite:
            raise ValueError(f"{directory} is not empty; use --overwrite to replace its contents")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    observations = ObservationBuffer(0, stale_timeout_s=max(2 * args.dt, 2.))
    with ROS2Runtime("create_topomap") as ros:
        ros.subscribe(Image, "/usb_cam/image_raw", observations.callback, image=True)
        ros.subscribe(Joy, "/joy", lambda msg: ros.stopped.set() if msg.buttons and msg.buttons[0] else None)
        previous, count, next_sample = None, 0, 0.
        while ros.ok():
            frame_id, frames = observations.snapshot()
            now = time.monotonic()
            if frames is not None and frame_id != previous and now >= next_sample:
                frames[-1].save(directory / f"{count}.png")
                previous, count, next_sample = frame_id, count + 1, now + args.dt
            if count and not observations.fresh():
                break
            ros.sleep(.02)
    print(f"Saved {count} nodes to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", "-d", default="topomap")
    parser.add_argument("--dt", "-t", type=float, default=1.)
    parser.add_argument("--overwrite", action="store_true")
    try:
        main(parser.parse_args(application_args()))
    except KeyboardInterrupt:
        pass

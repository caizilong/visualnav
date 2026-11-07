"""Start a ROS 2 policy and its controller with matching command semantics."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
from deployment_config import model_paths
from ros2_runtime import application_args


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("entry", choices=("navigate", "explore"))
    parser.add_argument("--model", "-m", default="nomad")
    args, _ = parser.parse_known_args(application_args())
    policy_args = sys.argv[2:]
    if "--help" in policy_args or "-h" in policy_args:
        return subprocess.call([sys.executable, str(Path(__file__).with_name(args.entry + ".py")), *policy_args])
    paths = model_paths(args.model)
    config = yaml.safe_load(paths["config_path"].read_text())
    mode = "metric" if "action_scale_m" in config else "legacy"
    children = []
    # SIGTERM gets the same cleanup as Ctrl-C, including a final zero velocity.
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        children.append(subprocess.Popen([sys.executable, str(Path(__file__).with_name("pd_controller.py")),
                                          "--control-mode", mode]))
        children.append(subprocess.Popen([sys.executable, str(Path(__file__).with_name(args.entry + ".py")),
                                          *policy_args]))
        while all(child.poll() is None for child in children):
            time.sleep(.1)
        return next(child.returncode for child in children if child.poll() is not None)
    except KeyboardInterrupt:
        return 130
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())

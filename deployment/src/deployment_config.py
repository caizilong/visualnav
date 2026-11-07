"""Resolve deployment paths independently of the shell's working directory."""

import argparse
import math
from pathlib import Path

import yaml


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = DEPLOYMENT_ROOT / "config"


def robot_config():
    return yaml.safe_load((CONFIG_ROOT / "robot.yaml").read_text())


def model_paths(name):
    entries = yaml.safe_load((CONFIG_ROOT / "models.yaml").read_text())
    entry = dict(entries[name])
    # Existing models.yaml paths are relative to deployment/src.
    for key in ("config_path", "ckpt_path"):
        path = Path(entry[key])
        entry[key] = path if path.is_absolute() else DEPLOYMENT_ROOT / "src" / path
    return entry


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


"""Bind calibration to exact weights, label semantics and decision settings."""

import json
import math
from pathlib import Path

from vint_train.data.simulation import digest_json


def selector_spec(config):
    supplied = config.get("r3", {}).get("selector", {})
    defaults = {"max_forward_skip": 2, "use_distance": True, "close_threshold": 3.0, "radius": 4}
    return {key: supplied.get(key, default) for key, default in defaults.items()}


def load_calibration(path, model, config):
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != 1 or artifact.get("status") != "calibrated":
        raise ValueError("Calibration is missing or failed")
    expected = {"weights_sha256": model._weights_sha256,
                "weight_source": model._weights_origin,
                "label_definition_sha256": digest_json(config["r3"]["label_definition"]),
                "selector_spec": selector_spec(config)}
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Calibration mismatch: {key}")
    temperature, threshold = artifact["temperature"], artifact["threshold"]
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid calibration temperature/threshold")
    return artifact

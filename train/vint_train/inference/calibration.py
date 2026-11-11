"""Bind calibration to exact weights, label semantics and decision settings."""

import json
import math
from pathlib import Path

from vint_train.data.simulation import digest_json


def selector_spec(config):
    supplied = config.get("goal_relation", {}).get("selector", {})
    defaults = {"max_forward_skip": 2, "use_distance": True, "close_threshold": 3.0, "radius": 4}
    return {key: supplied.get(key, default) for key, default in defaults.items()}


def calibration_binding(model, config):
    """Bind shape-invisible SPATIAL_BOTTLENECK semantics as well as every model tensor."""
    return {"weights_sha256": model._weights_sha256,
            "weight_source": model._weights_origin,
            "model_config_sha256": digest_json(model._model_config),
            "label_definition_sha256": digest_json(config["goal_relation"]["label_definition"]),
            "selector_spec": selector_spec(config)}


def load_calibration(path, model, config):
    artifact = json.loads(Path(path).read_text())
    schema = artifact.get("schema_version")
    if schema not in (1, 2) or artifact.get("status") != "calibrated":
        raise ValueError("Calibration is missing or failed")
    expected = calibration_binding(model, config)
    if schema == 1:
        # Preserve old plain-GOAL_RELATION artifacts. Any SPATIAL_BOTTLENECK model must be recalibrated
        # with a binding that distinguishes auxiliary vs bottleneck semantics.
        if model._model_config.get("spatial_bottleneck", {}).get("enabled"):
            raise ValueError("SPATIAL_BOTTLENECK+GOAL_RELATION requires a schema 2 calibration; recalibrate this checkpoint")
        expected.pop("model_config_sha256")
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Calibration mismatch: {key}")
    temperature, threshold = artifact["temperature"], artifact["threshold"]
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid calibration temperature/threshold")
    return artifact

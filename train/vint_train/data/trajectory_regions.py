"""SPATIAL_BOTTLENECK target definition and train-only radial statistics; no image/depth inputs."""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .simulation import digest_json, load_trajectory, training_signature


TARGET = "future_waypoint_visit_distribution"


def metric_visitation_histograms(points, definition):
    """Metric local POSITIONS [B,H,2] -> per-window mass and point counts.

    Shared by train-only prior fitting and training labels. Never integrate
    these positions: diffusion deltas must be decoded before this interface.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite metric local waypoint positions [B,H,2]")
    sectors, radial = definition["num_sectors"], definition["num_radial_bins"]
    radius = np.linalg.norm(points, axis=-1)
    moving = radius > definition["stationary_epsilon_m"]
    step = 2 * math.pi / sectors
    angles = np.arctan2(points[..., 1], points[..., 0])
    sector = np.floor(np.remainder(angles + step / 2, 2 * math.pi) / step).astype(np.int64)
    sector = np.minimum(sector, sectors - 1)
    distance = np.searchsorted(definition["radial_boundaries_m"], radius, side="right")
    flat = sector * radial + distance
    counts = moving.sum(axis=1)
    histogram = np.zeros((len(points), sectors * radial), dtype=np.float64)
    np.add.at(histogram, (np.arange(len(points))[:, None], flat), moving.astype(np.float64))
    histogram /= np.maximum(counts, 1)[:, None]
    return histogram.reshape(-1, sectors, radial), counts


def validate_definition(definition):
    if not isinstance(definition, dict):
        raise ValueError("SPATIAL_BOTTLENECK requires fitted discretization; run prepare_trajectory_regions first")
    expected = {"schema_version": 1, "target": TARGET, "distance_unit": "meter",
                "coordinate_frame": "nomad_xy_ccw", "angular_convention": "uniform_centered_forward_ccw",
                "radial_boundary_rule": "left_closed_right_open", "waypoint_weighting": "uniform"}
    if any(definition.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported SPATIAL_BOTTLENECK trajectory-region definition")
    result = deepcopy(definition)
    for key in ("num_sectors", "num_radial_bins"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(f"SPATIAL_BOTTLENECK {key} must be a positive integer")
    epsilon = result.get("stationary_epsilon_m")
    if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("SPATIAL_BOTTLENECK stationary_epsilon_m must be finite and nonnegative")
    boundaries = result.get("radial_boundaries_m")
    if (not isinstance(boundaries, list) or len(boundaries) != result["num_radial_bins"] - 1
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in boundaries)
            or any(b <= a for a, b in zip([epsilon] + boundaries, boundaries))):
        raise ValueError("SPATIAL_BOTTLENECK radial boundaries must be strictly increasing and above stationary_epsilon_m")
    return result


def fit_settings(config):
    spatial_bottleneck = config.get("spatial_bottleneck", {})
    result = {"num_sectors": spatial_bottleneck.get("num_sectors", 6), "num_radial_bins": spatial_bottleneck.get("num_radial_bins", 4),
              "stationary_epsilon_m": spatial_bottleneck.get("stationary_epsilon_m", 1.e-4),
              "max_goal_distance": config["distance"]["max_dist_cat"],
              "min_action_distance": config["action"]["min_dist_cat"],
              "max_action_distance": config["action"]["max_dist_cat"],
              "max_fit_points": spatial_bottleneck.get("max_fit_points", 200000), "fit_seed": spatial_bottleneck.get("fit_seed", 0),
              "prior_smoothing_mass": spatial_bottleneck.get("prior_smoothing_mass", 1.e-4)}
    for key in ("num_sectors", "num_radial_bins", "max_fit_points"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"spatial_bottleneck.{key} must be a positive integer")
    for key in ("max_goal_distance", "min_action_distance", "max_action_distance", "fit_seed"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"SPATIAL_BOTTLENECK fitting {key} must be a nonnegative integer")
    if result["max_action_distance"] <= result["min_action_distance"] + 1:
        raise ValueError("SPATIAL_BOTTLENECK needs at least one strictly valid action goal distance")
    epsilon = result["stationary_epsilon_m"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("spatial_bottleneck.stationary_epsilon_m must be finite and nonnegative")
    if result["max_fit_points"] < result["num_radial_bins"]:
        raise ValueError("max_fit_points must be at least num_radial_bins")
    smooth = result["prior_smoothing_mass"]
    if type(smooth) not in (int, float) or not math.isfinite(smooth) or not 0 < smooth < 1:
        raise ValueError("spatial_bottleneck.prior_smoothing_mass must lie strictly between zero and one")
    return result


def fit_visitation_prior(bundle, config, definition, settings):
    """Exact expected mean of valid train action-sample histograms.

    ViNT samples each observation once and a goal offset uniformly in [0,M].
    Weight each histogram by P(action_mask=1 | observation), not its number of
    waypoints. This accounts for the shorter goal range near trajectory ends.
    """
    spacing = bundle["manifest"]["waypoint_spacing"]
    context, horizon = config["context_size"], config["len_traj_pred"]
    offsets = np.arange(1, horizon + 1) * spacing
    total = np.zeros((definition["num_sectors"], definition["num_radial_bins"]), dtype=np.float64)
    weight_sum, valid_windows = 0., 0
    for record in bundle["trajectories"]["train"]:
        positions, yaw = load_trajectory(bundle["root"] / record["name"] / "traj_data.pkl")
        for begin in range(context * spacing, len(positions) - horizon * spacing, 8192):
            times = np.arange(begin, min(begin + 8192, len(positions) - horizon * spacing))
            maximum = np.minimum(settings["max_goal_distance"], (len(positions) - times - 1) // spacing)
            valid_offsets = np.maximum(0, np.minimum(maximum, settings["max_action_distance"] - 1)
                                        - settings["min_action_distance"])
            weights = valid_offsets / (maximum + 1)
            delta = positions[times[:, None] + offsets[None]] - positions[times, None]
            cosine, sine = np.cos(yaw[times])[:, None], np.sin(yaw[times])[:, None]
            local = np.stack((delta[..., 0] * cosine + delta[..., 1] * sine,
                              -delta[..., 0] * sine + delta[..., 1] * cosine), axis=-1)
            # Match the dataset's float32 normalized action target, including
            # points that lie exactly on fitted radial boundaries.
            scale = bundle["manifest"]["action_scale_m"]
            local = (local / scale).astype(np.float32).astype(np.float64) * scale
            histograms, counts = metric_visitation_histograms(local, definition)
            weights *= counts > 0
            total += (histograms * weights[:, None, None]).sum(axis=0)
            weight_sum += float(weights.sum())
            valid_windows += int((weights > 0).sum())
    if weight_sum <= 0:
        raise ValueError("SPATIAL_BOTTLENECK train prior has no valid action samples")
    smoothing = settings["prior_smoothing_mass"]
    probability = (1 - smoothing) * total / weight_sum + smoothing / total.size
    return {"method": "expected_valid_action_sample_mean_v1", "fit_split": "train",
            "probability_flat": probability.reshape(-1).tolist(),
            "smoothing_mass": smoothing, "valid_windows": valid_windows,
            "expected_valid_samples_per_epoch": weight_sum}


def fit_region_statistics(bundle, config):
    """Fit equal-frequency radial boundaries from eligible train action windows.

    A bounded, deterministic uniform priority reservoir avoids storing every
    waypoint. Each eligible observation contributes once, irrespective of how
    often goals or the GOAL_RELATION relation stream sample it. No validation data is read
    here. Radii are invariant to the existing XY/yaw rotation.
    """
    settings = fit_settings(config)
    context, horizon = config["context_size"], config["len_traj_pred"]
    if type(context) is not int or context < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError("Expected context_size >= 0 and len_traj_pred > 0")
    spacing = bundle["manifest"]["waypoint_spacing"]
    rng = np.random.default_rng(settings["fit_seed"])
    values, priorities = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    eligible_windows = nonstationary_points = stationary_points = 0
    offsets = np.arange(1, horizon + 1) * spacing
    for record in bundle["trajectories"]["train"]:
        positions, _ = load_trajectory(bundle["root"] / record["name"] / "traj_data.pkl")
        for begin in range(context * spacing, len(positions) - horizon * spacing, 8192):
            times = np.arange(begin, min(begin + 8192, len(positions) - horizon * spacing))
            max_goal = np.minimum(settings["max_goal_distance"], (len(positions) - times - 1) // spacing)
            # ViNT uses strict min_action_distance < goal_offset < max_action_distance.
            times = times[max_goal >= max(1, settings["min_action_distance"] + 1)]
            if not len(times):
                continue
            eligible_windows += len(times)
            points = positions[times[:, None] + offsets[None]] - positions[times, None]
            radii = np.linalg.norm(points, axis=-1).reshape(-1)
            moving = radii > settings["stationary_epsilon_m"]
            stationary_points += int((~moving).sum())
            radii = radii[moving]
            nonstationary_points += len(radii)
            values = np.concatenate((values, radii))
            priorities = np.concatenate((priorities, rng.random(len(radii))))
            limit = settings["max_fit_points"]
            if len(values) > limit:
                keep = np.argpartition(priorities, limit - 1)[:limit]
                values, priorities = values[keep], priorities[keep]
    if not len(values):
        raise ValueError("SPATIAL_BOTTLENECK fitting found no eligible nonstationary train waypoints")
    boundaries = np.quantile(values, np.arange(1, settings["num_radial_bins"]) / settings["num_radial_bins"]).tolist()
    definition = validate_definition({"schema_version": 1, "target": TARGET, "distance_unit": "meter",
        "coordinate_frame": "nomad_xy_ccw", "angular_convention": "uniform_centered_forward_ccw",
        "radial_boundary_rule": "left_closed_right_open", "waypoint_weighting": "uniform",
        "num_sectors": settings["num_sectors"], "num_radial_bins": settings["num_radial_bins"],
        "radial_boundaries_m": boundaries, "stationary_epsilon_m": settings["stationary_epsilon_m"]})
    return {"schema_version": 2, "fit_split": "train", "fit_method": "uniform_priority_reservoir_quantiles_v1",
            "training_signature": training_signature(bundle, context, horizon), "fit_settings": settings,
            "definition": definition, "eligible_windows": eligible_windows,
            "nonstationary_points": nonstationary_points, "stationary_points": stationary_points,
            "reservoir_points": len(values),
            "visitation_prior": fit_visitation_prior(bundle, config, definition, settings)}


def load_region_statistics(path, bundle, config):
    with Path(path).open() as stream:
        saved = json.load(stream)
    if (saved.get("schema_version") != 2 or saved.get("fit_split") != "train"
            or saved.get("fit_method") != "uniform_priority_reservoir_quantiles_v1"):
        raise ValueError("Expected SPATIAL_BOTTLENECK statistics schema 2 with a train-only prior; rerun prepare_trajectory_regions")
    if saved.get("training_signature") != training_signature(bundle, config["context_size"], config["len_traj_pred"]):
        raise ValueError("SPATIAL_BOTTLENECK statistics do not match the train trajectories, scale or sampling")
    if saved.get("fit_settings") != fit_settings(config):
        raise ValueError("SPATIAL_BOTTLENECK statistics fitting settings differ from the training config")
    definition = validate_definition(saved["definition"])
    for key in ("num_sectors", "num_radial_bins", "stationary_epsilon_m"):
        if definition[key] != saved["fit_settings"][key]:
            raise ValueError(f"SPATIAL_BOTTLENECK statistics definition disagrees with fitting setting {key}")
    prior = saved.get("visitation_prior", {})
    probability = np.asarray(prior.get("probability_flat", []), dtype=np.float64)
    if (prior.get("method") != "expected_valid_action_sample_mean_v1" or prior.get("fit_split") != "train"
            or prior.get("smoothing_mass") != saved["fit_settings"]["prior_smoothing_mass"]
            or probability.shape != (definition["num_sectors"] * definition["num_radial_bins"],)
            or not np.isfinite(probability).all() or np.any(probability <= 0)
            or not np.isclose(probability.sum(), 1., rtol=0, atol=1.e-10)):
        raise ValueError("Invalid SPATIAL_BOTTLENECK train-only visitation prior; rerun prepare_trajectory_regions")
    return definition, digest_json(saved), saved

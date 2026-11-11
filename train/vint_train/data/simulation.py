"""Habitat file contract and provenance checks; no simulator or torch dependency."""

import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "calibration_temperature", "calibration_threshold", "test")
UNKNOWN_STATUSES = {"timeout", "incomplete_map", "invalid_projection", "indeterminate"}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be filled in with a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object: {path}:{line_number}")
            rows.append(row)
    return rows


def validate_label_definition(labels):
    required = ("label_schema_version", "distance_unit", "max_path_length_m",
                "robot_footprint_id", "planner_config_id", "motion_constraint_id",
                "unknown_space_policy", "query_region", "max_projection_error_m")
    for key in required:
        if key not in labels or labels[key] is None or labels[key] == "":
            raise ValueError(f"Missing label definition: {key}")
    if labels["label_schema_version"] != 1 or labels["distance_unit"] != "meter":
        raise ValueError("Expected label schema 1 with meter distances")
    if labels["motion_constraint_id"] != "static_footprint_holonomic":
        raise ValueError("GOAL_RELATION v1 labels describe static footprint geometry, not vehicle dynamics")
    if labels["query_region"] != "full_scene_navmesh":
        raise ValueError("GOAL_RELATION v1 requires full_scene_navmesh queries")
    result = dict(labels)
    result["max_path_length_m"] = positive(labels["max_path_length_m"], "max_path_length_m")
    tolerance = float(labels["max_projection_error_m"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_projection_error_m must be finite and nonnegative")
    result["max_projection_error_m"] = tolerance
    return result


def validate_relation(row, labels, context_size, root):
    for key in ("reachable", "path_length_m", "start_projection_error_m", "goal_projection_error_m"):
        if key not in row:
            raise ValueError(f"Relation requires explicit {key} (use null for unknown)")
    for key in ("pair_id", "scene_id", "episode_id", "geometry_map_id",
                "start_pose_ref", "goal_pose_ref", "query_status"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Relation requires nonempty {key}")
    paths = row.get("context_images")
    if not isinstance(paths, list) or len(paths) != context_size + 1:
        raise ValueError(f"{row['pair_id']}: expected oldest-to-current context_size+1 images")
    for path in paths + [row.get("goal_image")]:
        if not isinstance(path, str) or not (root / path).is_file():
            raise ValueError(f"{row['pair_id']}: RGB file missing: {path}")
    reachable = row.get("reachable")
    if reachable is not None and (type(reachable) is not int or reachable not in (0, 1)):
        raise ValueError("reachable must be 0, 1, or null")
    if type(row.get("distance_valid")) is not bool or type(row.get("is_shortest")) is not bool:
        raise ValueError("distance_valid and is_shortest must be booleans")
    status = row["query_status"]
    errors = [row.get("start_projection_error_m"), row.get("goal_projection_error_m")]
    if any(x is not None and (type(x) not in (int, float) or not math.isfinite(x) or x < 0) for x in errors):
        raise ValueError("Projection errors must be finite and nonnegative, or null")
    projection_ok = all(type(x) in (int, float) and math.isfinite(x) and
                        0 <= x <= labels["max_projection_error_m"] for x in errors)
    if reachable is not None and not projection_ok:
        raise ValueError("Known labels require valid endpoint projections")
    length = row.get("path_length_m")
    length_ok = type(length) in (int, float) and math.isfinite(length) and length >= 0
    if length is not None and not length_ok:
        raise ValueError("path_length_m must be a nonnegative finite number or null")
    bound = labels["max_path_length_m"]
    if reachable == 1:
        if status != "found" or not length_ok or length > bound:
            raise ValueError("R=1 requires a found feasible path of length <= H")
        if row["distance_valid"] != row["is_shortest"]:
            raise ValueError("Only verified shortest positive paths supervise D")
    elif reachable == 0:
        certified_disconnected = status == "no_path_certified"
        if certified_disconnected and (length is not None or row["is_shortest"]):
            raise ValueError("Certified disconnection must have path_length_m=null and is_shortest=false")
        too_long = status == "found" and row["is_shortest"] and length_ok and length > bound
        if not (certified_disconnected or too_long):
            raise ValueError("R=0 requires certified disconnection or a shortest path > H")
    elif status not in UNKNOWN_STATUSES and not (status == "found" and not row["is_shortest"] and length_ok and length > bound):
        raise ValueError("Unknown labels need an indeterminate query outcome")
    if row["distance_valid"] and (reachable != 1 or not row["is_shortest"] or not length_ok):
        raise ValueError("D requires a known reachable shortest path")
    if reachable != 1 and row["distance_valid"]:
        raise ValueError("Negative/unknown pairs cannot supervise D")
    return row


def load_trajectory(path):
    # Same trusted-local-pickle contract as ViNT_Dataset.
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    position = np.asarray(data["position"], dtype=np.float64)
    yaw = np.asarray(data["yaw"], dtype=np.float64).reshape(-1)
    if position.shape != (len(yaw), 2) or not np.isfinite(position).all() or not np.isfinite(yaw).all():
        raise ValueError(f"Invalid metric XY/radian trajectory: {path}")
    return position, yaw


def load_manifest(path, context_size=3, verify_files=True):
    path = Path(path).resolve()
    with path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("platform") != "habitat":
        raise ValueError("Expected Habitat simulation manifest schema_version=1")
    if manifest.get("coordinate_frame") != "nomad_xy_ccw":
        raise ValueError("Convert Habitat coordinates to nomad_xy_ccw before training")
    if not manifest.get("data_revision"):
        raise ValueError("Provide an immutable data_revision for the RGB export")
    if not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) - set(SPLITS):
        raise ValueError(f"splits must use only {SPLITS}")
    manifest["action_scale_m"] = positive(manifest["action_scale_m"], "action_scale_m")
    manifest["action_time_step_s"] = positive(manifest["action_time_step_s"], "action_time_step_s")
    if type(manifest.get("waypoint_spacing")) is not int or manifest["waypoint_spacing"] < 1:
        raise ValueError("waypoint_spacing must be a positive integer")
    labels = validate_label_definition(manifest["label_definition"])
    manifest["label_definition"] = labels
    root = (path.parent / manifest["data_root"]).resolve()
    split_scenes, trajectories, relations, hashes = {}, {}, {}, {}
    trajectory_splits, pair_ids = {}, set()
    for split in SPLITS:
        spec = manifest["splits"].get(split, {"trajectories": []})
        split_scenes[split] = set()
        trajectories[split] = spec.get("trajectories", [])
        for record in trajectories[split]:
            for key in ("name", "scene_id", "episode_id", "geometry_map_id"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"Trajectory requires {key}")
            if record["name"] in trajectory_splits:
                raise ValueError(f"Duplicate trajectory {record['name']}")
            trajectory_splits[record["name"]] = split
            split_scenes[split].add(record["scene_id"])
            trajectory_path = root / record["name"] / "traj_data.pkl"
            if verify_files:
                position, _ = load_trajectory(trajectory_path)
                if len(position) < 2:
                    raise ValueError(f"Trajectory too short: {trajectory_path}")
                for t in range(len(position)):
                    if not (root / record["name"] / f"{t}.jpg").is_file():
                        raise ValueError(f"Missing RGB frame {record['name']}/{t}.jpg")
                hashes[f"trajectory/{record['name']}"] = file_sha256(trajectory_path)
        relation_file = spec.get("relations")
        rows = [] if relation_file is None else read_jsonl(path.parent / relation_file)
        if relation_file:
            hashes[f"relations/{split}"] = file_sha256(path.parent / relation_file)
        for row in rows:
            validate_relation(row, labels, context_size, root)
            if row["pair_id"] in pair_ids:
                raise ValueError(f"Duplicate relation pair_id: {row['pair_id']}")
            pair_ids.add(row["pair_id"])
            split_scenes[split].add(row["scene_id"])
        relations[split] = rows
    # Catch reused RGB files or mislabeled scene IDs across the two streams.
    declared_directories = {(root / r["name"]).resolve(): (split, r)
                            for split in SPLITS for r in trajectories[split]}
    image_splits = {}
    for split in SPLITS:
        for row in relations[split]:
            for relative in row["context_images"] + [row["goal_image"]]:
                image = (root / relative).resolve()
                owner = declared_directories.get(image.parent)
                if owner is not None and (owner[0] != split or owner[1]["scene_id"] != row["scene_id"]):
                    raise ValueError(f"Relation RGB conflicts with action trajectory split/scene: {relative}")
                if image in image_splits and image_splits[image] != split:
                    raise ValueError(f"RGB reused across relation splits: {relative}")
                image_splits[image] = split
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            overlap = split_scenes[first] & split_scenes[second]
            if overlap:
                raise ValueError(f"Scene leakage between {first} and {second}: {sorted(overlap)}")
    if not trajectories["train"]:
        raise ValueError("Scratch training needs simulation action trajectories")
    return {"manifest": manifest, "path": path, "root": root, "relations": relations,
            "trajectories": trajectories, "scenes": split_scenes,
            "fingerprint": digest_json({"manifest": manifest, "files": hashes}), "file_hashes": hashes}


def training_signature(bundle, context_size, horizon):
    manifest = bundle["manifest"]
    records = bundle["trajectories"]["train"]
    return digest_json({"records": records, "hashes": {r["name"]: bundle["file_hashes"][f"trajectory/{r['name']}"] for r in records},
                        "action_scale_m": manifest["action_scale_m"],
                        "action_time_step_s": manifest["action_time_step_s"],
                        "waypoint_spacing": manifest["waypoint_spacing"],
                        "context_size": context_size, "horizon": horizon,
                        "data_revision": manifest["data_revision"]})


def load_action_statistics(path, bundle, context_size, horizon):
    with Path(path).open() as stream:
        saved = json.load(stream)
    expected = training_signature(bundle, context_size, horizon)
    if saved.get("schema_version") != 1 or saved.get("training_signature") != expected:
        raise ValueError("Action statistics do not match this training split, scale, or sampling configuration")
    if saved.get("fit_split") != "train":
        raise ValueError("Action bounds must be fitted on the training split only")
    for key in ("action_scale_m", "action_time_step_s"):
        if saved.get(key) != bundle["manifest"][key]:
            raise ValueError(f"Action statistics metadata mismatch: {key}")
    return saved

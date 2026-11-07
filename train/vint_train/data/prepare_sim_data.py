"""Validate exported Habitat data and fit action bounds on train trajectories only."""

import argparse
import json
from pathlib import Path

import numpy as np

from .simulation import load_manifest, load_trajectory, training_signature


def compute_statistics(bundle, context_size, horizon):
    if type(context_size) is not int or context_size < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError("Expected context_size >= 0 and horizon > 0")
    spec, root = bundle["manifest"], bundle["root"]
    spacing, scale = spec["waypoint_spacing"], spec["action_scale_m"]
    minimum, maximum = np.full(2, np.inf), np.full(2, -np.inf)
    count = 0
    for record in bundle["trajectories"]["train"]:
        positions, yaw = load_trajectory(root / record["name"] / "traj_data.pkl")
        for t in range(context_size * spacing, len(positions) - horizon * spacing):
            c, s = np.cos(yaw[t]), np.sin(yaw[t])
            rotation = np.array([[c, -s], [s, c]])
            points = positions[t:t + (horizon + 1) * spacing:spacing]
            deltas = np.diff((points - positions[t]) @ rotation, axis=0) / scale
            minimum = np.minimum(minimum, deltas.min(axis=0))
            maximum = np.maximum(maximum, deltas.max(axis=0))
            count += 1
    if not count or not np.isfinite([minimum, maximum]).all() or np.any(maximum <= minimum):
        raise ValueError("Training data has no complete windows or a degenerate action dimension")
    return {"schema_version": 1, "fit_split": "train", "num_windows": count,
            "training_signature": training_signature(bundle, context_size, horizon),
            "action_stats": {"min": minimum.tolist(), "max": maximum.tolist()},
            "action_scale_m": scale, "action_time_step_s": spec["action_time_step_s"]}


def audit_labels(bundle):
    report = {}
    bound = bundle["manifest"]["label_definition"]["max_path_length_m"]
    for split, rows in bundle["relations"].items():
        base_distances = [r["path_length_m"] for r in rows if r.get("is_base_subgoal") is True
                          and r["is_shortest"] and r["query_status"] == "found"
                          and r.get("path_length_m") is not None]
        report[split] = {"pairs": len(rows), "scenes": len(bundle["scenes"][split]),
                         "positive": sum(r["reachable"] == 1 for r in rows),
                         "negative": sum(r["reachable"] == 0 for r in rows),
                         "unknown": sum(r["reachable"] is None for r in rows),
                         "distance_valid": sum(r["distance_valid"] for r in rows),
                         "base_shortest_path_count": len(base_distances),
                         "base_path_quantiles_m": None if not base_distances else np.quantile(base_distances, [0, .5, .9, .95, 1]).tolist(),
                         "base_fraction_within_H": None if not base_distances else float(np.mean(np.array(base_distances) <= bound))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Training action-statistics JSON")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    bundle = load_manifest(args.manifest, args.context_size)
    statistics = compute_statistics(bundle, args.context_size, args.horizon)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    report = {"data_fingerprint": bundle["fingerprint"], "labels": audit_labels(bundle)}
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote train-only bounds to {output}; audit to {output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()

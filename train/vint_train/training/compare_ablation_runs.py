"""Compare saved run metadata and observed budgets; does not launch training."""

import argparse
import copy
import json
from pathlib import Path


def common_spec(provenance):
    spec = copy.deepcopy(provenance["training_spec"])
    spec.pop("spatial_bottleneck", None)
    spec.pop("goal_relation", None)
    spec["normalization"] = spec["normalization"].replace("; visitation_valid_sample_count", "")
    model = copy.deepcopy(provenance.get("model_config", {}))
    model.pop("spatial_bottleneck", None)
    model.pop("goal_relation", None)
    return {"training": spec, "model": model, "data": provenance["data_fingerprint"],
            "initial_common_weights": provenance["initial_common_weights_sha256"]}


def compare_runs(folders):
    records, differences = [], []
    reference = None
    spatial_bottleneck_reference = goal_relation_reference = None
    spatial_bottleneck_dimensions = None
    supervised_coefficient = None
    initial_extensions = {}
    for folder in map(Path, folders):
        provenance = json.loads((folder / "provenance.json").read_text())
        spec = provenance["training_spec"]
        common = common_spec(provenance)
        if not provenance.get("common_initialization_verified") or not common["model"]:
            differences.append({"run": str(folder), "field": "missing current initialization/model audit"})
        if reference is None:
            reference = common
        for key in common:
            if common[key] != reference[key]:
                differences.append({"run": str(folder), "field": key})
        spatial_bottleneck, goal_relation = spec.get("spatial_bottleneck", {}), spec.get("goal_relation", {})
        # Common supervised SPATIAL_BOTTLENECK target/weight settings must match even across
        # query/global/pooled controls. tokens_only intentionally has no CE.
        if spatial_bottleneck.get("enabled"):
            dimensions = {k: provenance.get("model_config", {}).get("spatial_bottleneck", {}).get(k)
                          for k in ("num_sectors", "num_radial_bins", "token_dim", "num_attention_heads", "fusion")}
            if spatial_bottleneck_dimensions is None:
                spatial_bottleneck_dimensions = dimensions
            if dimensions != spatial_bottleneck_dimensions:
                differences.append({"run": str(folder), "field": "SPATIAL_BOTTLENECK common dimensions/fusion"})
            target_settings = {k: spatial_bottleneck.get(k) for k in ("discretization", "statistics_sha256", "prior_probability")}
            if spatial_bottleneck_reference is None:
                spatial_bottleneck_reference = target_settings
            if target_settings != spatial_bottleneck_reference:
                differences.append({"run": str(folder), "field": "SPATIAL_BOTTLENECK target/statistics"})
            if spatial_bottleneck.get("lambda_visitation", 0) > 0:
                if supervised_coefficient is None:
                    supervised_coefficient = spatial_bottleneck["lambda_visitation"]
                if spatial_bottleneck["lambda_visitation"] != supervised_coefficient:
                    differences.append({"run": str(folder), "field": "SPATIAL_BOTTLENECK supervised loss weight"})
        if goal_relation.get("enabled"):
            if goal_relation_reference is None:
                goal_relation_reference = goal_relation
            if goal_relation != goal_relation_reference:
                differences.append({"run": str(folder), "field": "GOAL_RELATION supervision/selector"})
        for extension in ("spatial_bottleneck", "goal_relation"):
            architecture = copy.deepcopy(provenance.get("model_config", {}).get(extension, {}))
            if architecture.get("enabled"):
                architecture.pop("mode", None)
                architecture.pop("visitation_head_trained", None)
                key = extension + json.dumps(architecture, sort_keys=True)
                fingerprint = provenance.get(f"initial_{extension}_weights_sha256")
                if key not in initial_extensions:
                    initial_extensions[key] = fingerprint
                if fingerprint is None or fingerprint != initial_extensions[key]:
                    differences.append({"run": str(folder), "field": extension + " matched-architecture initialization"})
        path = folder / "metrics.jsonl"
        metrics = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        epochs = [m["epoch"] for m in metrics]
        fields = ("action_samples", "action_supervised_samples", "relation_samples", "updates", "amp_skips",
                  "updated_action_samples", "updated_action_supervised_samples", "updated_relation_samples",
                  "updated_visitation_valid_samples", "wall_time_s")
        totals = {key: sum(m["train"][key] for m in metrics) if metrics and all(key in m["train"] for m in metrics) else None
                  for key in fields}
        complete = (epochs == list(range(spec["epochs"])) and all(totals[key] is not None for key in fields))
        records.append({"run": str(folder), "spatial_bottleneck": {k: spatial_bottleneck.get(k) for k in ("enabled", "mode", "representation", "lambda_visitation")},
                        "goal_relation_enabled": goal_relation.get("enabled", False), "epochs_logged": epochs,
                        "full_epoch_coverage": complete, "budget": totals,
                        "parameter_counts": provenance.get("parameter_counts"),
                        "peak_cuda_allocated_bytes": max((m["train"].get("peak_cuda_allocated_bytes") or 0 for m in metrics), default=0) or None})
    complete = bool(records) and all(r["full_epoch_coverage"] for r in records)
    action_keys = ("action_samples", "action_supervised_samples")
    update_keys = ("updates", "updated_action_samples", "updated_action_supervised_samples")
    def matched(keys):
        return (len({tuple(r["budget"][k] for k in keys) for r in records}) == 1) if complete else None
    return {"configuration_and_initialization_match": not differences, "differences": differences,
            "consumed_action_budgets_match": matched(action_keys), "successful_update_budgets_match": matched(update_keys),
            "runs": records,
            "scope": "Compare paired runs of one seed. Partial/resumed-only logs cannot certify full budgets. "
                     "GOAL_RELATION extra labels, parameter counts and time are reported separately. "
                     "This audit does not establish navigation improvements or equal compute."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Folders containing provenance.json and metrics.jsonl")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two runs with the same seed")
    report = compare_runs(args.runs)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(content)
    if (not report["configuration_and_initialization_match"] or report["consumed_action_budgets_match"] is False
            or report["successful_update_budgets_match"] is False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Fit R3 on two held-out splits, or evaluate a fixed artifact on test data."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vint_train.data.simulation import load_manifest
from vint_train.data.goal_relation_dataset import GoalRelationDataset
from vint_train.models.weights import load_model
from vint_train.inference.goal_relation_selector import (choose_subgoal, localization_window,
                                                        prepare_route_decision, legal_candidate_nodes)
from vint_train.inference.calibration import load_calibration, selector_spec, calibration_binding
from .simulation_train import collect_relations, finite_json
from .relation_metrics import binary_metrics, choose_threshold, distance_metrics


def fit_temperature(logits, targets):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    if not len(logits) or len(targets.unique()) != 2 or not torch.isfinite(logits).all():
        raise ValueError("Temperature fitting requires finite logits and both classes")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp(), targets)
        if not torch.isfinite(loss):
            raise ValueError("Temperature optimization became non-finite")
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature fit failed")
    return temperature


def decision_report(rows, predictions, temperature, threshold, settings):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("decision_id") is not None:
            groups[(row["scene_id"], row["episode_id"], row["decision_id"])].append(i)
    if not groups:
        raise ValueError("Decision evaluation requires complete groups with decision_id and route_context")
    totals = {key: 0 for key in ("groups", "selected", "abstain", "selected_known", "selected_unknown",
                                "selected_unreachable", "base_accepted", "fallback", "fallback_selected", "distance_changed")}
    for indices in groups.values():
        context = rows[indices[0]].get("route_context")
        if context is None:
            raise ValueError("Grouped decision evaluation requires route_context")
        node_to_index = {rows[i]["candidate_node_index"]: i for i in indices}
        if len(node_to_index) != len(indices) or set(node_to_index) != set(context["candidate_nodes"]):
            raise ValueError("A decision group must include every candidate exactly once")
        if any(rows[i].get("route_context") != context or rows[i]["context_images"] != rows[indices[0]]["context_images"] for i in indices):
            raise ValueError("Decision groups must share RGB context and route metadata")
        ordered = localization_window(context["previous_localized_node"], settings["radius"], context["route_nodes"],
                                      context["completed_through"], context.get("required_nodes", []))
        if set(ordered) - set(node_to_index):
            raise ValueError("Decision group is missing localization-window candidates")
        temporal = [predictions["temporal_distance"][node_to_index[n]] for n in ordered]
        if not np.isfinite(temporal).all():
            raise ValueError("Non-finite temporal predictions")
        prepared = prepare_route_decision(ordered, temporal, settings["close_threshold"], context["route_nodes"],
                                          context["completed_through"], context.get("required_nodes", []), context.get("broken_after", []))
        base, completed = prepared["base_subgoal"], prepared["completed_through"]
        legal = legal_candidate_nodes(base, completed, context["route_nodes"], context.get("required_nodes", []),
                                      context.get("broken_after", []), settings["max_forward_skip"])
        if set(legal) - set(node_to_index):
            raise ValueError("Decision group is missing legal base/forward candidates")
        candidates = [{"node_index": node, "probability": float(torch.sigmoid(torch.tensor(predictions["logit"][node_to_index[node]], dtype=torch.float64) / temperature)),
                       "distance_m": predictions["distance_m"][node_to_index[node]]} for node in node_to_index]
        choice = choose_subgoal(candidates, base, completed, context["route_nodes"],
                                context.get("required_nodes", []), context.get("broken_after", []), threshold,
                                settings["use_distance"], settings["max_forward_skip"])
        totals["groups"] += 1
        totals["base_accepted"] += int(choice["base_accepted"])
        totals["fallback"] += int(choice["fallback"])
        totals["distance_changed"] += int(choice["distance_changed_selection"])
        selected = choice["selected_subgoal"]
        if selected is None:
            totals["abstain"] += 1
        else:
            totals["selected"] += 1
            totals["fallback_selected"] += int(choice["fallback"])
            label = rows[node_to_index[selected]]["reachable"]
            totals["selected_unknown"] += int(label is None)
            totals["selected_known"] += int(label is not None)
            totals["selected_unreachable"] += int(label == 0)
    n = totals["groups"]
    totals.update(base_accept_rate=totals["base_accepted"] / n if n else None,
                  fallback_rate=totals["fallback"] / n if n else None,
                  distance_changed_selection_rate=totals["distance_changed"] / n if n else None,
                  distance_changed_selection_rate_on_fallback=totals["distance_changed"] / totals["fallback_selected"] if totals["fallback_selected"] else None,
                  decision_error=totals["selected_unreachable"] / totals["selected_known"] if totals["selected_known"] else None,
                  grouped_candidate_count=sum(len(indices) for indices in groups.values()),
                  ungrouped_candidate_count=sum(row.get("decision_id") is None for row in rows),
                  note="Offline grouped decisions; not closed-loop success or a risk guarantee")
    return totals


def report(rows, predictions, temperature, threshold, settings, distance_head_trained=True):
    p = torch.sigmoid(torch.tensor(predictions["logit"], dtype=torch.float64) / temperature).numpy()
    known = np.asarray(predictions["reach_valid"], bool)
    accepted_known_indices = np.flatnonzero(known & (p >= threshold))
    accepted_indices = np.flatnonzero(p >= threshold)
    metrics = binary_metrics(p[known], np.asarray(predictions["target"])[known], threshold)
    return {"candidate_metrics": metrics, "unknown_count": int((~known).sum()),
            "accepted_unknown_count": int(((~known) & (p >= threshold)).sum()),
            "accepted_scene_count": len({rows[i]["scene_id"] for i in accepted_indices}),
            "accepted_episode_count": len({(rows[i]["scene_id"], rows[i]["episode_id"]) for i in accepted_indices}),
            "accepted_known_scene_count": len({rows[i]["scene_id"] for i in accepted_known_indices}),
            "accepted_known_episode_count": len({(rows[i]["scene_id"], rows[i]["episode_id"]) for i in accepted_known_indices}),
            **distance_metrics(predictions, distance_head_trained),
            "decisions": decision_report(rows, predictions, temperature, threshold, settings)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fdr", type=float, default=.05)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--calibration", help="Evaluate this artifact; do not refit")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format_version") != 2 or "inference_config" not in saved:
        raise ValueError("Calibration requires a complete v2 simulation checkpoint")
    config = saved["inference_config"]
    if not config.get("r3", {}).get("enabled"):
        raise ValueError("This model has no R3 head")
    bundle = load_manifest(args.manifest, config["context_size"])
    if bundle["manifest"]["label_definition"] != config["r3"]["label_definition"]:
        raise ValueError("Calibration labels do not match the trained model")
    if bundle["fingerprint"] != saved["data_fingerprint"]:
        raise ValueError("The dataset/split fingerprint changed since training")
    model = load_model(args.checkpoint, config, torch.device(args.device), use_ema=args.weights == "ema")
    settings = selector_spec(config)
    distance_head_trained = config.get("r3", {}).get("lambda_distance", 1.) > 0
    def collect(split):
        rows = bundle["relations"][split]
        if not rows:
            raise ValueError(f"No relation rows in {split}")
        data = GoalRelationDataset(rows, bundle["root"], config["image_size"])
        predictions = collect_relations(model, data, config, torch.device(args.device))
        if not all(np.isfinite(predictions[key]).all() for key in ("logit", "distance_m", "temporal_distance")):
            raise ValueError("Non-finite model predictions cannot be calibrated or evaluated")
        return rows, predictions
    if args.calibration:
        artifact = load_calibration(args.calibration, model, config)
        if artifact.get("data_fingerprint") != bundle["fingerprint"]:
            raise ValueError("Calibration data/split fingerprint differs from the evaluation manifest")
        rows, predictions = collect(args.split)
        output = {"status": "evaluated", "split": args.split,
                  "weights_sha256": model._weights_sha256,
                  "report": report(rows, predictions, artifact["temperature"], artifact["threshold"], settings, distance_head_trained)}
    else:
        try:
            temp_rows, temp = collect("calibration_temperature")
            valid = np.asarray(temp["reach_valid"], bool)
            temperature = fit_temperature(np.asarray(temp["logit"])[valid], np.asarray(temp["target"])[valid])
            rows, predictions = collect("calibration_threshold")
            valid = np.asarray(predictions["reach_valid"], bool)
            p = torch.sigmoid(torch.tensor(predictions["logit"], dtype=torch.float64) / temperature).numpy()
            threshold, _ = choose_threshold(p[valid], np.asarray(predictions["target"])[valid], args.max_fdr, args.min_accepted)
            output = {"schema_version": 2, "status": "calibrated", "temperature": temperature,
                      "threshold": threshold, **calibration_binding(model, config),
                      "label_definition": config["r3"]["label_definition"],
                      "data_fingerprint": bundle["fingerprint"], "max_empirical_fdr": args.max_fdr,
                      "min_accepted": args.min_accepted,
                      "temperature_fit_count": int(np.asarray(temp["reach_valid"]).sum()),
                      "report": report(rows, predictions, temperature, threshold, settings, distance_head_trained),
                      "uncalibrated_report": report(rows, predictions, 1., .5, settings, distance_head_trained),
                      "guarantee": "none; empirical held-out metrics only"}
        except (ValueError, KeyError, TypeError) as exc:
            output = {"schema_version": 2, "status": "failed", "reason": str(exc)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(finite_json(output), indent=2, allow_nan=False) + "\n")
    print(f"{output['status']}: {destination}")
    if output["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

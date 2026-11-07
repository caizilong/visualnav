"""Habitat-only, random-initialized NoMaD/R3 training with two data streams."""

import copy
import hashlib
import itertools
import json
import math
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from vint_train.data.simulation import load_manifest, load_action_statistics
from vint_train.data.goal_relation_dataset import GoalRelationDataset
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.factory import build_model
from vint_train.models.weights import weights_sha256
from vint_train.models.action_codec import normalize_rgb, validate_action_stats
from vint_train.inference.rgb_policy import make_noise_scheduler, score_rgb_pairs, sample_conditioned_actions
from .checkpoint import save_checkpoint
from .simulation_checkpoint import make_sim_checkpoint, restore_sim_checkpoint
from .goal_relation_train import (action_loss_sums, relation_loss_sums, relation_weight,
                                  window_denominators, weighted_loss, optimizer_update)
from .relation_metrics import binary_metrics, distance_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def stream_seed(seed, epoch, stream, cycle=0):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{stream}:{cycle}".encode()).digest()[:8], "big")


@contextmanager
def isolated_evaluation_rng(cuda_devices):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def loader_for(dataset, batch_size, workers, seed, shuffle=True):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=False, persistent_workers=False, worker_init_fn=seed_worker,
                      generator=generator, pin_memory=torch.cuda.is_available())


class RelationStream:
    def __init__(self, dataset, config, epoch):
        self.dataset, self.config, self.epoch = dataset, config, epoch
        self.cycle = self.seen = 0
        self.iterator = None

    def next(self):
        if self.iterator is None:
            self.iterator = iter(loader_for(self.dataset, self.config["relation_batch_size"],
                                 self.config["num_workers"], stream_seed(self.config["seed"], self.epoch, "relation", self.cycle)))
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.cycle += 1
            self.iterator = None
            return self.next()
        self.seen += len(batch["obs_image"])
        return batch


def build_action_dataset(bundle, split, config):
    records = bundle["trajectories"][split]
    if not records:
        return None
    cache_root = Path(config["simulation"].get("cache_dir", bundle["path"].parent / ".nomad_cache"))
    split_dir = cache_root / bundle["fingerprint"] / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "traj_names.txt").write_text("\n".join(r["name"] for r in records) + "\n")
    spacing = bundle["manifest"]["waypoint_spacing"]
    return ViNT_Dataset(
        data_folder=str(bundle["root"]), data_split_folder=str(split_dir), dataset_name="habitat",
        image_size=config["image_size"], waypoint_spacing=spacing,
        min_dist_cat=config["distance"]["min_dist_cat"], max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"], max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=True, len_traj_pred=config["len_traj_pred"], learn_angle=False,
        context_size=config["context_size"], context_type="temporal", end_slack=0,
        goals_per_obs=1, normalize=True, dataset_index=0,
        dataset_config={"metric_waypoint_spacing": config["action_scale_m"] / spacing})


def make_optimizer(model, config):
    groups = {name: [] for name in ("vision", "transformer", "temporal", "diffusion", "relation")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"Scratch training must not freeze {name}")
        if name.startswith("vision_encoder.sa_"):
            group = "transformer"
        elif name.startswith("vision_encoder."):
            group = "vision"
        elif name.startswith("dist_pred_net."):
            group = "temporal"
        elif name.startswith("noise_pred_net."):
            group = "diffusion"
        elif name.startswith("goal_relation_head."):
            group = "relation"
        else:
            raise ValueError(f"Unassigned trainable parameter: {name}")
        groups[group].append(parameter)
    specs = [{"name": name, "params": parameters, "lr": float(config["lr"])}
             for name, parameters in groups.items() if parameters]
    ids = [id(p) for group in specs for p in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise ValueError("Optimizer must cover every parameter exactly once")
    return torch.optim.AdamW(specs, weight_decay=float(config.get("weight_decay", .01)))


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(finite_json(value), allow_nan=False) + "\n")


def autocast(config, device):
    return torch.cuda.amp.autocast(dtype=torch.float16) if config["amp"] and device.type == "cuda" else nullcontext()


def gradient_probe(loss, model):
    named = dict(model.named_parameters())
    names = ("vision_encoder.compress_obs_enc.weight",
             f"vision_encoder.sa_encoder.layers.{len(model.vision_encoder.sa_encoder.layers)-1}.linear2.weight")
    parameters = [named[n] for n in names if n in named]
    if not parameters or loss is None:
        return None
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten().cpu()
                      for p, g in zip(parameters, gradients)])


def train_epoch(model, ema, optimizer, scaler, noise_scheduler, action_dataset,
                relation_dataset, config, device, epoch, successful_steps, folder):
    model.train()
    actions = iter(loader_for(action_dataset, config["batch_size"], config["num_workers"],
                              stream_seed(config["seed"], epoch, "action")))
    relation_stream = RelationStream(relation_dataset, config, epoch) if relation_dataset is not None else None
    mask_generator = torch.Generator().manual_seed(stream_seed(config["seed"], epoch, "mask"))
    totals = {k: 0 for k in ("windows", "updates", "amp_skips", "unsupervised_windows", "action_samples",
                             "relation_samples", "r3_positive_count", "r3_negative_count", "r3_unknown_count",
                             "r3_distance_valid_count")}
    objectives = []
    alpha = float(config["alpha"])
    r3 = config.get("r3", {})
    while True:
        window = list(itertools.islice(actions, config["gradient_accumulation_steps"]))
        if not window:
            break
        relations = [relation_stream.next() for _ in window] if relation_stream else []
        masks = [(torch.rand(len(batch[0]), generator=mask_generator) < config["goal_mask_prob"]).long() for batch in window]
        denom = window_denominators(window, masks, relations)
        weight = relation_weight(successful_steps, r3.get("lambda_start", .1), r3.get("lambda_end", 1.), r3.get("warmup_steps", 500))
        weights_a = {"action": 1 - alpha, "temporal": alpha}
        weights_r = {"reach": weight * r3.get("lambda_reach", 1.),
                     "distance": weight * r3.get("lambda_distance", 1.)}
        totals["windows"] += 1
        totals["action_samples"] += denom["action_samples"]
        for batch in relations:
            totals["relation_samples"] += len(batch["obs_image"])
            known = batch["reach_valid"]
            totals["r3_positive_count"] += int(((batch["reachable"] == 1) & known).sum())
            totals["r3_negative_count"] += int(((batch["reachable"] == 0) & known).sum())
            totals["r3_unknown_count"] += int((~known).sum())
            totals["r3_distance_valid_count"] += int(batch["distance_valid"].sum())
        optimizer.zero_grad(set_to_none=True)
        sums_cpu = {key: 0.0 for key in ("action", "temporal", "reach", "distance")}
        objective, backward_count = 0.0, 0
        probe_a = probe_r = None
        probe_every = config.get("gradient_probe_every", 200)
        probe = probe_every > 0 and successful_steps % probe_every == 0
        for index, (batch, mask) in enumerate(zip(window, masks)):
            if (denom["action_count"] and weights_a["action"] > 0) or (denom["temporal_count"] and weights_a["temporal"] > 0):
                with autocast(config, device):
                    sums = action_loss_sums(model, batch, mask, noise_scheduler, config, device, config["activation_checkpointing"])
                    loss = weighted_loss(sums, denom, weights_a)
                for key, value in sums.items():
                    sums_cpu[key] += float(value.detach())
                if loss is not None:
                    if probe and index == 0:
                        probe_a = gradient_probe(loss, model)
                    objective += float(loss.detach())
                    scaler.scale(loss).backward()
                    backward_count += 1
                del sums, loss
            if relations and ((denom["reach"] and weights_r["reach"] > 0) or (denom["distance"] and weights_r["distance"] > 0)):
                with autocast(config, device):
                    sums = relation_loss_sums(model, relations[index], config, device, config["activation_checkpointing"])
                    loss = weighted_loss(sums, denom, weights_r)
                for key, value in sums.items():
                    sums_cpu[key] += float(value.detach())
                if loss is not None:
                    if probe and index == 0:
                        probe_r = gradient_probe(loss, model)
                    objective += float(loss.detach())
                    scaler.scale(loss).backward()
                    backward_count += 1
                del sums, loss
        record = {"epoch": epoch, "window": totals["windows"], "successful_steps_before": successful_steps,
                  "lambda_r3": weight, "window_denominators": denom,
                  "losses": {k: sums_cpu[k] / denom[k] if denom[k] else None for k in sums_cpu},
                  "objective": objective, "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups}}
        if probe_a is not None and probe_r is not None:
            record["sampled_anchor_gradients"] = {
                "nomad_norm": float(probe_a.norm()), "r3_norm": float(probe_r.norm()),
                "cosine": float(torch.nn.functional.cosine_similarity(probe_a[None], probe_r[None]))}
        if backward_count:
            did_update, norm = optimizer_update(model, optimizer, scaler, config["max_norm"])
            record.update(did_update=did_update, grad_norm_before_clip=norm, amp_scale=scaler.get_scale())
            if did_update:
                successful_steps += 1
                totals["updates"] += 1
                ema.step(model)
                objectives.append(objective)
            else:
                totals["amp_skips"] += 1
        else:
            record["did_update"] = False
            totals["unsupervised_windows"] += 1
        append_json(Path(folder) / "steps.jsonl", record)
        if config["print_log_freq"] > 0 and totals["windows"] % config["print_log_freq"] == 0:
            print(f"epoch={epoch} window={totals['windows']} successful_steps={successful_steps} objective={objective:.6g}")
    optimizer.zero_grad(set_to_none=True)
    totals["mean_successful_window_objective"] = float(np.mean(objectives)) if objectives else None
    totals["relation_cycles_restarted"] = relation_stream.cycle if relation_stream else 0
    totals["relation_repeated_samples"] = max(0, totals["relation_samples"] - len(relation_dataset)) if relation_dataset is not None else 0
    totals["r3_valid_ratio"] = ((totals["r3_positive_count"] + totals["r3_negative_count"]) / totals["relation_samples"] if totals["relation_samples"] else None)
    return totals, successful_steps


@torch.no_grad()
def collect_relations(model, dataset, config, device):
    model.eval()
    result = {key: [] for key in ("logit", "distance_m", "target", "target_distance_m", "reach_valid", "distance_valid", "temporal_distance", "row_index")}
    for batch in loader_for(dataset, config["eval_batch_size"], 0, 0, shuffle=False):
        predictions = score_rgb_pairs(model, batch["obs_image"].to(device), batch["goal_image"].to(device))
        values = {"logit": predictions["reachability_logit"], "distance_m": predictions["distance_m"],
                  "temporal_distance": predictions["temporal_distance"], "target": batch["reachable"],
                  "target_distance_m": batch["distance_m"], "reach_valid": batch["reach_valid"],
                  "distance_valid": batch["distance_valid"], "row_index": batch["row_index"]}
        for key, value in values.items():
            result[key].extend(value.detach().cpu().tolist())
    return result


@torch.no_grad()
def evaluate(model, action_dataset, relation_dataset, config, device, epoch):
    model.eval()
    result = {}
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    # Validation must not consume the training dropout/noise RNG stream.
    with isolated_evaluation_rng(cuda_devices):
        seed_all(stream_seed(config["seed"], epoch, "evaluation"))
        if action_dataset is not None:
            scheduler = make_noise_scheduler(config)
            for mode in ("random", "goal", "goal_free"):
                # Keep NumPy goal sampling identical across the three modes.
                seed_all(stream_seed(config["seed"], epoch, "eval_action"))
                numerator_a = numerator_t = denominator_a = denominator_t = 0.0
                sampler_metrics = []
                for index, batch in enumerate(loader_for(action_dataset, config["eval_batch_size"], 0,
                                                  stream_seed(config["seed"], epoch, "eval_action"), shuffle=False)):
                    masks = ((torch.rand(len(batch[0])) < config["goal_mask_prob"]).long() if mode == "random" else
                             torch.full((len(batch[0]),), int(mode == "goal_free"), dtype=torch.long))
                    sums = action_loss_sums(model, batch, masks, scheduler, config, device)
                    denom = window_denominators([batch], [masks], [])
                    numerator_a += float(sums["action"])
                    numerator_t += float(sums["temporal"])
                    denominator_a += denom["action"]
                    denominator_t += denom["temporal"]
                    if mode != "random" and index < config.get("eval_sampling_batches", 1) and (batch[6] > 0).any():
                        valid = batch[6] > 0
                        condition = model("vision_encoder", obs_img=normalize_rgb(batch[0][valid].to(device)),
                                          goal_img=normalize_rgb(batch[1][valid].to(device)), input_goal_mask=masks[valid].to(device))
                        samples = sample_conditioned_actions(model, condition, config, config.get("eval_action_samples", 4))
                        target = batch[2][valid].to(device) * config["action_scale_m"]
                        ade = (samples - target[:, None]).norm(dim=-1).mean(dim=-1)
                        sampler_metrics.append({"mean_ADE_m": float(ade.mean()), "min_ADE_m": float(ade.min(dim=1).values.mean()),
                                                "sample_spread_m": float((samples - samples.mean(dim=1, keepdim=True)).norm(dim=-1).mean())})
                result[mode] = {"diffusion_loss": numerator_a / denominator_a if denominator_a else None,
                                "temporal_loss": numerator_t / denominator_t if denominator_t and mode != "goal_free" else None}
                if sampler_metrics:
                    result[mode].update({key: float(np.mean([m[key] for m in sampler_metrics])) for key in sampler_metrics[0]})
        if relation_dataset is not None and len(relation_dataset):
            collected = collect_relations(model, relation_dataset, config, device)
            valid = np.array(collected["reach_valid"], bool)
            probabilities = torch.sigmoid(torch.tensor(collected["logit"], dtype=torch.float64)).numpy()
            result["relation"] = binary_metrics(probabilities[valid], np.array(collected["target"])[valid])
            result["relation"].update(distance_metrics(collected, config.get("r3", {}).get("lambda_distance", 1.) > 0))
    return result


def run_habitat_scratch(config):
    config = copy.deepcopy(config)
    if config.get("load_run") or config.get("pretrained_checkpoint") or config.get("init_from"):
        raise ValueError("habitat_scratch forbids pretrained initialization; use resume_from only for complete v2 runs")
    if config["model_type"] != "nomad" or config["vision_encoder"] != "nomad_vint" or config.get("learn_angle") or not config.get("normalize", True):
        raise ValueError("Habitat scratch v1 requires normalized XY NoMaD_ViNT")
    defaults = {"batch_size": 4, "relation_batch_size": 4, "eval_batch_size": 4,
                "gradient_accumulation_steps": 4, "amp": True, "activation_checkpointing": False,
                "max_norm": 1., "warmup_epochs": 4, "num_workers": 4, "seed": 0,
                "weight_decay": .01, "print_log_freq": 50, "checkpoint_every": 10}
    for key, value in defaults.items():
        config.setdefault(key, value)
    for key in ("batch_size", "relation_batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "seed", "warmup_epochs", "checkpoint_every"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "max_norm"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(float(config["weight_decay"])) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.get("optimizer", "adamw").lower() != "adamw" or not config.get("train", True):
        raise ValueError("habitat_scratch uses AdamW training; use the separate evaluation command for evaluation")
    if config.get("scheduler") not in (None, "linear_warmup_cosine"):
        raise ValueError("habitat_scratch uses scheduler: linear_warmup_cosine")
    config["scheduler"] = "linear_warmup_cosine"
    if config.get("train_subset", 1.) != 1.:
        raise ValueError("An epoch must traverse the whole action training split")
    if not 0 < config["goal_mask_prob"] < 1 or not 0 < float(config["alpha"]) < 1:
        raise ValueError("Scratch joint training requires 0 < goal_mask_prob, alpha < 1")
    for key, lo, hi in (("distance", "min_dist_cat", "max_dist_cat"), ("action", "min_dist_cat", "max_dist_cat")):
        bounds = config[key]
        if any(type(bounds[x]) is not int for x in (lo, hi)) or not 0 <= bounds[lo] <= bounds[hi]:
            raise ValueError(f"Invalid nonnegative integer {key} bounds")
    if config["distance"]["max_dist_cat"] < max(1, config["action"]["min_dist_cat"]):
        raise ValueError("Action supervision is impossible with these distance/action bounds")
    if config.get("datasets"):
        raise ValueError("habitat_scratch uses simulation.manifest, not public datasets")
    if config["epochs"] <= config["warmup_epochs"] or config["warmup_epochs"] < 1:
        raise ValueError("Set 0 < warmup_epochs < epochs")
    gpu_ids = config.get("gpu_ids", [0])
    gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
    if len(gpu_ids) != 1:
        raise ValueError("The scratch trainer supports one GPU")
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    if config["amp"] and device.type != "cuda":
        raise ValueError("FP16 AMP requires CUDA; set amp=false for CPU checks")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bundle = load_manifest(config["simulation"]["manifest"], config["context_size"])
    stats = load_action_statistics(config["simulation"]["action_statistics"], bundle, config["context_size"], config["len_traj_pred"])
    config["action_stats"] = validate_action_stats(stats["action_stats"])
    config["action_scale_m"] = bundle["manifest"]["action_scale_m"]
    config["action_time_step_s"] = bundle["manifest"]["action_time_step_s"]
    config["observation_stride"] = bundle["manifest"]["waypoint_spacing"]
    config.setdefault("r3", {})["label_definition"] = bundle["manifest"]["label_definition"]
    enabled = config["r3"].get("enabled", False)
    if enabled:
        r3 = config["r3"]
        for key in ("lambda_reach", "lambda_distance"):
            value = float(r3.get(key, 1.))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"r3.{key} must be finite and nonnegative")
        if r3.get("lambda_reach", 1.) <= 0:
            raise ValueError("R3 requires reachability supervision")
        if type(r3.get("warmup_steps", 500)) is not int or r3.get("warmup_steps", 500) < 0:
            raise ValueError("r3.warmup_steps must be a nonnegative integer")
        relation_weight(0, r3.get("lambda_start", .1), r3.get("lambda_end", 1.), r3.get("warmup_steps", 500))
        if not math.isfinite(float(r3.get("distance_beta", .1))) or r3.get("distance_beta", .1) <= 0:
            raise ValueError("r3.distance_beta must be finite and positive")
        if r3.get("lambda_distance", 1.) == 0 and r3.get("selector", {}).get("use_distance", True):
            raise ValueError("R-only training must disable distance-based selection")
        rows = bundle["relations"]["train"]
        if not rows or not any(row["reachable"] is not None for row in rows):
            raise ValueError("R3 training requires known relation labels")
        if {row["reachable"] for row in rows if row["reachable"] is not None} != {0, 1}:
            raise ValueError("R3 training requires both reachable and certified negative examples")
        if config["r3"].get("lambda_distance", 1.) > 0 and not any(row["distance_valid"] for row in rows):
            raise ValueError("R3 distance loss requires verified shortest positive paths")
    folder = Path(config["project_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    # Common parameters are constructed before the new head, for matched seeds.
    model = build_model(config).to(device)
    initial_common_sha256 = weights_sha256(model, exclude_prefixes=("goal_relation_head.",))
    optimizer = make_optimizer(model, config)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, end_factor=1., total_iters=config["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] - config["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[config["warmup_epochs"]])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    ema = EMAModel(model=model, power=.75)
    noise_scheduler = make_noise_scheduler(config)
    action_train = build_action_dataset(bundle, "train", config)
    action_val = build_action_dataset(bundle, "validation", config)
    if not len(action_train):
        raise ValueError("No action windows: check trajectory length, spacing and context")
    relation_train = GoalRelationDataset(bundle["relations"]["train"], bundle["root"], config["image_size"]) if enabled else None
    relation_val = GoalRelationDataset(bundle["relations"]["validation"], bundle["root"], config["image_size"]) if enabled else None
    keys = ("seed", "epochs", "lr", "weight_decay", "scheduler", "batch_size", "relation_batch_size", "gradient_accumulation_steps",
            "amp", "activation_checkpointing", "max_norm", "warmup_epochs", "num_workers", "alpha", "goal_mask_prob",
            "distance", "action", "context_size", "len_traj_pred", "action_stats", "action_scale_m", "action_time_step_s", "observation_stride", "r3")
    spec = {key: copy.deepcopy(config.get(key)) for key in keys}
    spec.update(initialization="scratch", epoch_basis="action_dataset", sampler_version=1,
                normalization="window_sums; nomad_count+0.01B; relation_valid_count", lr_start_factor=.1)
    epoch_start = successful_steps = 0
    if config.get("resume_from"):
        saved = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
        epoch_start, successful_steps = restore_sim_checkpoint(saved, model, ema, optimizer, scheduler, scaler, spec, bundle["fingerprint"])
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (folder / "dataset_manifest.json").write_text(json.dumps({"manifest": bundle["manifest"],
                                                            "files": bundle["file_hashes"]}, indent=2, allow_nan=False))
    compatibility_only = [n for n, _ in model.named_parameters() if ".sa_layer." in n or "._fc." in n]
    (folder / "provenance.json").write_text(json.dumps({"data_fingerprint": bundle["fingerprint"],
        "initial_common_weights_sha256": initial_common_sha256,
        "training_spec": spec, "compatibility_parameters_not_used_in_forward": compatibility_only,
        "resume_from": config.get("resume_from"), "torch_version": torch.__version__}, indent=2))
    for epoch in range(epoch_start, config["epochs"]):
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics, successful_steps = train_epoch(model, ema, optimizer, scaler, noise_scheduler,
            action_train, relation_train, config, device, epoch, successful_steps, folder)
        train_metrics["wall_time_s"] = time.monotonic() - started
        train_metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        validation = evaluate(ema.averaged_model, action_val, relation_val, config, device, epoch)
        if train_metrics["updates"]:
            scheduler.step()
        metrics = {"epoch": epoch, "successful_steps": successful_steps, "train": train_metrics,
                   "validation": validation, "next_lr": optimizer.param_groups[0]["lr"],
                   "epoch_wall_time_s": time.monotonic() - started}
        append_json(folder / "metrics.jsonl", metrics)
        saved = make_sim_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, successful_steps,
                                    spec, bundle["fingerprint"], config,
                                    {"manifest": bundle["manifest"], "files": bundle["file_hashes"]})
        save_checkpoint(saved, str(folder / "checkpoint_latest.pth"))
        if (config["checkpoint_every"] and (epoch + 1) % config["checkpoint_every"] == 0) or epoch + 1 == config["epochs"]:
            save_checkpoint(saved, str(folder / f"checkpoint_{epoch}.pth"))
        print(f"Epoch {epoch} metrics: {finite_json(metrics)}")

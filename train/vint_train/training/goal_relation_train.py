"""Loss sums and window denominators for scratch multi-task training."""

import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from vint_train.models.action_codec import normalize_actions, normalize_rgb


def relation_weight(successful_steps, start=0.1, end=1.0, warmup_steps=500):
    if not all(math.isfinite(x) for x in (start, end)) or not 0 < start <= end or warmup_steps < 0:
        raise ValueError("R3 loss warmup must start positive and be nondecreasing")
    return end if warmup_steps == 0 else start + (end - start) * min(1, successful_steps / warmup_steps)


def window_denominators(action_batches, goal_masks, relation_batches):
    b = sum(len(batch[0]) for batch in action_batches)
    action_count = sum(int((batch[6] > 0).sum()) for batch in action_batches)
    temporal_count = sum(int((mask == 0).sum()) for mask in goal_masks)
    r_count = sum(int(batch["reach_valid"].sum()) for batch in relation_batches)
    d_count = sum(int(batch["distance_valid"].sum()) for batch in relation_batches)
    return {"action": action_count + .01 * b, "temporal": temporal_count + .01 * b,
            "reach": r_count, "distance": d_count, "action_count": action_count,
            "temporal_count": temporal_count, "action_samples": b}


def model_call(model, name, activation_checkpointing=False, **inputs):
    if activation_checkpointing and model.training and torch.is_grad_enabled():
        keys = tuple(inputs)
        def forward(*values):
            return model(name, **dict(zip(keys, values)))
        # Non-reentrant checkpointing also works for RGB inputs that do not
        # themselves require gradients. Preserve dropout/drop-connect RNG.
        return checkpoint(forward, *inputs.values(), use_reentrant=False, preserve_rng_state=True)
    return model(name, **inputs)


def action_loss_sums(model, batch, goal_mask, scheduler, config, device, activation_checkpointing=False):
    obs, goal, actions, distance, _, _, action_mask = batch
    mask = goal_mask.to(device)
    condition = model_call(model, "vision_encoder", activation_checkpointing,
                           obs_img=normalize_rgb(obs.to(device)), goal_img=normalize_rgb(goal.to(device)),
                           input_goal_mask=mask)
    visible, valid_action = mask == 0, action_mask.to(device) > 0
    zero = condition.new_zeros((), dtype=torch.float32)
    temporal = zero
    if visible.any():
        pred = model("dist_pred_net", obsgoal_cond=condition).flatten().float()
        temporal = F.mse_loss(pred[visible], distance.to(device).float()[visible], reduction="sum")
    action = zero
    if valid_action.any():
        targets = normalize_actions(actions.to(device), config["action_stats"])
        noise = torch.randn_like(targets)
        steps = torch.randint(0, scheduler.config.num_train_timesteps, (len(targets),), device=device)
        noisy = scheduler.add_noise(targets, noise, steps)
        pred = model_call(model, "noise_pred_net", activation_checkpointing,
                          sample=noisy, timestep=steps, global_cond=condition)
        per_sample = (pred.float() - noise).square().mean(dim=(1, 2))
        action = per_sample[valid_action].sum()
    return {"action": action, "temporal": temporal}


def relation_loss_sums(model, batch, config, device, activation_checkpointing=False):
    masks = torch.zeros(len(batch["obs_image"]), device=device, dtype=torch.long)
    condition = model_call(model, "vision_encoder", activation_checkpointing,
                           obs_img=normalize_rgb(batch["obs_image"].to(device)),
                           goal_img=normalize_rgb(batch["goal_image"].to(device)), input_goal_mask=masks)
    output = model("goal_relation_head", obsgoal_cond=condition)
    valid_r, valid_d = batch["reach_valid"].to(device), batch["distance_valid"].to(device)
    zero = condition.new_zeros((), dtype=torch.float32)
    reach, distance = zero, zero
    # Index before reducing: NaN padding must never be multiplied by zero.
    if valid_r.any():
        reach = F.binary_cross_entropy_with_logits(output["reachability_logit"][valid_r],
                                                   batch["reachable"].to(device)[valid_r], reduction="sum")
    if valid_d.any():
        bound = config["r3"]["label_definition"]["max_path_length_m"]
        distance = F.smooth_l1_loss(output["distance_m"][valid_d] / bound,
                                   batch["distance_m"].to(device)[valid_d] / bound,
                                   beta=config["r3"].get("distance_beta", .1), reduction="sum")
    return {"reach": reach, "distance": distance}


def weighted_loss(sums, denominators, weights):
    terms = [sums[key] * weight / denominators[key] for key, weight in weights.items()
             if weight > 0 and denominators[key] > 0 and sums[key].requires_grad]
    return sum(terms) if terms else None


def optimizer_update(model, optimizer, scaler, max_grad_norm):
    """Return did_update and pre-clip norm; never inspect optimizer.step's return."""
    old_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        raise FloatingPointError("Non-finite full-precision gradient")
    # With standard GradScaler growth/backoff, an overflow strictly reduces scale.
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= old_scale, float(norm)

"""RGB-only model operations, independent of simulator, ROS and training loggers."""

import math
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vint_train.models.action_codec import decode_actions, normalize_rgb


def make_noise_scheduler(config):
    return DDPMScheduler(num_train_timesteps=config["num_diffusion_iters"],
                         beta_schedule="squaredcos_cap_v2", clip_sample=True,
                         prediction_type="epsilon")


@torch.no_grad()
def score_rgb_pairs(model, obs_images, goal_images, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    masks = torch.zeros(len(obs_images), dtype=torch.long, device=obs_images.device)
    condition = model("vision_encoder", obs_img=normalize_rgb(obs_images),
                      goal_img=normalize_rgb(goal_images), input_goal_mask=masks)
    result = {"condition": condition,
              "temporal_distance": model("dist_pred_net", obsgoal_cond=condition).flatten()}
    if hasattr(model, "goal_relation_head"):
        relation = model("goal_relation_head", obsgoal_cond=condition)
        result.update(relation)
        # Match the precision used by offline temperature/threshold fitting.
        logits = relation["reachability_logit"].double()
        result["probability"] = torch.where(torch.isfinite(logits), torch.sigmoid(logits / temperature),
                                             torch.full_like(logits, float("nan")))
    return result


@torch.no_grad()
def sample_conditioned_actions(model, condition, config, num_samples=1):
    """Returns [batch, samples, horizon, 2] in meters, using checkpoint bounds."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    scheduler = make_noise_scheduler(config)
    scheduler.set_timesteps(config["num_diffusion_iters"])
    expanded = condition.repeat_interleave(num_samples, dim=0)
    action = torch.randn((len(expanded), config["len_traj_pred"], 2), device=condition.device)
    for step in scheduler.timesteps:
        noise = model("noise_pred_net", sample=action, timestep=step, global_cond=expanded)
        action = scheduler.step(model_output=noise, timestep=step, sample=action).prev_sample
    metric = decode_actions(action, config["action_stats"], config["action_scale_m"])
    return metric.reshape(len(condition), num_samples, config["len_traj_pred"], 2)

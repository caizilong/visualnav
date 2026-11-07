"""Action conversion shared by simulation training and RGB-only inference."""

import math

import torch


def validate_action_stats(stats):
    if not isinstance(stats, dict) or set(stats) != {"min", "max"}:
        raise ValueError("action_stats must contain min and max")
    if len(stats["min"]) != 2 or len(stats["max"]) != 2:
        raise ValueError("Expected two-dimensional action delta bounds")
    for lo, hi in zip(stats["min"], stats["max"]):
        if not all(math.isfinite(float(x)) for x in (lo, hi)) or float(hi) <= float(lo):
            raise ValueError("Every action dimension requires finite, nondegenerate bounds")
    return {key: [float(x) for x in stats[key]] for key in ("min", "max")}


def normalize_actions(actions, stats):
    """Normalized local waypoints -> DDPM delta targets in the training range."""
    actions = actions.float()
    delta = torch.diff(actions, dim=1, prepend=torch.zeros_like(actions[:, :1]))
    lo, hi = (actions.new_tensor(stats[key]) for key in ("min", "max"))
    return 2 * (delta - lo) / (hi - lo) - 1


def decode_actions(predicted_deltas, stats, action_scale_m=1.0):
    """DDPM output -> local XY waypoints, in meters when scale is supplied."""
    predicted_deltas = predicted_deltas.float()
    lo, hi = (predicted_deltas.new_tensor(stats[key]) for key in ("min", "max"))
    deltas = (predicted_deltas + 1) * 0.5 * (hi - lo) + lo
    return deltas.cumsum(dim=1) * action_scale_m


def normalize_rgb(images):
    """Accept RGB or channel-stacked RGB tensors, before ImageNet normalization."""
    if images.ndim != 4 or images.shape[1] % 3 or not images.is_floating_point():
        raise ValueError("Expected [B, 3 * frames, height, width] RGB")
    mean = images.new_tensor([0.485, 0.456, 0.406]).repeat(images.shape[1] // 3)[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225]).repeat(images.shape[1] // 3)[None, :, None, None]
    return (images - mean) / std

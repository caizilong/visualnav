"""RGB-conditioned local reachability and conditional metric path length."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GoalRelationHead(nn.Module):
    def __init__(self, embedding_dim, max_path_length_m, hidden_dims=(128, 64)):
        super().__init__()
        if not math.isfinite(max_path_length_m) or max_path_length_m <= 0:
            raise ValueError("max_path_length_m must be finite and positive")
        if not hidden_dims or any(type(d) is not int or d <= 0 for d in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        layers = []
        previous = embedding_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.shared = nn.Sequential(*layers)
        self.reachability = nn.Linear(previous, 1)
        self.distance = nn.Linear(previous, 1)
        self.register_buffer("distance_scale_m", torch.tensor(float(max_path_length_m)))

    def forward(self, obsgoal_cond):
        hidden = self.shared(obsgoal_cond)
        # Compute the physical output in FP32 even under autocast.
        return {
            "reachability_logit": self.reachability(hidden).squeeze(-1).float(),
            "distance_m": self.distance_scale_m * F.softplus(self.distance(hidden).squeeze(-1).float()),
        }

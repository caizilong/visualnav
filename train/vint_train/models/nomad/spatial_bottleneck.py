"""R1 spatial tokens supervised by the existing demonstration waypoints.

Tokens depend only on current RGB features. The auxiliary distribution depends
on tokens and the goal-masked condition, and describes demonstration visitation,
not occupancy, collision risk or physical traversability.
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


def resolve_r1_config(config):
    """Canonical architecture settings, including shape-invisible ablations."""
    if type(config.get("enabled", False)) is not bool:
        raise ValueError("r1.enabled must be a boolean")
    if not config.get("enabled", False):
        return None
    resolved = {"enabled": True}
    for key, default in (("num_sectors", 6), ("num_radial_bins", 4),
                         ("token_dim", 64), ("num_attention_heads", 4)):
        value = config.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"r1.{key} must be a positive integer")
        resolved[key] = value
    if resolved["token_dim"] % resolved["num_attention_heads"]:
        raise ValueError("r1.token_dim must be divisible by num_attention_heads")
    mode = config.get("mode", "bottleneck")
    if mode not in ("bottleneck", "auxiliary"):
        raise ValueError("r1.mode must be bottleneck or auxiliary")
    if config.get("fusion", "shared") != "shared":
        raise ValueError("R1 uses a shared condition for temporal, R3 and diffusion heads")
    resolved.update(mode=mode, fusion="shared")
    representation = config.get("representation", "query_tokens")
    if representation not in ("query_tokens", "pooled_spatial", "global_condition"):
        raise ValueError("r1.representation must be query_tokens, pooled_spatial or global_condition")
    if representation == "global_condition" and mode != "auxiliary":
        raise ValueError("The global-condition control is auxiliary-only")
    resolved["representation"] = representation
    from vint_train.data.trajectory_regions import validate_definition
    resolved["discretization"] = validate_definition(config.get("discretization"))
    for key in ("num_sectors", "num_radial_bins"):
        if resolved[key] != resolved["discretization"][key]:
            raise ValueError(f"R1 model and trajectory discretization disagree on {key}")
    trained = config.get("visitation_head_trained", True)
    if type(trained) is not bool:
        raise ValueError("r1.visitation_head_trained must be a boolean")
    resolved["visitation_head_trained"] = trained
    return resolved


class SpatialBottleneck(nn.Module):
    """Six learned queries read spatial RGB features before global pooling.

    One query predicts one sector's radial visitation logits. Image coordinates
    distinguish spatial keys; they are not metric robot-frame coordinates.
    Token extraction is goal-independent; the readout and visitation head see
    the goal-masked condition. Targets never enter either operation.
    """

    def __init__(self, feature_dim, condition_dim, num_sectors=6,
                 num_radial_bins=4, token_dim=64, num_attention_heads=4,
                 representation="query_tokens"):
        super().__init__()
        for value in (feature_dim, condition_dim, num_sectors, num_radial_bins, token_dim, num_attention_heads):
            if type(value) is not int or value < 1:
                raise ValueError("Spatial bottleneck dimensions must be positive integers")
        if token_dim % num_attention_heads:
            raise ValueError("token_dim must be divisible by num_attention_heads")
        self.num_sectors = num_sectors
        self.num_radial_bins = num_radial_bins
        self.token_dim = token_dim
        if representation not in ("query_tokens", "pooled_spatial"):
            raise ValueError("Spatial encoding requires query_tokens or pooled_spatial")
        self.representation = representation
        self.spatial_projection = nn.Conv2d(feature_dim, token_dim, kernel_size=1)
        self.position_projection = nn.Linear(2, token_dim)
        self.spatial_norm = nn.LayerNorm(token_dim)
        if representation == "query_tokens":
            self.queries = nn.Parameter(torch.empty(num_sectors, token_dim))
            nn.init.normal_(self.queries, std=0.02)
            self.query_norm = nn.LayerNorm(token_dim)
            self.spatial_attention = nn.MultiheadAttention(token_dim, num_attention_heads,
                                                           dropout=0.0, batch_first=True)
        else:
            # Fixed image-grid pooling; no learned extraction queries/attention.
            # For six slots this is 2 x 3. Slots are not metric polar regions.
            rows = math.isqrt(num_sectors)
            while num_sectors % rows:
                rows -= 1
            self.pool_shape = (rows, num_sectors // rows)
        self.ff_norm = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(nn.Linear(token_dim, 2 * token_dim), nn.GELU(),
                                nn.Linear(2 * token_dim, token_dim))
        self.token_norm = nn.LayerNorm(token_dim)
        self.visitation_context = nn.Sequential(nn.LayerNorm(condition_dim),
                                               nn.Linear(condition_dim, token_dim))
        self.visitation_head = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.GELU(),
                                            nn.Linear(token_dim, num_radial_bins))
        self.condition_query = nn.Sequential(nn.LayerNorm(condition_dim),
                                            nn.Linear(condition_dim, token_dim))
        self.readout_attention = nn.MultiheadAttention(token_dim, num_attention_heads,
                                                       dropout=0.0, batch_first=True)
        self.residual_projection = nn.Linear(token_dim, condition_dim)
        # At initialization the shared condition exactly equals the baseline.
        # Gradients reach the readout after this output projection starts moving.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def encode(self, spatial_features):
        """[B,C,Hf,Wf] -> [B,sectors,token_dim], using current-frame RGB only."""
        if spatial_features.ndim != 4:
            raise ValueError("R1 spatial features must have shape [B,C,H,W]")
        projected = self.spatial_projection(spatial_features)
        batch, _, height, width = projected.shape
        x = (torch.arange(width, device=projected.device, dtype=projected.dtype) + .5) * (2.0 / width) - 1
        y = (torch.arange(height, device=projected.device, dtype=projected.dtype) + .5) * (2.0 / height) - 1
        coordinates = torch.stack((x[None, :].expand(height, width),
                                   y[:, None].expand(height, width)), dim=-1).reshape(-1, 2)
        memory = projected.flatten(2).transpose(1, 2)
        memory = self.spatial_norm(memory + self.position_projection(coordinates)[None])
        if self.representation == "query_tokens":
            tokens = self.queries[None].expand(batch, -1, -1)
            attended, _ = self.spatial_attention(self.query_norm(tokens), memory, memory,
                                                 need_weights=False)
            tokens = tokens + attended
        else:
            grid = memory.transpose(1, 2).reshape(batch, self.token_dim, height, width)
            tokens = F.adaptive_avg_pool2d(grid, self.pool_shape).flatten(2).transpose(1, 2)
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return self.token_norm(tokens)

    def predict_visitation(self, tokens, condition):
        """[B,sectors,radial_bins] logits; softmax is over ALL spatial bins."""
        context = self.visitation_context(condition)[:, None].expand(-1, self.num_sectors, -1)
        return self.visitation_head(torch.cat((tokens, context), dim=-1))

    def fuse(self, condition, tokens):
        """Read spatial tokens with the goal-masked condition; retain its size."""
        query = self.condition_query(condition).unsqueeze(1)
        readout, _ = self.readout_attention(query, tokens, tokens, need_weights=False)
        return condition + self.residual_projection(readout[:, 0])


class GlobalVisitationHead(nn.Module):
    """Matched-supervision control: original global condition -> 24 logits.

    No spatial features, extraction queries or feedback to the policy condition.
    The auxiliary loss still updates the shared encoder through the condition.
    """

    representation = "global_condition"

    def __init__(self, condition_dim, num_sectors=6, num_radial_bins=4, token_dim=64):
        super().__init__()
        self.num_sectors, self.num_radial_bins = num_sectors, num_radial_bins
        self.visitation_head = nn.Sequential(nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, token_dim), nn.GELU(),
            nn.Linear(token_dim, num_sectors * num_radial_bins))

    def predict_visitation(self, tokens, condition):
        return self.visitation_head(condition).reshape(-1, self.num_sectors, self.num_radial_bins)

"""Paired scratch-experiment initialization, and random streams."""

import copy
import hashlib
from contextlib import contextmanager

import torch

from vint_train.models.factory import build_model
from vint_train.models.weights import weights_sha256


RNG_PROTOCOL = "epoch_streams_and_per_microbatch_branch_rng_v2"
EXTENSION_PREFIXES = ("goal_relation_head.", "vision_encoder.spatial_bottleneck.")


def stream_seed(seed, epoch, stream, cycle=0):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{stream}:{cycle}".encode()).digest()[:8], "big")


class DiffusionStreams:
    """Independent device generators for epsilon and DDPM time indices.

    Recreated from (seed, epoch) at epoch boundaries, including after resume.
    Evaluation gets a separate namespace. Never share them with sampling/dropout.
    """

    def __init__(self, seed, epoch, device, namespace="train"):
        self.noise = torch.Generator(device=device).manual_seed(stream_seed(seed, epoch, namespace + "/noise"))
        self.timesteps = torch.Generator(device=device).manual_seed(stream_seed(seed, epoch, namespace + "/timesteps"))

    def draw(self, targets, num_steps):
        noise = torch.randn(targets.shape, dtype=targets.dtype, device=targets.device, generator=self.noise)
        steps = torch.randint(num_steps, (len(targets),), device=targets.device, generator=self.timesteps)
        return noise, steps


@contextmanager
def branch_rng(seed, device):
    """Keep GOAL_RELATION dropout and stochastic depth out of the next action forward."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


def audit_scratch_initialization(model, config):
    """Fail if any common initial tensor differs from the same-seed baseline.

    Construct the reference on CPU without advancing training RNG. Neither
    optimizer state nor weights from another experiment are loaded.
    """
    reference_config = copy.deepcopy(config)
    reference_config.setdefault("spatial_bottleneck", {})["enabled"] = False
    reference_config.setdefault("goal_relation", {})["enabled"] = False
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
        reference = build_model(reference_config)
    expected = weights_sha256(reference)
    actual = weights_sha256(model, exclude_prefixes=EXTENSION_PREFIXES)
    if actual != expected:
        raise ValueError("Scratch common initialization differs from the same-seed NoMaD baseline")
    return {"initialization": "scratch", "common_initialization_verified": True,
            "initial_common_weights_sha256": actual, "reference_common_weights_sha256": expected,
            "initial_spatial_bottleneck_weights_sha256": (weights_sha256(model.vision_encoder.spatial_bottleneck)
                                          if hasattr(model.vision_encoder, "spatial_bottleneck") else None),
            "initial_goal_relation_weights_sha256": (weights_sha256(model.goal_relation_head)
                                          if hasattr(model, "goal_relation_head") else None)}

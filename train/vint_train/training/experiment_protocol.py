"""Paired scratch-experiment initialization, random streams and gradient probes."""

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
    """Keep R3 dropout and stochastic depth out of the next action forward."""
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
    reference_config.setdefault("r1", {})["enabled"] = False
    reference_config.setdefault("r3", {})["enabled"] = False
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
        reference = build_model(reference_config)
    expected = weights_sha256(reference)
    actual = weights_sha256(model, exclude_prefixes=EXTENSION_PREFIXES)
    if actual != expected:
        raise ValueError("Scratch common initialization differs from the same-seed NoMaD baseline")
    return {"initialization": "scratch", "common_initialization_verified": True,
            "initial_common_weights_sha256": actual, "reference_common_weights_sha256": expected,
            "initial_r1_weights_sha256": (weights_sha256(model.vision_encoder.spatial_bottleneck)
                                          if hasattr(model.vision_encoder, "spatial_bottleneck") else None),
            "initial_r3_weights_sha256": (weights_sha256(model.goal_relation_head)
                                          if hasattr(model, "goal_relation_head") else None)}


def gradient_probe(loss, model):
    """Separate shared, direct spatial and feedback routes; None means unused."""
    if loss is None:
        return None
    named = dict(model.named_parameters())
    last_layer = len(model.vision_encoder.sa_encoder.layers) - 1
    names = ["vision_encoder.obs_encoder._conv_head.weight", "vision_encoder.goal_encoder._conv_head.weight",
             "vision_encoder.compress_obs_enc.weight", f"vision_encoder.sa_encoder.layers.{last_layer}.linear2.weight",
             "vision_encoder.spatial_bottleneck.spatial_projection.weight",
             "vision_encoder.spatial_bottleneck.queries",
             "vision_encoder.spatial_bottleneck.residual_projection.weight"]
    spatial = getattr(model.vision_encoder, "spatial_bottleneck", None)
    if spatial is not None:
        names.append(f"vision_encoder.spatial_bottleneck.visitation_head.{len(spatial.visitation_head)-1}.weight")
    names = [name for name in names if name in named]
    gradients = torch.autograd.grad(loss, [named[name] for name in names], retain_graph=True, allow_unused=True)
    return {name: (None if value is None else value.detach().float().flatten().cpu())
            for name, value in zip(names, gradients)}


def gradient_report(probes):
    names = sorted({name for probe in probes.values() if probe is not None for name in probe})
    result = {"scope": "weighted losses; first microbatch only; before scaler and clipping", "parameters": {}}
    for name in names:
        values = {task: probe.get(name) if probe is not None else None for task, probe in probes.items()}
        record = {task: {"loss_available": probes[task] is not None, "connected": value is not None,
                         "norm": float(value.norm()) if value is not None else None,
                         "finite": bool(torch.isfinite(value).all()) if value is not None else None}
                  for task, value in values.items()}
        record["cosines"] = {}
        for left, right in (("nomad", "visitation"), ("nomad", "r3"), ("visitation", "r3")):
            a, b = values.get(left), values.get(right)
            comparable = (a is not None and b is not None and torch.isfinite(a).all()
                          and torch.isfinite(b).all() and a.norm() > 0 and b.norm() > 0)
            record["cosines"][left + "/" + right] = float(torch.nn.functional.cosine_similarity(a[None], b[None])) if comparable else None
        result["parameters"][name] = record
    return result

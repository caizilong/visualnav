"""Strict checkpoint loading without ROS or training/logging dependencies."""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import warnings

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from vint_train.models.factory import build_model


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_config(model):
    return deepcopy(getattr(unwrap_model(model), "_model_config", None))


def validate_checkpoint_config(model, checkpoint):
    saved = checkpoint.get("model_config") if isinstance(checkpoint, Mapping) else None
    if saved is None:
        warnings.warn("Legacy weights have no model_config metadata. Tensor keys and shapes "
                      "will be checked strictly, but attention heads, normalization and other "
                      "shape-independent settings must match the original training config.",
                      UserWarning, stacklevel=2)
        return
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint model_config must be a mapping")
    expected = model_config(model)
    if expected is None:
        raise ValueError("Cannot validate checkpoint configuration: construct the model with build_model")
    differences = [f"{key}: checkpoint={saved.get(key)!r}, requested={expected.get(key)!r}"
                   for key in sorted(set(saved) | set(expected))
                   if key not in saved or key not in expected or saved[key] != expected[key]]
    if differences:
        raise ValueError("Checkpoint model configuration mismatch: " + "; ".join(differences))


def _is_state_dict(value):
    return (isinstance(value, Mapping) and bool(value)
            and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()))


def checkpoint_state(checkpoint, model_type, *, use_ema=True):
    """Return weights and their origin; never fall back from missing EMA to raw."""
    if not isinstance(use_ema, bool):
        raise ValueError("use_ema must be a boolean")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state_dict or a checkpoint mapping")
    if "format_version" in checkpoint:
        if checkpoint["format_version"] != 1 or model_type != "nomad":
            raise ValueError(f"Unsupported checkpoint format_version={checkpoint['format_version']!r} "
                             f"for model_type={model_type!r}")
        if use_ema:
            ema = checkpoint.get("ema")
            state = ema.get("model") if isinstance(ema, Mapping) else None
            origin = "ema"
        else:
            state, origin = checkpoint.get("model"), "raw"
    elif _is_state_dict(checkpoint):
        state, origin = checkpoint, "state_dict (as saved; raw/EMA unspecified)"
    elif model_type in ("vint", "gnm") and "model" in checkpoint:
        saved_model = checkpoint["model"]
        state = unwrap_model(saved_model).state_dict() if isinstance(saved_model, nn.Module) else saved_model
        origin = "legacy model"
    else:
        raise ValueError("Unrecognized checkpoint structure; expected raw state_dict, "
                         "NoMaD format_version=1, or a ViNT/GNM model checkpoint")
    if not _is_state_dict(state):
        raise ValueError(f"Checkpoint {origin} weights are missing, empty or not a tensor state_dict")
    state_copy = OrderedDict(state)
    if hasattr(state, "_metadata"):
        state_copy._metadata = deepcopy(state._metadata)
    if all(key.startswith("module.") for key in state_copy):
        consume_prefix_in_state_dict_if_present(state_copy, "module.")
    return state_copy, origin


def load_state_dict_strict(model, state):
    """Check all keys and shapes before copying any tensors into the target model."""
    model = unwrap_model(model)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [f"{key}: {tuple(state[key].shape)} != {tuple(expected[key].shape)}"
                  for key in sorted(set(expected) & set(state))
                  if state[key].shape != expected[key].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError("Checkpoint parameters do not match the model: "
                           f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                           f"shape_mismatch={mismatched[:10]}")
    model.load_state_dict(state, strict=True)


def load_model_weights(model, checkpoint, model_type, *, use_ema=False):
    state, origin = checkpoint_state(checkpoint, model_type, use_ema=use_ema)
    validate_checkpoint_config(model, checkpoint)
    load_state_dict_strict(model, state)
    return origin


def weights_sha256(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(unwrap_model(model).state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_model(model_path, config, device=torch.device("cpu"), *, use_ema=True):
    """Load trusted local checkpoint files and return a fully loaded inference model."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    origin = load_model_weights(model, checkpoint, config["model_type"], use_ema=use_ema)
    fingerprint = weights_sha256(model)
    epoch = checkpoint.get("epoch", "unknown")
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint {model_path}: weights={origin}, epoch={epoch}, sha256={fingerprint}")
    return model

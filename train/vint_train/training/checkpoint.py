"""NoMaD checkpoints, including EMA progress and random-number generators."""

import os
import random
import warnings

import numpy as np
import torch

from vint_train.models.weights import (
    checkpoint_state, load_model_weights, load_state_dict_strict, model_config, unwrap_model,
)


EMA_FIELDS = ("optimization_step", "decay", "update_after_step", "inv_gamma",
              "power", "min_value", "max_value")


def scheduler_state(scheduler):
    if scheduler is None:
        return None
    state = scheduler.state_dict().copy()
    # warmup-scheduler 0.3 puts a live scheduler (and its optimizer) in state_dict.
    nested = state.pop("after_scheduler", None)
    return {"type": type(scheduler).__name__, "state": state,
            "after_scheduler": scheduler_state(nested)}


def restore_scheduler(scheduler, saved):
    if (scheduler is None) != (saved is None):
        raise ValueError("Resume configuration must use the checkpoint's scheduler setting")
    if scheduler is None:
        return
    if type(scheduler).__name__ != saved["type"]:
        raise ValueError("Resume scheduler type does not match the checkpoint")
    scheduler.load_state_dict(saved["state"])
    restore_scheduler(getattr(scheduler, "after_scheduler", None), saved["after_scheduler"])


def make_nomad_checkpoint(model, ema_model, optimizer, scheduler, epoch):
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return {
        "format_version": 1,
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "model_config": model_config(model),
        "ema": {"model": unwrap_model(ema_model.averaged_model).state_dict(),
                **{key: getattr(ema_model, key) for key in EMA_FIELDS}},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state(scheduler),
        "rng": rng,
    }


def restore_nomad_checkpoint(checkpoint, model, ema_model, optimizer, scheduler):
    """Restore full checkpoints; old state_dict files remain usable for warm starts."""
    load_model_weights(model, checkpoint, "nomad", use_ema=False)
    if "format_version" not in checkpoint:
        state, _ = checkpoint_state(checkpoint, "nomad", use_ema=False)
        load_state_dict_strict(ema_model.averaged_model, state)
        warnings.warn("Loaded legacy NoMaD weights only: epoch, optimizer, scheduler, "
                      "EMA progress and RNG state are unavailable; starting at epoch 0.")
        return 0
    ema_state, _ = checkpoint_state(checkpoint, "nomad", use_ema=True)
    load_state_dict_strict(ema_model.averaged_model, ema_state)
    for key in EMA_FIELDS:
        setattr(ema_model, key, checkpoint["ema"][key])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_scheduler(scheduler, checkpoint["scheduler"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return checkpoint["epoch"] + 1


def save_checkpoint(checkpoint, path):
    """Avoid leaving a partially written latest checkpoint if a run is interrupted."""
    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

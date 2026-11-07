"""Strict epoch-boundary checkpoints for scratch simulation experiments."""

import copy

from vint_train.models.weights import load_model_weights, load_state_dict_strict
from vint_train.data.simulation import digest_json
from .checkpoint import make_nomad_checkpoint, restore_scheduler, EMA_FIELDS


def parameter_groups(optimizer, model):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{"name": group["name"], "parameters": [names[id(p)] for p in group["params"]]}
            for group in optimizer.param_groups]


def scale_fingerprint(config):
    return digest_json({key: config[key] for key in ("action_stats", "action_scale_m", "action_time_step_s", "observation_stride")})


def make_sim_checkpoint(model, ema, optimizer, scheduler, scaler, epoch,
                        successful_steps, training_spec, data_fingerprint, inference_config, data_manifest):
    result = make_nomad_checkpoint(model, ema, optimizer, scheduler, epoch)
    result.update(format_version=2, training_mode="full_joint", initialization="scratch",
                  checkpoint_boundary="epoch", successful_steps=successful_steps,
                  scaler=scaler.state_dict(), parameter_groups=parameter_groups(optimizer, model),
                  training_spec=copy.deepcopy(training_spec), data_fingerprint=data_fingerprint,
                  scale_fingerprint=scale_fingerprint(inference_config),
                  data_manifest=copy.deepcopy(data_manifest),
                  inference_config=copy.deepcopy(inference_config))
    return result


def restore_sim_checkpoint(saved, model, ema, optimizer, scheduler, scaler, training_spec, data_fingerprint):
    import random
    import numpy as np
    import torch

    if (saved.get("format_version") != 2 or saved.get("initialization") != "scratch"
            or saved.get("training_mode") != "full_joint" or saved.get("checkpoint_boundary") != "epoch"):
        raise ValueError("Resume requires a complete scratch full_joint v2 epoch checkpoint; pretrained warm starts are not supported")
    if saved["training_spec"] != training_spec or saved["data_fingerprint"] != data_fingerprint:
        raise ValueError("Resume training/data specification differs from the checkpoint")
    if digest_json(saved["data_manifest"]) != data_fingerprint:
        raise ValueError("Checkpoint data manifest does not match its fingerprint")
    if saved.get("scale_fingerprint") != scale_fingerprint(training_spec):
        raise ValueError("Resume action-scale fingerprint differs from the checkpoint")
    if saved["parameter_groups"] != parameter_groups(optimizer, model):
        raise ValueError("Optimizer parameter names/order differ from the checkpoint")
    if saved["successful_steps"] != saved["ema"]["optimization_step"]:
        raise ValueError("EMA progress and successful optimizer steps disagree")
    load_model_weights(model, saved, "nomad", use_ema=False)
    load_state_dict_strict(ema.averaged_model, saved["ema"]["model"])
    for key in EMA_FIELDS:
        setattr(ema, key, saved["ema"][key])
    optimizer.load_state_dict(saved["optimizer"])
    restore_scheduler(scheduler, saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    rng = saved["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return saved["epoch"] + 1, saved["successful_steps"]

"""Model construction shared by training and deployment."""

from copy import deepcopy
from pathlib import Path
import math

import yaml


def resolve_model_config(config):
    """Keep only settings that affect inference, including shape-invisible ones."""
    common = ("model_type", "context_size", "len_traj_pred", "learn_angle", "image_size")
    resolved = {key: deepcopy(config[key]) for key in common}
    size = resolved["image_size"]
    resolved["image_size"] = [size, size] if isinstance(size, int) else list(size)
    resolved.update(
        normalize=config.get("normalize", True),
        context_type=config.get("context_type", "temporal"),
        goal_type=config.get("goal_type", "image"),
    )
    model_type = resolved["model_type"]
    if config.get("spatial_bottleneck", {}).get("enabled", False) and (
            model_type != "nomad" or config.get("vision_encoder") != "nomad_vint"):
        raise ValueError("SPATIAL_BOTTLENECK supports NoMaD with nomad_vint")
    if model_type == "gnm":
        keys = ("obs_encoding_size", "goal_encoding_size")
    elif model_type == "vint":
        keys = ("obs_encoder", "obs_encoding_size", "late_fusion",
                "mha_num_attention_heads", "mha_num_attention_layers", "mha_ff_dim_factor")
        resolved["goal_encoder"] = "efficientnet-b0"
    elif model_type == "nomad":
        keys = ("vision_encoder", "encoding_size", "mha_num_attention_heads",
                "mha_num_attention_layers", "down_dims", "cond_predict_scale",
                "num_diffusion_iters")
        encoder = config["vision_encoder"]
        if encoder == "nomad_vint":
            # Existing checkpoints were trained with b0 even if obs_encoder said otherwise.
            # Refuse that ambiguity rather than silently changing the architecture here.
            if config.get("obs_encoder", "efficientnet-b0") != "efficientnet-b0":
                raise ValueError("NoMaD baseline uses obs_encoder=efficientnet-b0; "
                                 "other backbones require the encoder support fix")
            resolved.update(obs_encoder="efficientnet-b0", goal_encoder="efficientnet-b0",
                            mha_ff_dim_factor=config["mha_ff_dim_factor"])
        elif encoder == "vit":
            resolved["patch_size"] = config.get("patch_size", 16)
        else:
            raise ValueError(f"Vision encoder {encoder!r} is not implemented")
        with (Path(__file__).parent.parent / "data/data_config.yaml").open() as stream:
            resolved["action_stats"] = yaml.safe_load(stream)["action_stats"]
        if "action_stats" in config:
            from vint_train.models.action_codec import validate_action_stats
            resolved["action_stats"] = validate_action_stats(config["action_stats"])
        if "action_scale_m" in config:
            for name in ("action_scale_m", "action_time_step_s"):
                value = float(config[name])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                resolved[name] = value
            stride = config.get("observation_stride", 1)
            if type(stride) is not int or stride < 1:
                raise ValueError("observation_stride must be a positive integer")
            resolved["observation_stride"] = stride
        goal_relation = config.get("goal_relation", {})
        if goal_relation.get("enabled", False):
            if encoder != "nomad_vint":
                raise ValueError("GOAL_RELATION v1 supports nomad_vint")
            labels = deepcopy(goal_relation["label_definition"])
            horizon = float(labels["max_path_length_m"])
            if not math.isfinite(horizon) or horizon <= 0:
                raise ValueError("GOAL_RELATION requires an explicit positive path-length bound")
            resolved["goal_relation"] = {"enabled": True, "label_definition": labels,
                              "hidden_dims": list(goal_relation.get("hidden_dims", [128, 64]))}
        if config.get("spatial_bottleneck", {}).get("enabled", False):
            from vint_train.models.nomad.spatial_bottleneck import resolve_spatial_bottleneck_config
            resolved["spatial_bottleneck"] = resolve_spatial_bottleneck_config(config["spatial_bottleneck"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    resolved.update({key: deepcopy(config[key]) for key in keys})
    if "down_dims" in resolved:
        resolved["down_dims"] = list(resolved["down_dims"])
    return resolved


def build_model(config):
    """Build the same architecture at train and inference time, with lazy imports."""
    c = resolve_model_config(config)
    if c["model_type"] == "gnm":
        from vint_train.models.gnm.gnm import GNM

        model = GNM(c["context_size"], c["len_traj_pred"], c["learn_angle"],
                    c["obs_encoding_size"], c["goal_encoding_size"])
    elif c["model_type"] == "vint":
        from vint_train.models.vint.vint import ViNT

        model = ViNT(**{key: c[key] for key in (
            "context_size", "len_traj_pred", "learn_angle", "obs_encoder",
            "obs_encoding_size", "late_fusion", "mha_num_attention_heads",
            "mha_num_attention_layers", "mha_ff_dim_factor")})
    else:
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        encoder_args = dict(
            obs_encoding_size=c["encoding_size"], context_size=c["context_size"],
            mha_num_attention_heads=c["mha_num_attention_heads"],
            mha_num_attention_layers=c["mha_num_attention_layers"],
        )
        if c["vision_encoder"] == "nomad_vint":
            encoder = NoMaD_ViNT(**encoder_args, obs_encoder=c["obs_encoder"],
                                 mha_ff_dim_factor=c["mha_ff_dim_factor"])
        else:
            from vint_train.models.vint.vit import ViT

            encoder = ViT(**encoder_args, image_size=c["image_size"], patch_size=c["patch_size"])
        model = NoMaD(
            vision_encoder=replace_bn_with_gn(encoder),
            noise_pred_net=ConditionalUnet1D(
                input_dim=2, global_cond_dim=c["encoding_size"], down_dims=c["down_dims"],
                cond_predict_scale=c["cond_predict_scale"]),
            dist_pred_net=DenseNetwork(embedding_dim=c["encoding_size"]),
        )
        # Construct common modules first: same seed gives the same NoMaD weights
        # for the scratch baseline and GOAL_RELATION. Never download pretrained encoders.
        if c.get("spatial_bottleneck", {}).get("enabled"):
            import torch
            from vint_train.models.nomad.spatial_bottleneck import SpatialBottleneck, GlobalVisitationHead

            # Isolate the additional CPU initialization from GOAL_RELATION's RNG stream.
            # Both SPATIAL_BOTTLENECK and SPATIAL_BOTTLENECK+GOAL_RELATION start their spatial module at this same point.
            with torch.random.fork_rng(devices=[]):
                shared = {key: c["spatial_bottleneck"][key] for key in ("num_sectors", "num_radial_bins", "token_dim")}
                if c["spatial_bottleneck"]["representation"] == "global_condition":
                    encoder.spatial_bottleneck = GlobalVisitationHead(condition_dim=c["encoding_size"], **shared)
                else:
                    encoder.spatial_bottleneck = SpatialBottleneck(
                        feature_dim=encoder.num_obs_features, condition_dim=c["encoding_size"],
                        num_attention_heads=c["spatial_bottleneck"]["num_attention_heads"],
                        representation=c["spatial_bottleneck"]["representation"], **shared)
            encoder.spatial_bottleneck_mode = c["spatial_bottleneck"]["mode"]
        if c.get("goal_relation", {}).get("enabled"):
            from vint_train.models.nomad.goal_relation_head import GoalRelationHead
            model.goal_relation_head = GoalRelationHead(
                c["encoding_size"], c["goal_relation"]["label_definition"]["max_path_length_m"],
                c["goal_relation"]["hidden_dims"],
            )
    model._model_config = c
    return model

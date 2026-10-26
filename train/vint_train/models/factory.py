"""Model construction shared by training and deployment."""

from copy import deepcopy
from pathlib import Path

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
    model._model_config = c
    return model

"""Validation for optional NoMaD CUDA BF16 autocast."""

import torch


def validate_bf16(device, use_bf16):
    if not isinstance(use_bf16, bool):
        raise ValueError("use_bf16 must be true or false (a YAML boolean)")
    if not use_bf16:
        return
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("use_bf16=true requires a CUDA GPU; set use_bf16=false for CPU training")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"{device} does not support native BF16; set use_bf16=false")

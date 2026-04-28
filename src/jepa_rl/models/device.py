from __future__ import annotations

import torch


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve a configured device string, preferring Apple Silicon MPS when available."""

    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch build")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_torch_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


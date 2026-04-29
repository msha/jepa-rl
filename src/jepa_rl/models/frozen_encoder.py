"""Frozen JEPA encoder loading utilities."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.utils.config import ProjectConfig


class FrozenEncoder(nn.Module):
    """No-grad wrapper around a pretrained JEPA encoder."""

    def __init__(self, encoder: ConvEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze()

    def freeze(self) -> None:
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)


def build_conv_encoder_from_config(config: ProjectConfig) -> ConvEncoder:
    if config.world_model.encoder.type != "conv":
        raise ValueError(
            "frozen_jepa_dqn currently supports only conv JEPA encoders; "
            f"got {config.world_model.encoder.type!r}"
        )
    return ConvEncoder(
        input_channels=config.observation.input_channels,
        hidden_channels=list(config.world_model.encoder.hidden_channels),
        latent_dim=config.world_model.latent_dim,
    )


def load_frozen_encoder(
    config: ProjectConfig,
    checkpoint_path: Path,
    *,
    strict: bool = True,
) -> FrozenEncoder:
    """Load and freeze the online encoder from a JEPA checkpoint."""

    encoder = build_conv_encoder_from_config(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(_extract_encoder_state_dict(state, checkpoint_path), strict=strict)
    return FrozenEncoder(encoder)


def _extract_encoder_state_dict(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """Return ConvEncoder weights from supported JEPA checkpoint layouts."""

    raw_state = _select_model_state(checkpoint)
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"{checkpoint_path} does not contain a model state dict")

    state = dict(raw_state)
    prefixed = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    if prefixed:
        return prefixed

    module_prefixed = {
        key.removeprefix("module.encoder."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("module.encoder.")
    }
    if module_prefixed:
        return module_prefixed

    raw_encoder_keys = ("backbone.", "pool.", "project.")
    if any(isinstance(key, str) and key.startswith(raw_encoder_keys) for key in state):
        return {key: value for key, value in state.items() if isinstance(key, str)}

    raise ValueError(
        f"No encoder weights found in {checkpoint_path}. Expected keys like "
        "'encoder.backbone.0.weight' or raw ConvEncoder keys."
    )


def _select_model_state(checkpoint: Mapping[str, Any]) -> Any:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "jepa_model" in checkpoint:
        return checkpoint["jepa_model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

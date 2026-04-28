from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Small Atari-style convolutional encoder for low-resolution browser frames."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: list[int],
        latent_dim: int,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must not be empty")

        default_kernels = [8, 4, 3, 3]
        default_strides = [4, 2, 1, 1]
        layers: list[nn.Module] = []
        in_channels = input_channels
        for index, out_channels in enumerate(hidden_channels):
            kernel = default_kernels[min(index, len(default_kernels) - 1)]
            stride = default_strides[min(index, len(default_strides) - 1)]
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride),
                    nn.GroupNorm(1, out_channels),
                    activation(),
                ]
            )
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got shape {tuple(obs.shape)}")
        x = obs.float()
        if x.max() > 2:
            x = x / 255.0
        return self.project(self.pool(self.backbone(x)))


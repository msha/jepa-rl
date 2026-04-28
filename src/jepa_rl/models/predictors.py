from __future__ import annotations

import torch
from torch import nn


class ActionConditionedPredictor(nn.Module):
    """Transformer predictor for p[t+k] = f(z[t], actions[t:t+k], k)."""

    def __init__(
        self,
        *,
        latent_dim: int,
        num_actions: int,
        action_embed_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        horizons: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        if not horizons:
            raise ValueError("horizons must not be empty")
        self.horizons = tuple(sorted(set(horizons)))
        self.max_horizon = max(self.horizons)
        self.context_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.action_proj = nn.Linear(action_embed_dim, hidden_dim)
        self.horizon_embed = nn.Embedding(self.max_horizon + 1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> dict[int, torch.Tensor]:
        if actions.ndim != 2:
            raise ValueError(f"expected actions [B,T], got shape {tuple(actions.shape)}")
        if actions.shape[1] < self.max_horizon:
            raise ValueError(
                f"actions length {actions.shape[1]} shorter than max horizon {self.max_horizon}"
            )
        batch = context.shape[0]
        context_token = self.context_proj(context).unsqueeze(1)
        action_tokens = self.action_proj(self.action_embed(actions[:, : self.max_horizon]))
        predictions: dict[int, torch.Tensor] = {}
        for horizon in self.horizons:
            horizon_ids = torch.full(
                (batch, 1), horizon, dtype=torch.long, device=context.device
            )
            horizon_token = self.horizon_embed(horizon_ids)
            tokens = torch.cat([context_token, action_tokens[:, :horizon], horizon_token], dim=1)
            encoded = self.transformer(tokens)
            predictions[horizon] = self.output(encoded[:, -1])
        return predictions


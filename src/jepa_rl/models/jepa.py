from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.models.losses import (
    covariance_loss,
    effective_rank,
    normalized_prediction_loss,
    variance_loss,
)
from jepa_rl.models.predictors import ActionConditionedPredictor
from jepa_rl.utils.config import ProjectConfig


@dataclass(frozen=True)
class JepaWorldModelConfig:
    input_channels: int
    latent_dim: int
    hidden_channels: list[int]
    num_actions: int
    predictor_hidden_dim: int
    predictor_depth: int
    predictor_heads: int
    action_embed_dim: int
    horizons: list[int]
    lambda_var: float
    lambda_cov: float
    variance_floor: float

    @classmethod
    def from_project_config(cls, config: ProjectConfig) -> JepaWorldModelConfig:
        return cls(
            input_channels=config.observation.input_channels,
            latent_dim=config.world_model.latent_dim,
            hidden_channels=list(config.world_model.encoder.hidden_channels),
            num_actions=config.actions.num_actions,
            predictor_hidden_dim=config.world_model.predictor.hidden_dim,
            predictor_depth=config.world_model.predictor.depth,
            predictor_heads=config.world_model.predictor.num_heads,
            action_embed_dim=config.world_model.predictor.action_embed_dim,
            horizons=list(config.world_model.predictor.horizons),
            lambda_var=config.world_model.loss.lambda_var,
            lambda_cov=config.world_model.loss.lambda_cov,
            variance_floor=config.world_model.loss.variance_floor,
        )


@dataclass(frozen=True)
class JepaBatch:
    context_obs: torch.Tensor
    target_obs: dict[int, torch.Tensor]
    actions: torch.Tensor


class JepaWorldModel(nn.Module):
    """Action-conditioned JEPA world model with EMA target encoder."""

    def __init__(self, config: JepaWorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            latent_dim=config.latent_dim,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = ActionConditionedPredictor(
            latent_dim=config.latent_dim,
            num_actions=config.num_actions,
            action_embed_dim=config.action_embed_dim,
            hidden_dim=config.predictor_hidden_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            horizons=config.horizons,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(obs)

    @torch.no_grad()
    def update_target_encoder(self, tau: float) -> None:
        if not 0 <= tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

    def forward(self, batch: JepaBatch) -> dict[str, torch.Tensor]:
        context = self.encoder(batch.context_obs)
        predictions = self.predictor(context, batch.actions)
        prediction_losses = []
        metrics: dict[str, torch.Tensor] = {}
        target_latents = []
        for horizon, pred in predictions.items():
            if horizon not in batch.target_obs:
                raise ValueError(f"missing target observation for horizon {horizon}")
            target = self.encode_target(batch.target_obs[horizon])
            target_latents.append(target)
            loss = normalized_prediction_loss(pred, target)
            prediction_losses.append(loss)
            metrics[f"prediction_loss_h{horizon}"] = loss.detach()

        target_concat = torch.cat(target_latents, dim=0)
        pred_loss = torch.stack(prediction_losses).mean()
        var_loss = variance_loss(context, self.config.variance_floor) + variance_loss(
            target_concat, self.config.variance_floor
        )
        cov_loss = covariance_loss(context)
        total = pred_loss + self.config.lambda_var * var_loss + self.config.lambda_cov * cov_loss
        metrics.update(
            {
                "loss": total,
                "prediction_loss": pred_loss.detach(),
                "variance_loss": var_loss.detach(),
                "covariance_loss": cov_loss.detach(),
                "latent_std_mean": context.std(dim=0, unbiased=False).mean().detach(),
                "target_std_mean": target_concat.std(dim=0, unbiased=False).mean().detach(),
                "effective_rank": effective_rank(context).detach(),
            }
        )
        return metrics


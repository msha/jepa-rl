"""Frozen JEPA encoder + trainable DQN Q-head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.models.frozen_encoder import load_frozen_encoder
from jepa_rl.utils.config import ProjectConfig


class FrozenJepaQNetwork(nn.Module):
    """DQN Q-network with a frozen JEPA-pretrained encoder."""

    def __init__(
        self,
        *,
        encoder: ConvEncoder,
        latent_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...],
        dueling: bool = True,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if dueling:
            trunk_dims = hidden_dims[:-1]
            trunk_layers: list[nn.Module] = []
            in_dim = latent_dim
            for out_dim in trunk_dims:
                trunk_layers += [nn.Linear(in_dim, out_dim), activation()]
                in_dim = out_dim
            self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
            trunk_out_dim = trunk_dims[-1] if trunk_dims else latent_dim

            branch_dim = hidden_dims[-1]
            self.value_branch = nn.Sequential(
                nn.Linear(trunk_out_dim, branch_dim),
                activation(),
                nn.Linear(branch_dim, 1),
            )
            self.advantage_branch = nn.Sequential(
                nn.Linear(trunk_out_dim, branch_dim),
                activation(),
                nn.Linear(branch_dim, num_actions),
            )
            self._dueling = True
        else:
            layers: list[nn.Module] = []
            in_dim = latent_dim
            for out_dim in hidden_dims:
                layers += [nn.Linear(in_dim, out_dim), activation()]
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, num_actions))
            self.q_head = nn.Sequential(*layers)
            self._dueling = False

    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        if self._dueling:
            h = self.trunk(z)
            value = self.value_branch(h)
            advantage = self.advantage_branch(h)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(z)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def build_frozen_jepa_q_network(
    config: ProjectConfig,
    *,
    num_actions: int,
    jepa_checkpoint_path: Path,
) -> FrozenJepaQNetwork:
    """Build a FrozenJepaQNetwork from a pretrained JEPA checkpoint."""
    frozen_encoder = load_frozen_encoder(config, jepa_checkpoint_path)

    return FrozenJepaQNetwork(
        encoder=frozen_encoder.encoder,
        latent_dim=config.world_model.latent_dim,
        num_actions=num_actions,
        hidden_dims=config.agent.q_network.hidden_dims,
        dueling=config.agent.q_network.dueling,
    )

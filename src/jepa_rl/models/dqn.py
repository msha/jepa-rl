from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.utils.config import ProjectConfig


class QNetwork(nn.Module):
    def __init__(
        self,
        *,
        encoder: ConvEncoder,
        latent_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), activation()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.q_head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(obs))


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        *,
        encoder: ConvEncoder,
        latent_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.encoder = encoder

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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = self.trunk(z)
        value = self.value_branch(h)
        advantage = self.advantage_branch(h)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(config: ProjectConfig, *, num_actions: int) -> nn.Module:
    if config.agent.q_network.distributional:
        raise NotImplementedError("distributional DQN is not implemented in V1")
    encoder = ConvEncoder(
        input_channels=config.observation.input_channels,
        hidden_channels=list(config.world_model.encoder.hidden_channels),
        latent_dim=config.world_model.latent_dim,
    )
    cls = DuelingQNetwork if config.agent.q_network.dueling else QNetwork
    return cls(
        encoder=encoder,
        latent_dim=config.world_model.latent_dim,
        num_actions=num_actions,
        hidden_dims=config.agent.q_network.hidden_dims,
    )


@torch.no_grad()
def double_dqn_target(
    *,
    online_net: nn.Module,
    target_net: nn.Module,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    next_q_online = online_net(next_obs)
    best_actions = next_q_online.argmax(dim=1, keepdim=True)
    next_q_target = target_net(next_obs)
    next_q_values = next_q_target.gather(1, best_actions).squeeze(1)
    return rewards + gamma * next_q_values * (~dones)

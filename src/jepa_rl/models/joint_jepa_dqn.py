"""Latent Q-head for Phase 8 Joint JEPA + DQN.

The encoder lives in JepaWorldModel. This module provides only the Q-head
that maps pre-computed latent vectors to Q-values. Both JEPA loss and DQN
loss backpropagate through the shared encoder in JepaWorldModel.
"""
from __future__ import annotations

import torch
from torch import nn


class LatentQHead(nn.Module):
    """Q-network head that operates on latent vectors produced by JepaWorldModel.encoder.

    Unlike QNetwork / DuelingQNetwork in models/dqn.py, this module does not own
    an encoder. The encoder is shared with JEPA and lives in the JepaWorldModel.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...],
        dueling: bool = True,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.dueling = dueling

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
        else:
            layers: list[nn.Module] = []
            in_dim = latent_dim
            for out_dim in hidden_dims:
                layers += [nn.Linear(in_dim, out_dim), activation()]
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, num_actions))
            self.q_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            h = self.trunk(z)
            value = self.value_branch(h)
            advantage = self.advantage_branch(h)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_layers(z)

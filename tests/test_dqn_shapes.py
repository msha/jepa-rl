from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from jepa_rl.models.dqn import DuelingQNetwork, QNetwork, build_q_network, double_dqn_target
from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.utils.config import load_config


def _encoder(
    input_channels: int = 1,
    hidden_channels: list[int] | None = None,
    latent_dim: int = 64,
) -> ConvEncoder:
    return ConvEncoder(
        input_channels=input_channels,
        hidden_channels=hidden_channels or [16, 32],
        latent_dim=latent_dim,
    )


@pytest.mark.parametrize(
    "input_channels,hidden_channels,latent_dim,hw",
    [
        # 2-layer encoder — 32×32 is sufficient
        (1, [16, 32], 64, 32),
        # 3-layer encoder — needs ≥40px so the 3rd k=3,s=1 conv gets ≥3px input
        (4, [16, 32, 64], 128, 84),
        # 4-layer encoder — 84px leaves 5px after all four conv steps
        (4, [32, 64, 128, 256], 512, 84),
    ],
    ids=["tiny", "small", "base"],
)
@pytest.mark.parametrize("dueling", [False, True])
def test_q_network_forward_shape(
    input_channels: int,
    hidden_channels: list[int],
    latent_dim: int,
    hw: int,
    dueling: bool,
) -> None:
    encoder = _encoder(
        input_channels=input_channels, hidden_channels=hidden_channels, latent_dim=latent_dim
    )
    num_actions = 4
    cls = DuelingQNetwork if dueling else QNetwork
    net = cls(encoder=encoder, latent_dim=latent_dim, num_actions=num_actions, hidden_dims=(256,))
    obs = torch.zeros(2, input_channels, hw, hw)
    out = net(obs)
    assert out.shape == (2, num_actions)
    assert out.dtype == torch.float32


def test_double_dqn_target_shape_and_dtype() -> None:
    encoder = _encoder()
    net = QNetwork(encoder=encoder, latent_dim=64, num_actions=4, hidden_dims=(64,))
    target = copy.deepcopy(net)

    B = 4
    obs = torch.zeros(B, 1, 32, 32)
    rewards = torch.zeros(B)
    dones = torch.zeros(B, dtype=torch.bool)

    result = double_dqn_target(
        online_net=net,
        target_net=target,
        next_obs=obs,
        rewards=rewards,
        dones=dones,
        gamma=0.99,
    )
    assert result.shape == (B,)
    assert result.dtype == torch.float32
    assert not result.requires_grad

    # dones=True must zero the bootstrap term so target == reward only
    dones_all = torch.ones(B, dtype=torch.bool)
    result_done = double_dqn_target(
        online_net=net,
        target_net=target,
        next_obs=obs,
        rewards=rewards,
        dones=dones_all,
        gamma=0.99,
    )
    torch.testing.assert_close(result_done, rewards)


def test_target_network_sync_zero_diff() -> None:
    encoder = _encoder()
    net = QNetwork(encoder=encoder, latent_dim=64, num_actions=4, hidden_dims=(64,))
    target = copy.deepcopy(net)

    # Perturb online net so it diverges from target
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p))

    # Before sync they must differ
    assert any(
        not torch.equal(net.state_dict()[k], target.state_dict()[k])
        for k in net.state_dict()
    )

    # After sync every parameter must match exactly
    target.load_state_dict(net.state_dict())
    for k in net.state_dict():
        assert torch.equal(net.state_dict()[k], target.state_dict()[k]), f"mismatch at {k}"


def test_build_q_network_rejects_distributional() -> None:
    config = load_config("tests/fixtures/configs/dqn_smoke.yaml")
    qn_cfg = dataclasses.replace(config.agent.q_network, distributional=True)
    agent_cfg = dataclasses.replace(config.agent, q_network=qn_cfg)
    patched = dataclasses.replace(config, agent=agent_cfg)
    with pytest.raises(NotImplementedError):
        build_q_network(patched, num_actions=4)

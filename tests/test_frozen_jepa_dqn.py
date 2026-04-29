"""Tests for the FrozenJepaQNetwork model."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.models.frozen_encoder import load_frozen_encoder
from jepa_rl.models.frozen_jepa_dqn import FrozenJepaQNetwork, build_frozen_jepa_q_network
from jepa_rl.utils.config import load_config
from tests.fakes.scripted_env import ScriptedEnv


def _make_encoder(latent_dim: int = 64, input_channels: int = 4) -> ConvEncoder:
    return ConvEncoder(
        input_channels=input_channels,
        hidden_channels=[16, 32],
        latent_dim=latent_dim,
    )


class TestFrozenJepaQNetworkShapes:
    @pytest.mark.parametrize("dueling", [True, False])
    @pytest.mark.parametrize("latent_dim", [32, 64])
    def test_forward_shape(self, dueling: bool, latent_dim: int) -> None:
        num_actions = 4
        net = FrozenJepaQNetwork(
            encoder=_make_encoder(latent_dim=latent_dim),
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dims=(128, 64),
            dueling=dueling,
        )
        obs = torch.randn(8, 4, 84, 84)
        q = net(obs)
        assert q.shape == (8, num_actions)

    def test_encode_output_shape(self) -> None:
        latent_dim = 64
        net = FrozenJepaQNetwork(
            encoder=_make_encoder(latent_dim=latent_dim),
            latent_dim=latent_dim,
            num_actions=4,
            hidden_dims=(128,),
            dueling=True,
        )
        obs = torch.randn(3, 4, 84, 84)
        z = net.encode(obs)
        assert z.shape == (3, latent_dim)


class TestFrozenEncoder:
    def test_encoder_params_have_no_grad(self) -> None:
        net = FrozenJepaQNetwork(
            encoder=_make_encoder(),
            latent_dim=64,
            num_actions=4,
            hidden_dims=(128,),
            dueling=True,
        )
        for p in net.encoder.parameters():
            assert not p.requires_grad

    def test_gradients_flow_only_through_q_head(self) -> None:
        net = FrozenJepaQNetwork(
            encoder=_make_encoder(),
            latent_dim=64,
            num_actions=4,
            hidden_dims=(128,),
            dueling=True,
        )
        obs = torch.randn(4, 4, 84, 84)
        q = net(obs)
        loss = q.sum()
        loss.backward()

        for p in net.encoder.parameters():
            assert p.grad is None

        trainable = net.trainable_parameters
        assert len(trainable) > 0
        for p in trainable:
            assert p.grad is not None

    def test_encoder_weights_unchanged_after_backward(self) -> None:
        net = FrozenJepaQNetwork(
            encoder=_make_encoder(),
            latent_dim=64,
            num_actions=4,
            hidden_dims=(128,),
            dueling=True,
        )
        before = {n: p.clone() for n, p in net.encoder.named_parameters()}

        obs = torch.randn(4, 4, 84, 84)
        loss = net(obs).sum()
        loss.backward()

        for n, p in net.encoder.named_parameters():
            assert torch.equal(p, before[n]), f"encoder param {n} changed after backward"


class TestBuildFromCheckpoint:
    def test_build_from_jepa_checkpoint(self, tmp_path: Path) -> None:
        config = load_config("configs/presets/tiny.yaml")
        latent_dim = config.world_model.latent_dim
        num_actions = config.actions.num_actions

        encoder = ConvEncoder(
            input_channels=config.observation.input_channels,
            hidden_channels=list(config.world_model.encoder.hidden_channels),
            latent_dim=latent_dim,
        )
        ckpt_path = tmp_path / "jepa.pt"
        encoder_sd = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
        torch.save({"model": encoder_sd}, ckpt_path)

        net = build_frozen_jepa_q_network(
            config, num_actions=num_actions, jepa_checkpoint_path=ckpt_path
        )
        assert isinstance(net, FrozenJepaQNetwork)
        for p in net.encoder.parameters():
            assert not p.requires_grad

        obs = torch.randn(
            2,
            config.observation.input_channels,
            config.observation.height,
            config.observation.width,
        )
        q = net(obs)
        assert q.shape == (2, num_actions)

    def test_target_network_sync(self, tmp_path: Path) -> None:
        config = load_config("configs/presets/tiny.yaml")
        latent_dim = config.world_model.latent_dim
        num_actions = config.actions.num_actions

        encoder = ConvEncoder(
            input_channels=config.observation.input_channels,
            hidden_channels=list(config.world_model.encoder.hidden_channels),
            latent_dim=latent_dim,
        )
        ckpt_path = tmp_path / "jepa.pt"
        encoder_sd = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
        torch.save({"model": encoder_sd}, ckpt_path)

        online = build_frozen_jepa_q_network(
            config, num_actions=num_actions, jepa_checkpoint_path=ckpt_path
        )
        target = copy.deepcopy(online)

        obs = torch.randn(
            2,
            config.observation.input_channels,
            config.observation.height,
            config.observation.width,
        )
        assert torch.allclose(online(obs), target(obs))

        loss = online(obs).sum()
        loss.backward()
        for p in online.trainable_parameters:
            p.data -= 0.1 * p.grad
            p.grad = None

        assert not torch.allclose(online(obs), target(obs))


def test_load_frozen_encoder_returns_no_grad_latents(tmp_path: Path) -> None:
    config = load_config("configs/presets/tiny.yaml")
    encoder = ConvEncoder(
        input_channels=config.observation.input_channels,
        hidden_channels=list(config.world_model.encoder.hidden_channels),
        latent_dim=config.world_model.latent_dim,
    )
    ckpt_path = tmp_path / "jepa.pt"
    torch.save({"model": {f"encoder.{k}": v for k, v in encoder.state_dict().items()}}, ckpt_path)

    frozen = load_frozen_encoder(config, ckpt_path)
    obs = torch.randn(
        2,
        config.observation.input_channels,
        config.observation.height,
        config.observation.width,
        requires_grad=True,
    )
    z = frozen.encode(obs)

    assert z.shape == (2, config.world_model.latent_dim)
    assert not z.requires_grad
    assert all(not parameter.requires_grad for parameter in frozen.encoder.parameters())


def test_frozen_jepa_training_updates_head_and_keeps_encoder_fixed(tmp_path: Path) -> None:
    config = load_config("tests/fixtures/configs/joint_jepa_dqn_smoke.yaml")
    config = dataclasses.replace(
        config,
        experiment=dataclasses.replace(config.experiment, output_dir=str(tmp_path)),
        agent=dataclasses.replace(
            config.agent,
            algorithm="frozen_jepa_dqn",
            learning_starts=0,
            batch_size=8,
            target_update_interval=40,
        ),
    )
    encoder = ConvEncoder(
        input_channels=config.observation.input_channels,
        hidden_channels=list(config.world_model.encoder.hidden_channels),
        latent_dim=config.world_model.latent_dim,
    )
    encoder_state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    ckpt_path = tmp_path / "jepa.pt"
    torch.save({"model": {f"encoder.{k}": v for k, v in encoder.state_dict().items()}}, ckpt_path)

    from jepa_rl.training.frozen_jepa_dqn import train_frozen_jepa_dqn

    def _env_factory(cfg, headless):
        return ScriptedEnv(
            num_actions=cfg.actions.num_actions,
            height=cfg.observation.height,
            width=cfg.observation.width,
            channels=cfg.observation.input_channels,
            episode_length=32,
            seed=0,
        )

    summary = train_frozen_jepa_dqn(
        config,
        jepa_checkpoint=ckpt_path,
        experiment="frozen_train_test",
        steps=80,
        learning_starts=0,
        batch_size=8,
        dashboard_every=0,
        env_factory=_env_factory,
    )

    assert summary.checkpoint.exists()
    assert summary.update_count > 0
    assert summary.weight_delta_norm > 0.0

    saved = torch.load(summary.checkpoint, map_location="cpu", weights_only=False)
    for name, expected in encoder_state.items():
        assert torch.equal(saved["model"][f"encoder.{name}"], expected)

    run_summary = json.loads(
        (summary.run_dir / "metrics" / "train_summary.json").read_text(encoding="utf-8")
    )
    assert run_summary["algorithm"] == "frozen_jepa_dqn"
    assert run_summary["encoder_frozen"] is True
    assert run_summary["latent_path"] == "on_the_fly"

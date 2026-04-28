from __future__ import annotations

import pytest
import torch

from jepa_rl.models.encoders import ConvEncoder
from jepa_rl.models.jepa import JepaBatch, JepaWorldModel, JepaWorldModelConfig
from jepa_rl.models.losses import (
    covariance_loss,
    effective_rank,
    normalized_prediction_loss,
    variance_loss,
)
from jepa_rl.models.predictors import ActionConditionedPredictor
from jepa_rl.utils.config import load_config

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def smoke_config():
    return load_config("tests/fixtures/configs/jepa_smoke.yaml")


@pytest.fixture()
def world_model(smoke_config):
    cfg = JepaWorldModelConfig.from_project_config(smoke_config)
    return JepaWorldModel(cfg)


@pytest.fixture()
def jepa_batch(smoke_config):
    cfg = smoke_config
    B, C, H, W = 4, cfg.observation.input_channels, cfg.observation.height, cfg.observation.width
    max_horizon = max(cfg.world_model.predictor.horizons)
    context = torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)
    actions = torch.randint(0, cfg.actions.num_actions, (B, max_horizon))
    targets = {
        h: torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)
        for h in cfg.world_model.predictor.horizons
    }
    return JepaBatch(context_obs=context, target_obs=targets, actions=actions)


# ---------------------------------------------------------------------------
# Encoder shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_channels,hw", [(1, 32), (4, 84)])
def test_conv_encoder_output_shape(input_channels, hw):
    encoder = ConvEncoder(
        input_channels=input_channels,
        hidden_channels=[8, 16],
        latent_dim=32,
    )
    obs = torch.zeros(2, input_channels, hw, hw)
    out = encoder(obs)
    assert out.shape == (2, 32)
    assert out.dtype == torch.float32


def test_conv_encoder_uint8_auto_scale():
    encoder = ConvEncoder(input_channels=1, hidden_channels=[8], latent_dim=16)
    obs_uint8 = torch.randint(0, 256, (2, 1, 32, 32), dtype=torch.uint8)
    obs_float = obs_uint8.float() / 255.0
    out_uint8 = encoder(obs_uint8)
    out_float = encoder(obs_float)
    torch.testing.assert_close(out_uint8, out_float)


# ---------------------------------------------------------------------------
# Predictor shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizons", [[1], [1, 2], [1, 2, 4]])
def test_predictor_output_shape(horizons):
    pred = ActionConditionedPredictor(
        latent_dim=32,
        num_actions=4,
        action_embed_dim=8,
        hidden_dim=32,
        depth=1,
        num_heads=2,
        horizons=horizons,
    )
    B = 3
    context = torch.randn(B, 32)
    actions = torch.randint(0, 4, (B, max(horizons)))
    out = pred(context, actions)
    assert set(out.keys()) == set(horizons)
    for h in horizons:
        assert out[h].shape == (B, 32)


# ---------------------------------------------------------------------------
# EMA target encoder tests
# ---------------------------------------------------------------------------

def test_ema_update_interpolates(world_model):
    model = world_model
    # Store initial target weights
    target_before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
    # Perturb online encoder
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.ones_like(p) * 0.1)
    # EMA update with tau=0.9 (keeps 90% of target, adds 10% of online)
    tau = 0.9
    model.update_target_encoder(tau)
    target_after = model.target_encoder.state_dict()
    # New target should be between old target and new online
    online = model.encoder.state_dict()
    for k in target_before:
        if target_before[k].dtype.is_floating_point:
            expected = tau * target_before[k] + (1 - tau) * online[k]
            torch.testing.assert_close(target_after[k], expected, atol=1e-5, rtol=0)


def test_ema_tau_zero_copies_online(world_model):
    model = world_model
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.fill_(99.0)
    model.update_target_encoder(tau=0.0)  # full copy from online
    for k in model.encoder.state_dict():
        if model.encoder.state_dict()[k].dtype.is_floating_point:
            assert torch.equal(model.encoder.state_dict()[k], model.target_encoder.state_dict()[k])


def test_ema_tau_one_leaves_target_unchanged(world_model):
    model = world_model
    target_before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.fill_(99.0)
    model.update_target_encoder(tau=1.0)  # full retain of old target
    for k in target_before:
        assert torch.equal(model.target_encoder.state_dict()[k], target_before[k])


def test_target_encoder_has_no_grad(world_model):
    model = world_model
    for p in model.target_encoder.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# JepaWorldModel forward and loss shape tests
# ---------------------------------------------------------------------------

def test_jepa_forward_returns_all_keys(world_model, jepa_batch):
    out = world_model(jepa_batch)
    assert "loss" in out
    assert "prediction_loss" in out
    assert "variance_loss" in out
    assert "covariance_loss" in out
    assert "latent_std_mean" in out
    assert "effective_rank" in out


def test_jepa_forward_loss_is_scalar(world_model, jepa_batch):
    out = world_model(jepa_batch)
    assert out["loss"].shape == ()
    assert out["loss"].dtype == torch.float32


def test_jepa_per_horizon_losses_present(world_model, jepa_batch, smoke_config):
    out = world_model(jepa_batch)
    for h in smoke_config.world_model.predictor.horizons:
        assert f"prediction_loss_h{h}" in out


def test_jepa_loss_backward(world_model, jepa_batch):
    out = world_model(jepa_batch)
    out["loss"].backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in world_model.encoder.parameters()
    )
    assert has_grad, "no gradient flowed into encoder"


# ---------------------------------------------------------------------------
# Loss function unit tests
# ---------------------------------------------------------------------------

def test_normalized_prediction_loss_zero_for_identical():
    z = torch.randn(8, 32)
    loss = normalized_prediction_loss(z, z)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_normalized_prediction_loss_range():
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    loss = normalized_prediction_loss(z1, z2)
    assert 0.0 <= float(loss) <= 4.0


def test_variance_loss_zero_for_high_variance():
    z = torch.randn(64, 32) * 10
    loss = variance_loss(z, variance_floor=1.0)
    assert float(loss) == pytest.approx(0.0, abs=0.1)


def test_covariance_loss_nonnegative():
    z = torch.randn(16, 32)
    loss = covariance_loss(z)
    assert float(loss) >= 0.0


def test_effective_rank_range():
    z = torch.randn(16, 32)
    rank = effective_rank(z)
    assert 1.0 <= float(rank) <= 32.0

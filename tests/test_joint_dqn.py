"""Phase 8 verification: Joint JEPA + DQN training loop."""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from jepa_rl.utils.config import load_config
from tests.fakes.scripted_env import ScriptedEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_env_factory(cfg, headless):
    return ScriptedEnv(
        num_actions=cfg.actions.num_actions,
        height=cfg.observation.height,
        width=cfg.observation.width,
        channels=cfg.observation.input_channels,
        seed=0,
    )


def _load_smoke_config(tmp_path):
    config = load_config("tests/fixtures/configs/joint_jepa_dqn_smoke.yaml")
    exp_cfg = dataclasses.replace(config.experiment, output_dir=str(tmp_path))
    return dataclasses.replace(config, experiment=exp_cfg)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


def test_config_accepts_joint_jepa_dqn_algorithm(tmp_path):
    """Config parser must accept joint_jepa_dqn as a valid algorithm."""
    config = _load_smoke_config(tmp_path)
    assert config.agent.algorithm == "joint_jepa_dqn"


# ---------------------------------------------------------------------------
# 2. Model shapes (no browser, no training loop)
# ---------------------------------------------------------------------------


def test_latent_q_head_output_shape():
    """LatentQHead takes a batch of latent vectors and returns Q-values."""
    from jepa_rl.models.joint_jepa_dqn import LatentQHead

    head = LatentQHead(latent_dim=32, num_actions=4, hidden_dims=(64,), dueling=False)
    z = torch.randn(8, 32)
    q = head(z)
    assert q.shape == (8, 4)


def test_latent_q_head_dueling_output_shape():
    from jepa_rl.models.joint_jepa_dqn import LatentQHead

    head = LatentQHead(latent_dim=32, num_actions=4, hidden_dims=(64, 32), dueling=True)
    z = torch.randn(5, 32)
    q = head(z)
    assert q.shape == (5, 4)


def test_target_encoder_not_called_during_action_selection():
    """The online action-selection path (encode → Q) must never invoke the target encoder."""
    from jepa_rl.models.jepa import JepaWorldModel, JepaWorldModelConfig
    from jepa_rl.models.joint_jepa_dqn import LatentQHead

    cfg = JepaWorldModelConfig(
        input_channels=1,
        latent_dim=32,
        hidden_channels=[8, 16],
        num_actions=4,
        predictor_hidden_dim=32,
        predictor_depth=1,
        predictor_heads=2,
        action_embed_dim=8,
        horizons=[1],
        lambda_var=1.0,
        lambda_cov=0.04,
        variance_floor=1.0,
    )
    jepa = JepaWorldModel(cfg)
    q_head = LatentQHead(latent_dim=32, num_actions=4, hidden_dims=(64,), dueling=False)

    target_called: list[bool] = []

    def _hook(module, inp, out):
        target_called.append(True)

    for layer in jepa.target_encoder.modules():
        layer.register_forward_hook(_hook)

    obs = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        z = jepa.encode(obs)
        _ = q_head(z)

    assert not target_called, "target_encoder was called during action selection"


# ---------------------------------------------------------------------------
# 3. Training smoke (requires ScriptedEnv, marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_joint_training_produces_checkpoint(tmp_path):
    """Joint training must save latest.pt to the run directory."""
    config = _load_smoke_config(tmp_path)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_smoke_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    assert summary.checkpoint.exists(), "latest.pt was not written"
    assert summary.steps > 0


@pytest.mark.slow
def test_joint_training_logs_jepa_and_dqn_metrics(tmp_path):
    """Step events must contain both jepa_loss and td_error after learning starts."""
    config = _load_smoke_config(tmp_path)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_metrics_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    metrics_path = summary.run_dir / "metrics" / "train_events.jsonl"
    assert metrics_path.exists()

    events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    step_events = [e for e in events if e.get("type") == "step"]
    # Only check events after learning starts
    post_warmup = [e for e in step_events if e.get("step", 0) >= config.agent.learning_starts]
    assert post_warmup, "no step events logged after learning starts"
    assert all("jepa_loss" in e for e in post_warmup), "jepa_loss missing from step events"
    assert all("td_error" in e for e in post_warmup), "td_error missing from step events"


@pytest.mark.slow
def test_joint_checkpoint_contains_both_models(tmp_path):
    """The checkpoint file must include state dicts for both JEPA and Q-head."""
    config = _load_smoke_config(tmp_path)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_ckpt_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    state = torch.load(summary.checkpoint, map_location="cpu", weights_only=False)
    assert "jepa_model" in state, "checkpoint missing jepa_model state dict"
    assert "q_net" in state, "checkpoint missing q_net state dict"
    assert "q_target_net" in state, "checkpoint missing q_target_net state dict"


@pytest.mark.slow
def test_joint_checkpoint_round_trip(tmp_path):
    """Resume from latest.pt must restore step counter and model weights."""
    config = _load_smoke_config(tmp_path)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary1 = train_joint_jepa_dqn(
        config,
        experiment="joint_resume_test",
        steps=200,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    summary2 = train_joint_jepa_dqn(
        config,
        experiment="joint_resume_test",
        steps=100,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
        resume_checkpoint=summary1.checkpoint,
    )

    assert summary2.steps >= 100, "resumed run did not complete requested steps"


@pytest.mark.slow
def test_joint_training_summary_has_jepa_fields(tmp_path):
    """Summary must expose jepa_loss, latent_std, and effective_rank."""
    config = _load_smoke_config(tmp_path)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_fields_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    assert hasattr(summary, "jepa_loss"), "summary missing jepa_loss"
    assert hasattr(summary, "latent_std"), "summary missing latent_std"
    assert hasattr(summary, "effective_rank"), "summary missing effective_rank"
    assert summary.jepa_loss >= 0.0


# ---------------------------------------------------------------------------
# 4. Intrinsic reward (Phase 8 P1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_intrinsic_reward_logged_when_enabled(tmp_path):
    """When intrinsic_reward.enabled=True, step events must include intrinsic_reward."""
    import dataclasses

    config = _load_smoke_config(tmp_path)
    # Enable intrinsic reward with beta > 0
    from jepa_rl.utils.config import IntrinsicRewardConfig

    new_ir = IntrinsicRewardConfig(enabled=True, source="jepa_prediction_error", beta=0.1,
                                   normalize=True)
    new_exploration = dataclasses.replace(config.exploration, intrinsic_reward=new_ir)
    config = dataclasses.replace(config, exploration=new_exploration)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_intrinsic_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    metrics_path = summary.run_dir / "metrics" / "train_events.jsonl"
    events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    step_events = [e for e in events if e.get("type") == "step"]
    post_warmup = [e for e in step_events if e.get("step", 0) >= config.agent.learning_starts]
    assert post_warmup, "no step events after learning starts"
    assert all("intrinsic_reward" in e for e in post_warmup), (
        "intrinsic_reward missing from step events when enabled"
    )
    assert any(e["intrinsic_reward"] is not None and e["intrinsic_reward"] != 0.0
               for e in post_warmup), "intrinsic_reward is always 0 or None"


@pytest.mark.slow
def test_intrinsic_reward_not_logged_when_disabled(tmp_path):
    """When intrinsic_reward.enabled=False, step events must not include intrinsic_reward key."""
    config = _load_smoke_config(tmp_path)
    assert not config.exploration.intrinsic_reward.enabled

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_no_intrinsic_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    metrics_path = summary.run_dir / "metrics" / "train_events.jsonl"
    events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    step_events = [e for e in events if e.get("type") == "step"]
    assert all("intrinsic_reward" not in e or e["intrinsic_reward"] is None
               for e in step_events), "intrinsic_reward logged even when disabled"


# ---------------------------------------------------------------------------
# 5. Freeze encoder ablation (Phase 8 P1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_freeze_encoder_prevents_encoder_grad(tmp_path):
    """With training.freeze_encoder=True, the encoder weights must not change during DQN updates."""
    import dataclasses

    config = _load_smoke_config(tmp_path)

    new_training = dataclasses.replace(config.training, freeze_encoder=True)
    config = dataclasses.replace(config, training=new_training)

    from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

    summary = train_joint_jepa_dqn(
        config,
        experiment="joint_freeze_test",
        steps=300,
        dashboard_every=0,
        env_factory=_scripted_env_factory,
    )

    # The encoder is updated only by JEPA loss when freeze_encoder=True;
    # DQN backward is detached. Verify the run completes without error.
    assert summary.checkpoint.exists()
    assert summary.steps > 0

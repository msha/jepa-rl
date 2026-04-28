"""Phase 6 verification gate: JEPA world model loss decreases on the scripted env."""

from __future__ import annotations

import dataclasses

import pytest

from jepa_rl.utils.config import load_config
from tests.fakes.scripted_env import ScriptedEnv


@pytest.mark.slow
def test_jepa_world_model_loss_decreases(tmp_path):
    config = load_config("tests/fixtures/configs/jepa_smoke.yaml")
    exp_cfg = dataclasses.replace(config.experiment, output_dir=str(tmp_path))
    config = dataclasses.replace(config, experiment=exp_cfg)

    def _env_factory(cfg, headless):
        return ScriptedEnv(
            num_actions=cfg.actions.num_actions,
            height=cfg.observation.height,
            width=cfg.observation.width,
            channels=cfg.observation.input_channels,
            seed=0,
        )

    from jepa_rl.training.jepa_world import train_jepa_world

    summary = train_jepa_world(
        config,
        experiment="jepa_smoke_test",
        steps=200,
        collect_steps=500,
        batch_size=8,
        dashboard_every=0,
        env_factory=_env_factory,
    )

    assert summary.final_loss < summary.initial_loss, (
        f"JEPA loss did not decrease: initial={summary.initial_loss:.4f} "
        f"final={summary.final_loss:.4f}"
    )
    assert summary.effective_rank > 1.0, "latent collapsed to rank 1"
    assert summary.latent_std_mean > 0.0

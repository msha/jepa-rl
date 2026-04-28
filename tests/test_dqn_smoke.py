"""Phase 5 verification gate: DQN beats random on the scripted fake environment."""

from __future__ import annotations

import dataclasses
import random
from statistics import mean, pstdev

import pytest

from jepa_rl.utils.config import load_config
from tests.fakes.scripted_env import ScriptedEnv


def _random_baseline(num_actions: int, *, episodes: int = 20, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    scores = []
    for _ in range(episodes):
        env = ScriptedEnv(num_actions=num_actions, seed=0)
        env.reset()
        while not env.is_done():
            env.step(rng.randrange(num_actions))
        scores.append(env.read_score())
    return scores


@pytest.mark.slow
def test_dqn_beats_random_on_scripted_env(tmp_path):
    config = load_config("tests/fixtures/configs/dqn_smoke.yaml")
    # Redirect run output to tmp_path so tests don't write to runs/
    exp_cfg = dataclasses.replace(config.experiment, output_dir=str(tmp_path))
    config = dataclasses.replace(config, experiment=exp_cfg)

    num_actions = config.actions.num_actions  # 4

    rb_scores = _random_baseline(num_actions, episodes=20, seed=0)
    rb_mean = mean(rb_scores)
    rb_std = pstdev(rb_scores)

    from jepa_rl.training.pixel_dqn import evaluate_dqn, train_dqn

    def _env_factory(cfg, headless):
        return ScriptedEnv(
            num_actions=cfg.actions.num_actions,
            height=cfg.observation.height,
            width=cfg.observation.width,
            channels=cfg.observation.input_channels,
            seed=1,
        )

    summary = train_dqn(
        config,
        experiment="dqn_smoke_test",
        steps=10_000,
        learning_starts=200,
        dashboard_every=0,
        env_factory=_env_factory,
    )

    def _eval_factory(cfg, headless):
        return ScriptedEnv(
            num_actions=cfg.actions.num_actions,
            height=cfg.observation.height,
            width=cfg.observation.width,
            channels=cfg.observation.input_channels,
            seed=2,
        )

    eval_out = evaluate_dqn(
        config,
        checkpoint=summary.best_checkpoint,
        episodes=20,
        env_factory=_eval_factory,
    )

    dqn_mean = mean(eval_out["scores"])
    threshold = rb_mean + 3 * rb_std
    assert dqn_mean > threshold, (
        f"DQN mean score {dqn_mean:.2f} did not beat random threshold {threshold:.2f} "
        f"(random: mean={rb_mean:.2f}, std={rb_std:.2f})"
    )

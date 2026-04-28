from __future__ import annotations

import random

import numpy as np

from jepa_rl.training.simple_q import (
    FeatureTransition,
    LinearQModel,
    LinearReplay,
    run_linear_q_ml_smoke,
)


def test_linear_q_update_moves_selected_action_toward_target() -> None:
    rng = np.random.default_rng(1)
    model = LinearQModel(feature_dim=3, num_actions=2, rng=rng)
    features = np.array([0.2, -0.1, 1.0], dtype=np.float32)
    action = 1
    target = 1.5

    before = abs(target - float(model.q_values(features)[action]))
    loss = model.update(features, action, target, lr=0.1)
    after = abs(target - float(model.q_values(features)[action]))

    assert loss > 0
    assert after < before


def test_linear_q_ml_smoke_reduces_loss_and_changes_weights() -> None:
    summary = run_linear_q_ml_smoke(seed=4, steps=600, lr=0.03)

    assert summary.final_loss < summary.initial_loss
    assert summary.improvement > 0
    assert summary.weight_delta_norm > 0


def test_linear_replay_samples_minibatches() -> None:
    replay = LinearReplay(capacity=3)
    for index in range(5):
        replay.add(
            FeatureTransition(
                features=np.array([index, 1.0], dtype=np.float32),
                action=index % 2,
                reward=float(index),
                next_features=np.array([index + 1, 1.0], dtype=np.float32),
                done=False,
            )
        )

    batch = replay.sample(2, random.Random(2))

    assert len(replay) == 3
    assert len(batch) == 2
    assert {int(item.features[0]) for item in replay._storage} == {2, 3, 4}

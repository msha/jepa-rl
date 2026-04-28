from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from jepa_rl.models.dqn import build_q_network
from jepa_rl.utils.checkpoint import CheckpointPayload, load_torch_checkpoint, save_torch_checkpoint
from jepa_rl.utils.config import load_config


@pytest.fixture()
def smoke_config():
    return load_config("tests/fixtures/configs/dqn_smoke.yaml")


def test_checkpoint_round_trip(smoke_config, tmp_path):
    config = smoke_config
    net = build_q_network(config, num_actions=config.actions.num_actions)
    target_net = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Do a gradient step so optimizer state is non-trivial
    obs = torch.zeros(
        1, config.observation.input_channels, config.observation.height, config.observation.width
    )
    q = net(obs)
    q.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    py_rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    # Consume some RNG state so restore is non-trivial
    for _ in range(5):
        py_rng.random()
        np_rng.random()

    saved_py_state = py_rng.getstate()
    saved_np_state = np_rng.bit_generator.state
    saved_torch_state = torch.get_rng_state()

    payload = CheckpointPayload(
        step=100,
        episode=3,
        update_count=50,
        target_update_count=1,
        best_score=7.5,
        config_dict=config.to_dict(),
        rng_python=saved_py_state,
        rng_numpy=saved_np_state,
        rng_torch=saved_torch_state,
    )

    ckpt = tmp_path / "test.pt"
    save_torch_checkpoint(
        ckpt, model=net, target_model=target_net, optimizer=optimizer, payload=payload
    )

    # Load into fresh models
    fresh_net = build_q_network(config, num_actions=config.actions.num_actions)
    fresh_target = copy.deepcopy(fresh_net)
    fresh_opt = torch.optim.Adam(fresh_net.parameters(), lr=1e-3)

    loaded = load_torch_checkpoint(
        ckpt, model=fresh_net, target_model=fresh_target, optimizer=fresh_opt
    )

    # Model state dicts must be bitwise equal
    for k in net.state_dict():
        assert torch.equal(net.state_dict()[k], fresh_net.state_dict()[k]), f"model mismatch at {k}"
    for k in target_net.state_dict():
        orig = target_net.state_dict()[k]
        loaded_v = fresh_target.state_dict()[k]
        assert torch.equal(orig, loaded_v), f"target mismatch at {k}"

    # Payload scalar fields
    assert loaded.step == 100
    assert loaded.episode == 3
    assert loaded.update_count == 50
    assert loaded.target_update_count == 1
    assert loaded.best_score == pytest.approx(7.5)

    # RNG states must reproduce the same sequence after restore
    py_rng.setstate(saved_py_state)
    py_rng2 = random.Random(0)
    py_rng2.setstate(loaded.rng_python)
    for _ in range(10):
        assert py_rng.random() == py_rng2.random()


def test_checkpoint_without_target_and_optimizer(smoke_config, tmp_path):
    config = smoke_config
    net = build_q_network(config, num_actions=config.actions.num_actions)

    payload = CheckpointPayload(
        step=0,
        episode=0,
        update_count=0,
        target_update_count=0,
        best_score=0.0,
        config_dict={},
        rng_python=random.getstate(),
        rng_numpy=np.random.default_rng().bit_generator.state,
        rng_torch=torch.get_rng_state(),
    )
    ckpt = tmp_path / "minimal.pt"
    save_torch_checkpoint(ckpt, model=net, target_model=None, optimizer=None, payload=payload)

    fresh_net = build_q_network(config, num_actions=config.actions.num_actions)
    loaded = load_torch_checkpoint(ckpt, model=fresh_net, target_model=None, optimizer=None)

    for k in net.state_dict():
        assert torch.equal(net.state_dict()[k], fresh_net.state_dict()[k])
    assert loaded.step == 0

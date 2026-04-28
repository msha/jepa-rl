from __future__ import annotations

from pathlib import Path

import pytest

from jepa_rl.utils.config import ConfigError, load_config, load_config_dict, snapshot_config


def test_load_breakout_config_merges_base_and_small_preset() -> None:
    config = load_config("configs/games/breakout.yaml")

    assert config.experiment.name == "breakout_jepa_dqn_small"
    assert config.game.name == "breakout"
    assert config.observation.width == 160
    assert config.observation.height == 120
    assert config.observation.input_channels == 12
    assert config.actions.num_actions == 6
    assert config.world_model.latent_dim == 512
    assert config.agent.algorithm == "dqn"
    assert config.replay.sequence_length == 16


def test_tiny_preset_can_override_base_values(tmp_path) -> None:
    config_file = tmp_path / "tiny_breakout.yaml"
    base_config = Path("configs/base.yaml").resolve()
    tiny_preset = Path("configs/presets/tiny.yaml").resolve()
    config_file.write_text(
        "\n".join(
            [
                "extends:",
                f"  - {base_config}",
                f"  - {tiny_preset}",
                "experiment:",
                "  name: tiny_test",
                "game:",
                "  name: tiny_game",
                '  url: "http://localhost:8000/tiny.html"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.observation.width == 84
    assert config.observation.height == 84
    assert config.observation.grayscale is True
    assert config.observation.input_channels == 4
    assert config.world_model.latent_dim == 128
    assert config.training.total_env_steps == 10000


def test_load_config_dict_preserves_extends_merging() -> None:
    data = load_config_dict("configs/games/breakout.yaml")

    assert "extends" not in data
    assert data["game"]["name"] == "breakout"
    assert data["world_model"]["predictor"]["horizons"] == [1, 2, 4, 8]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("tests/fixtures/configs/invalid_gamma.yaml", "agent.gamma"),
        ("tests/fixtures/configs/invalid_javascript_reward.yaml", "javascript"),
        ("tests/fixtures/configs/invalid_v2_conditioning.yaml", "V2/V3"),
    ],
)
def test_invalid_configs_fail_fast(path: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_snapshot_config_writes_loadable_yaml(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    output = tmp_path / "run" / "config.yaml"

    snapshot_config(config, output)
    loaded = load_config(output)

    assert loaded.experiment.name == config.experiment.name
    assert loaded.actions.keys == config.actions.keys

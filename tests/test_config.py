from __future__ import annotations

from pathlib import Path

import pytest

from jepa_rl.utils.config import ConfigError, load_config, load_config_dict, snapshot_config


def test_load_breakout_config_merges_base_and_small_preset() -> None:
    config = load_config("configs/games/breakout.yaml")

    assert config.experiment.name == "breakout_jepa_dqn_small"
    assert config.game.name == "breakout"
    assert config.game.reset_key == "Space"
    assert config.game.reset_button_selector is None
    assert config.game.reset_javascript is None
    assert config.observation.width == 160
    assert config.observation.height == 120
    assert config.observation.input_channels == 12
    assert config.observation.dom_selectors == {"score": "#score", "lives": "#lives"}
    assert config.actions.num_actions == 6
    assert config.world_model.latent_dim == 512
    assert config.agent.algorithm == "dqn"
    assert config.replay.sequence_length == 16
    assert config.training.eval_budgets == (100_000, 500_000, 1_000_000, 5_000_000)
    assert config.reward.zero_score_patience_steps == 120
    assert config.reward.zero_score_penalty == 0.01


@pytest.mark.parametrize(
    ("path", "width", "height", "latent_dim"),
    [
        ("configs/presets/tiny.yaml", 84, 84, 128),
        ("configs/presets/small.yaml", 160, 120, 512),
        ("configs/presets/base.yaml", 224, 224, 768),
    ],
)
def test_presets_validate_as_full_configs(
    path: str, width: int, height: int, latent_dim: int
) -> None:
    config = load_config(path)

    assert config.observation.width == width
    assert config.observation.height == height
    assert config.world_model.latent_dim == latent_dim


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


def test_reward_zero_score_penalty_must_be_nonnegative() -> None:
    from jepa_rl.utils.config import RewardConfig

    with pytest.raises(ConfigError, match="zero_score_penalty"):
        RewardConfig.from_dict({
            "type": "score_delta",
            "score_reader": "dom",
            "score_selector": "#score",
            "zero_score_penalty": -0.01,
        })


def test_dom_assisted_observation_requires_dom_selectors(tmp_path) -> None:
    config_file = tmp_path / "invalid_dom_assisted.yaml"
    base_config = Path("configs/base.yaml").resolve()
    config_file.write_text(
        "\n".join(
            [
                "extends:",
                f"  - {base_config}",
                "observation:",
                "  mode: dom_assisted",
                "  dom_selectors: null",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="dom_selectors"):
        load_config(config_file)


def test_javascript_reset_requires_privileged_flag(tmp_path) -> None:
    config_file = tmp_path / "invalid_js_reset.yaml"
    base_config = Path("configs/base.yaml").resolve()
    config_file.write_text(
        "\n".join(
            [
                "extends:",
                f"  - {base_config}",
                "game:",
                "  reset_javascript: window.resetGame()",
                "reward:",
                "  privileged: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="javascript reset"):
        load_config(config_file)

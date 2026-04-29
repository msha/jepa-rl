from __future__ import annotations

import json

from jepa_rl.ui.server import (
    EvalJob,
    UiState,
    _apply_config_overrides,
    _get_defaults,
    _has_locked_start_overrides,
    build_state_payload,
    list_configs,
    list_runs,
)
from jepa_rl.utils.config import load_config, snapshot_config


def test_build_state_payload_reads_metrics(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    object.__setattr__(config.experiment, "output_dir", str(tmp_path))
    run_dir = tmp_path / "ui_test"
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "train_summary.json").write_text(
        json.dumps({"steps": 2, "requested_steps": 10, "best_score": 4}),
        encoding="utf-8",
    )
    (metrics / "train_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "step", "step": 1, "score": 1, "loss": 0.5}),
                json.dumps({"type": "episode", "episode": 0, "step": 2, "score": 4}),
            ]
        ),
        encoding="utf-8",
    )
    state = UiState(
        config_path=tmp_path / "config.yaml",
        config=config,
        experiment="ui_test",
        default_steps=10,
        learning_starts=0,
        batch_size=8,
        dashboard_every=5,
        run_dir=run_dir,
    )

    payload = build_state_payload(state)

    assert payload["summary"]["best_score"] == 4
    assert payload["latest_step"]["score"] == 1
    assert payload["config"]["action_keys"] == list(config.actions.keys)
    assert payload["base_model_info"]["algorithm"] == config.agent.algorithm
    assert len(payload["episodes"]) == 1


def test_build_state_payload_includes_unflushed_live_steps(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    object.__setattr__(config.experiment, "output_dir", str(tmp_path))
    run_dir = tmp_path / "ui_live"
    (run_dir / "metrics").mkdir(parents=True)
    state = UiState(
        config_path=tmp_path / "config.yaml",
        config=config,
        experiment="ui_live",
        default_steps=10,
        learning_starts=0,
        batch_size=8,
        dashboard_every=5,
        run_dir=run_dir,
    )
    state.live_step_events.append(
        {"type": "step", "step": 3, "episode": 0, "action": 2, "score": 0}
    )

    payload = build_state_payload(state)

    assert payload["latest_step"]["step"] == 3
    assert payload["latest_step"]["action"] == 2
    assert payload["steps"][-1]["step"] == 3


def test_build_state_payload_marks_eval_job_running(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    object.__setattr__(config.experiment, "output_dir", str(tmp_path))
    run_dir = tmp_path / "ui_eval"
    (run_dir / "metrics").mkdir(parents=True)
    state = UiState(
        config_path=tmp_path / "config.yaml",
        config=config,
        experiment="ui_eval",
        default_steps=10,
        learning_starts=0,
        batch_size=8,
        dashboard_every=5,
        run_dir=run_dir,
        eval_job=EvalJob(
            run_name="ui_eval",
            run_dir=run_dir,
            config=config,
            algorithm=config.agent.algorithm,
            episodes_target=3,
        ),
    )

    payload = build_state_payload(state)

    assert payload["eval"]["status"] == "running"
    assert payload["eval"]["running"] is True
    assert payload["eval"]["episode_count"] == 0
    assert payload["eval"]["episodes_target"] == 3


def test_get_defaults_returns_ui_state_values(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    object.__setattr__(config.experiment, "output_dir", str(tmp_path))
    run_dir = tmp_path / "my_run"
    state = UiState(
        config_path=tmp_path / "config.yaml",
        config=config,
        experiment="my_run",
        default_steps=200,
        learning_starts=10,
        batch_size=32,
        dashboard_every=7,
        run_dir=run_dir,
    )

    defaults = _get_defaults(state)

    assert defaults["run_name"] == "my_run"
    assert defaults["default_steps"] == 200
    assert defaults["learning_starts"] == 10
    assert defaults["batch_size"] == 32
    assert defaults["dashboard_every"] == 7


def test_run_scoped_model_overrides_do_not_mutate_game_config() -> None:
    config = load_config("configs/games/breakout.yaml")
    updated = _apply_config_overrides(
        config,
        [
            {"group": "agent", "key": "algorithm", "value": "frozen_jepa_dqn"},
            {"group": "observation", "key": "width", "value": "96"},
            {"group": "world_model", "key": "latent_dim", "value": "64"},
            {"group": "agent", "key": "optimizer.lr", "value": "0.0002"},
        ],
    )

    assert config.agent.algorithm == "dqn"
    assert config.observation.width == 160
    assert config.world_model.latent_dim == 512
    assert updated.agent.algorithm == "frozen_jepa_dqn"
    assert updated.observation.width == 96
    assert updated.world_model.latent_dim == 64
    assert updated.agent.optimizer.lr == 0.0002
    assert updated.game.name == "breakout"


def test_existing_run_start_rejects_locked_model_overrides() -> None:
    assert _has_locked_start_overrides(
        {"overrides": [{"group": "agent", "key": "algorithm", "value": "linear_q"}]}
    )
    assert _has_locked_start_overrides({"batch_size": 32})
    assert _has_locked_start_overrides({"learning_starts": 0})
    assert _has_locked_start_overrides({"lr": 0.01})
    assert not _has_locked_start_overrides({"steps": 100, "dashboard_every": 5})


def test_config_list_contains_one_entry_per_game() -> None:
    names = {entry["name"] for entry in list_configs()["configs"]}

    assert {"breakout", "snake", "asteroids"}.issubset(names)
    assert "breakout_tiny" not in names
    assert "breakout_frozen_jepa" not in names


def test_list_runs_uses_snapshot_algorithm_before_training(tmp_path) -> None:
    config = load_config("configs/games/breakout.yaml")
    object.__setattr__(config.experiment, "output_dir", str(tmp_path))
    run_dir = tmp_path / "created_only"
    run_dir.mkdir()
    run_config = _apply_config_overrides(
        config,
        [{"group": "agent", "key": "algorithm", "value": "frozen_jepa_dqn"}],
    )
    snapshot_config(run_config, run_dir / "config.yaml")
    state = UiState(
        config_path=tmp_path / "config.yaml",
        config=config,
        experiment="created_only",
        default_steps=10,
        learning_starts=0,
        batch_size=8,
        dashboard_every=5,
        run_dir=run_dir,
    )

    runs = list_runs(state)["runs"]

    assert runs[0]["name"] == "created_only"
    assert runs[0]["algorithm"] == "frozen_jepa_dqn"

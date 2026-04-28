from __future__ import annotations

import json

from jepa_rl.ui.server import (
    EvalJob,
    UiState,
    _get_defaults,
    build_state_payload,
)
from jepa_rl.utils.config import load_config


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

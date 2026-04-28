from __future__ import annotations

import json
from pathlib import Path

from jepa_rl.ui.server import UiState, build_state_payload, render_ui_html
from jepa_rl.utils.config import load_config


def test_render_ui_html_contains_game_and_training_controls() -> None:
    config = load_config("configs/games/breakout.yaml")
    state = UiState(
        config_path=config.__class__.__name__,
        config=config,
        experiment="ui_test",
        default_steps=100,
        learning_starts=0,
        batch_size=8,
        dashboard_every=5,
        run_dir=Path(config.experiment.output_dir) / "ui_test",
    )

    html = render_ui_html(state)

    assert 'src="/game"' in html
    assert "startTraining" in html
    assert "stopTraining" in html
    assert "loss" in html


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
    assert len(payload["episodes"]) == 1

from __future__ import annotations

import json

from jepa_rl.utils.dashboard import write_training_dashboard


def test_write_training_dashboard_creates_self_contained_html(tmp_path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "steps": 2,
                "episodes": 1,
                "best_score": 6,
                "mean_loss": 0.5,
                "mean_td_error": 0.25,
                "update_count": 2,
                "replay_size": 2,
                "weight_delta_norm": 0.1,
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "train_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "step",
                        "step": 1,
                        "score": 0,
                        "loss": 1.0,
                        "epsilon": 1.0,
                        "td_error": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "type": "episode",
                        "episode": 0,
                        "step": 2,
                        "return": 6,
                        "score": 6,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    dashboard = write_training_dashboard(run_dir)

    html = dashboard.read_text(encoding="utf-8")
    assert "jepa-rl" in html
    assert "best score" in html.lower()
    assert "static snapshot" in html

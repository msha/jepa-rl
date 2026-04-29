from __future__ import annotations

import json
from types import SimpleNamespace

import scripts.compare_baselines as compare_baselines
from jepa_rl.utils.config import load_config
from scripts.compare_baselines import (
    EvaluationResult,
    aggregate_results,
    assess_regression,
    run_phase7_comparison,
    write_sample_efficiency_plot,
    write_summary_markdown,
)


def test_phase7_aggregation_and_regression_check() -> None:
    results = (
        EvaluationResult(
            suite="primary",
            variant="pixel_dqn",
            seed=0,
            budget=100,
            run_dir="runs/pixel",
            checkpoint="runs/pixel/checkpoints/best.pt",
            jepa_checkpoint=None,
            episodes=2,
            best_score=10.0,
            mean_score=10.0,
            median_score=10.0,
            p95_score=10.0,
        ),
        EvaluationResult(
            suite="primary",
            variant="frozen_random_jepa_dqn",
            seed=0,
            budget=100,
            run_dir="runs/frozen",
            checkpoint="runs/frozen/checkpoints/best.pt",
            jepa_checkpoint="runs/jepa/checkpoints/latest.pt",
            episodes=2,
            best_score=9.5,
            mean_score=9.5,
            median_score=9.5,
            p95_score=9.5,
        ),
    )

    aggregates = aggregate_results(results)
    checks = assess_regression(
        aggregates,
        baseline_variant="pixel_dqn",
        candidate_variant="frozen_random_jepa_dqn",
        tolerance=0.10,
    )

    assert len(aggregates) == 2
    assert len(checks) == 1
    assert checks[0]["passed"] is True
    assert checks[0]["relative_gap"] == -0.05


def test_phase7_report_artifacts_are_written(tmp_path) -> None:
    results = (
        EvaluationResult(
            suite="primary",
            variant="pixel_dqn",
            seed=0,
            budget=100,
            run_dir="runs/pixel",
            checkpoint="runs/pixel/checkpoints/best.pt",
            jepa_checkpoint=None,
            episodes=2,
            best_score=10.0,
            mean_score=10.0,
            median_score=10.0,
            p95_score=10.0,
        ),
        EvaluationResult(
            suite="primary",
            variant="frozen_random_jepa_dqn",
            seed=0,
            budget=100,
            run_dir="runs/frozen",
            checkpoint="runs/frozen/checkpoints/best.pt",
            jepa_checkpoint="runs/jepa/checkpoints/latest.pt",
            episodes=2,
            best_score=11.0,
            mean_score=11.0,
            median_score=11.0,
            p95_score=11.0,
        ),
    )
    aggregates = aggregate_results(results)
    checks = assess_regression(
        aggregates,
        baseline_variant="pixel_dqn",
        candidate_variant="frozen_random_jepa_dqn",
        tolerance=0.10,
    )
    summary_path = tmp_path / "summary.md"
    plot_path = tmp_path / "sample_efficiency.png"
    results_path = tmp_path / "results.json"

    write_summary_markdown(
        summary_path,
        status="pass",
        variants=("pixel_dqn", "frozen_random_jepa_dqn"),
        seeds=(0,),
        budgets=(100,),
        aggregates=aggregates,
        regression_checks=checks,
        skipped=(),
    )
    write_sample_efficiency_plot(plot_path, aggregates)
    results_path.write_text(
        json.dumps({"results": [result.to_dict() for result in results]}) + "\n",
        encoding="utf-8",
    )

    assert "Phase 7 Sample Efficiency Comparison" in summary_path.read_text(encoding="utf-8")
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["variant"] == "pixel_dqn"


def test_run_phase7_comparison_writes_report_with_fixed_budgets(monkeypatch, tmp_path) -> None:
    config = load_config("tests/fixtures/configs/joint_jepa_dqn_smoke.yaml")
    jepa_checkpoint = tmp_path / "jepa.pt"
    jepa_checkpoint.write_bytes(b"fake")

    def _fake_train(cfg, **kwargs):
        run_dir = tmp_path / kwargs["experiment"]
        (run_dir / "checkpoints").mkdir(parents=True)
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint.write_bytes(b"fake")
        return SimpleNamespace(run_dir=run_dir, best_checkpoint=checkpoint)

    def _fake_eval_pixel(*args, **kwargs):
        return {
            "episodes": 2,
            "best_score": 10.0,
            "mean_score": 10.0,
            "median_score": 10.0,
            "p95_score": 10.0,
        }

    def _fake_eval_frozen(*args, **kwargs):
        return {
            "episodes": 2,
            "best_score": 9.5,
            "mean_score": 9.5,
            "median_score": 9.5,
            "p95_score": 9.5,
        }

    monkeypatch.setattr(compare_baselines, "train_dqn", _fake_train)
    monkeypatch.setattr(compare_baselines, "train_frozen_jepa_dqn", _fake_train)
    monkeypatch.setattr(compare_baselines, "evaluate_dqn", _fake_eval_pixel)
    monkeypatch.setattr(compare_baselines, "evaluate_frozen_jepa_dqn", _fake_eval_frozen)

    report = run_phase7_comparison(
        config,
        output_dir=tmp_path / "comparison",
        variants=("pixel_dqn", "frozen_random_jepa_dqn"),
        seeds=(0,),
        budgets=(100,),
        eval_episodes=2,
        random_jepa_checkpoint=jepa_checkpoint,
    )

    assert report.status == "pass"
    assert report.summary_path.exists()
    assert report.plot_path.exists()
    assert len(report.results) == 2

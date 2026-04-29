from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from jepa_rl.envs.browser_game_env import BrowserGameEnv
from jepa_rl.training.frozen_jepa_dqn import evaluate_frozen_jepa_dqn, train_frozen_jepa_dqn
from jepa_rl.training.jepa_world import train_jepa_world
from jepa_rl.training.pixel_dqn import evaluate_dqn, train_dqn
from jepa_rl.utils.config import ProjectConfig, load_config

PHASE7_VARIANTS = (
    "pixel_dqn",
    "frozen_random_jepa_dqn",
    "frozen_passive_jepa_dqn",
)
DEFAULT_OUTPUT_DIR = Path("runs/comparison_phase7")


@dataclass(frozen=True)
class EvaluationResult:
    suite: str
    variant: str
    seed: int
    budget: int
    run_dir: str
    checkpoint: str
    jepa_checkpoint: str | None
    episodes: int
    best_score: float
    mean_score: float
    median_score: float
    p95_score: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    output_dir: Path
    results: tuple[EvaluationResult, ...]
    aggregates: tuple[dict[str, Any], ...]
    regression_checks: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, str], ...]
    status: str
    summary_path: Path
    results_path: Path
    plot_path: Path


def run_phase7_comparison(
    config: ProjectConfig,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    variants: Sequence[str] = ("pixel_dqn", "frozen_random_jepa_dqn"),
    seeds: Sequence[int] = (0, 1, 2),
    budgets: Sequence[int] | None = None,
    eval_episodes: int | None = None,
    random_jepa_checkpoint: Path | None = None,
    passive_jepa_checkpoint: Path | None = None,
    random_jepa_steps: int | None = None,
    random_jepa_collect_steps: int | None = None,
    learning_starts: int | None = None,
    batch_size: int | None = None,
    headless: bool | None = None,
    dashboard_every: int = 0,
    tolerance: float = 0.10,
    suite: str = "primary",
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None = None,
) -> ComparisonReport:
    _validate_variants(variants)
    if not seeds:
        raise ValueError("at least one seed is required")

    budget_values = tuple(budgets or config.training.eval_budgets)
    if not budget_values or any(budget <= 0 for budget in budget_values):
        raise ValueError("budgets must be positive integers")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, str]] = []
    results: list[EvaluationResult] = []
    jepa_cache: dict[tuple[str, int], Path] = {}

    for seed in seeds:
        for variant in variants:
            if variant == "frozen_passive_jepa_dqn" and passive_jepa_checkpoint is None:
                skipped.append(
                    {
                        "suite": suite,
                        "variant": variant,
                        "seed": str(seed),
                        "reason": (
                            "passive JEPA checkpoint not provided; Phase 13 creates this artifact"
                        ),
                    }
                )
                continue

            jepa_checkpoint = _checkpoint_for_variant(
                config,
                output_dir=output_dir,
                variant=variant,
                seed=seed,
                random_jepa_checkpoint=random_jepa_checkpoint,
                passive_jepa_checkpoint=passive_jepa_checkpoint,
                random_jepa_steps=random_jepa_steps,
                random_jepa_collect_steps=random_jepa_collect_steps,
                headless=headless,
                dashboard_every=dashboard_every,
                env_factory=env_factory,
                cache=jepa_cache,
            )
            for budget in budget_values:
                results.append(
                    _train_and_eval_budget(
                        config,
                        output_dir=output_dir,
                        suite=suite,
                        variant=variant,
                        seed=seed,
                        budget=budget,
                        eval_episodes=eval_episodes or config.evaluation.episodes,
                        jepa_checkpoint=jepa_checkpoint,
                        learning_starts=learning_starts,
                        batch_size=batch_size,
                        headless=headless,
                        dashboard_every=dashboard_every,
                        env_factory=env_factory,
                    )
                )

    aggregates = aggregate_results(results)
    regression_checks = assess_regression(
        aggregates,
        baseline_variant="pixel_dqn",
        candidate_variant="frozen_random_jepa_dqn",
        tolerance=tolerance,
    )
    status = "pass"
    if any(not check["passed"] for check in regression_checks):
        status = "fail"
    if not regression_checks:
        status = "incomplete"

    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    plot_path = output_dir / "sample_efficiency.png"
    payload = {
        "status": status,
        "variants": list(variants),
        "seeds": list(seeds),
        "budgets": list(budget_values),
        "results": [result.to_dict() for result in results],
        "aggregates": list(aggregates),
        "regression_checks": list(regression_checks),
        "skipped": skipped,
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(
        summary_path,
        status=status,
        variants=variants,
        seeds=seeds,
        budgets=budget_values,
        aggregates=aggregates,
        regression_checks=regression_checks,
        skipped=skipped,
    )
    write_sample_efficiency_plot(plot_path, aggregates)

    return ComparisonReport(
        output_dir=output_dir,
        results=tuple(results),
        aggregates=tuple(aggregates),
        regression_checks=tuple(regression_checks),
        skipped=tuple(skipped),
        status=status,
        summary_path=summary_path,
        results_path=results_path,
        plot_path=plot_path,
    )


def aggregate_results(results: Iterable[EvaluationResult]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str, int], list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault((result.suite, result.variant, result.budget), []).append(result)

    aggregates = []
    for (suite, variant, budget), rows in sorted(grouped.items()):
        mean_scores = [row.mean_score for row in rows]
        best_scores = [row.best_score for row in rows]
        aggregates.append(
            {
                "suite": suite,
                "variant": variant,
                "budget": budget,
                "seeds": [row.seed for row in rows],
                "mean_score": mean(mean_scores),
                "median_score": median(mean_scores),
                "best_score": max(best_scores),
                "num_runs": len(rows),
            }
        )
    return tuple(aggregates)


def assess_regression(
    aggregates: Sequence[dict[str, Any]],
    *,
    baseline_variant: str,
    candidate_variant: str,
    tolerance: float,
) -> tuple[dict[str, Any], ...]:
    by_key = {
        (row["suite"], row["variant"], row["budget"]): row
        for row in aggregates
    }
    checks: list[dict[str, Any]] = []
    suites = sorted({row["suite"] for row in aggregates})
    for suite in suites:
        budgets = sorted(
            {
                row["budget"]
                for row in aggregates
                if row["suite"] == suite and row["variant"] == baseline_variant
            }
        )
        for budget in budgets:
            baseline = by_key.get((suite, baseline_variant, budget))
            candidate = by_key.get((suite, candidate_variant, budget))
            if baseline is None or candidate is None:
                continue
            baseline_score = float(baseline["mean_score"])
            candidate_score = float(candidate["mean_score"])
            denominator = max(abs(baseline_score), 1.0)
            relative_gap = (candidate_score - baseline_score) / denominator
            checks.append(
                {
                    "suite": suite,
                    "budget": budget,
                    "baseline_variant": baseline_variant,
                    "candidate_variant": candidate_variant,
                    "baseline_mean_score": baseline_score,
                    "candidate_mean_score": candidate_score,
                    "relative_gap": relative_gap,
                    "tolerance": tolerance,
                    "passed": relative_gap >= -tolerance,
                }
            )
    return tuple(checks)


def write_summary_markdown(
    path: Path,
    *,
    status: str,
    variants: Sequence[str],
    seeds: Sequence[int],
    budgets: Sequence[int],
    aggregates: Sequence[dict[str, Any]],
    regression_checks: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, str]],
) -> None:
    lines = [
        "# Phase 7 Sample Efficiency Comparison",
        "",
        f"Status: **{status}**",
        "",
        f"Variants: `{', '.join(variants)}`",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Budgets: `{', '.join(str(budget) for budget in budgets)}`",
        "",
        "## Aggregate Scores",
        "",
        "| suite | variant | budget | runs | mean score | median score | best score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {suite} | {variant} | {budget} | {num_runs} | "
            "{mean_score:.3f} | {median_score:.3f} | {best_score:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Regression Checks",
            "",
            "| suite | budget | baseline | candidate | gap | tolerance | passed |",
            "|---|---:|---|---|---:|---:|---|",
        ]
    )
    if regression_checks:
        for check in regression_checks:
            lines.append(
                "| {suite} | {budget} | {baseline_variant} | {candidate_variant} | "
                "{relative_gap:.3f} | {tolerance:.3f} | {passed} |".format(**check)
            )
    else:
        lines.append("| primary | n/a | pixel_dqn | frozen_random_jepa_dqn | n/a | n/a | n/a |")

    if skipped:
        lines.extend(["", "## Skipped", ""])
        for row in skipped:
            lines.append(
                f"- `{row['variant']}` seed `{row['seed']}` in `{row['suite']}`: "
                f"{row['reason']}"
            )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sample_efficiency_plot(path: Path, aggregates: Sequence[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 960, 540
    margin_left, margin_right, margin_top, margin_bottom = 82, 32, 42, 76
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    plot_rows = [row for row in aggregates if row["suite"] == "primary"]
    if not plot_rows:
        draw.text((margin_left, margin_top), "No primary comparison data", fill="black", font=font)
        image.save(path)
        return

    variants = sorted({row["variant"] for row in plot_rows})
    budgets = sorted({row["budget"] for row in plot_rows})
    scores = [float(row["mean_score"]) for row in plot_rows]
    y_min = min(0.0, min(scores))
    y_max = max(scores)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)
    draw.text((x0, 16), "Phase 7 sample efficiency", fill="black", font=font)
    draw.text((x0, height - 34), "Environment steps", fill="black", font=font)
    draw.text((8, y1), "Mean eval score", fill="black", font=font)

    def x_pos(budget: int) -> float:
        if len(budgets) == 1:
            return (x0 + x1) / 2
        idx = budgets.index(budget)
        return x0 + idx * (x1 - x0) / (len(budgets) - 1)

    def y_pos(score: float) -> float:
        return y0 - (score - y_min) * (y0 - y1) / (y_max - y_min)

    for budget in budgets:
        x = x_pos(budget)
        draw.line((x, y0, x, y0 + 5), fill="black")
        draw.text((x - 24, y0 + 10), _format_budget(budget), fill="black", font=font)

    for idx in range(5):
        value = y_min + idx * (y_max - y_min) / 4
        y = y_pos(value)
        draw.line((x0 - 5, y, x0, y), fill="black")
        draw.text((18, y - 5), f"{value:.1f}", fill="black", font=font)

    colors = {
        "pixel_dqn": (31, 119, 180),
        "frozen_random_jepa_dqn": (44, 160, 44),
        "frozen_passive_jepa_dqn": (214, 39, 40),
    }
    row_map = {(row["variant"], row["budget"]): row for row in plot_rows}
    for variant_index, variant in enumerate(variants):
        points = []
        for budget in budgets:
            row = row_map.get((variant, budget))
            if row is not None:
                points.append((x_pos(budget), y_pos(float(row["mean_score"]))))
        if not points:
            continue
        color = colors.get(variant, (100, 100, 100))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_y = margin_top + variant_index * 18
        draw.line((width - 270, legend_y + 6, width - 246, legend_y + 6), fill=color, width=3)
        draw.text((width - 240, legend_y), variant, fill="black", font=font)

    image.save(path)


def _train_and_eval_budget(
    config: ProjectConfig,
    *,
    output_dir: Path,
    suite: str,
    variant: str,
    seed: int,
    budget: int,
    eval_episodes: int,
    jepa_checkpoint: Path | None,
    learning_starts: int | None,
    batch_size: int | None,
    headless: bool | None,
    dashboard_every: int,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None,
) -> EvaluationResult:
    algorithm = "dqn" if variant == "pixel_dqn" else "frozen_jepa_dqn"
    cfg = _replace_seed_algorithm_output(config, seed, algorithm, output_dir)
    run_name = f"{suite}_{variant}_seed{seed}_steps{budget}"
    if variant == "pixel_dqn":
        summary = train_dqn(
            cfg,
            experiment=run_name,
            steps=budget,
            learning_starts=learning_starts,
            headless=headless,
            batch_size=batch_size,
            dashboard_every=dashboard_every,
            env_factory=env_factory,
        )
        eval_out = evaluate_dqn(
            cfg,
            checkpoint=summary.best_checkpoint,
            episodes=eval_episodes,
            headless=headless,
            env_factory=env_factory,
            run_dir=summary.run_dir,
        )
    else:
        if jepa_checkpoint is None:
            raise ValueError(f"{variant} requires a JEPA checkpoint")
        summary = train_frozen_jepa_dqn(
            cfg,
            jepa_checkpoint=jepa_checkpoint,
            experiment=run_name,
            steps=budget,
            learning_starts=learning_starts,
            headless=headless,
            batch_size=batch_size,
            dashboard_every=dashboard_every,
            env_factory=env_factory,
        )
        eval_out = evaluate_frozen_jepa_dqn(
            cfg,
            jepa_checkpoint=jepa_checkpoint,
            checkpoint=summary.best_checkpoint,
            episodes=eval_episodes,
            headless=headless,
            env_factory=env_factory,
            run_dir=summary.run_dir,
        )

    return EvaluationResult(
        suite=suite,
        variant=variant,
        seed=seed,
        budget=budget,
        run_dir=str(summary.run_dir),
        checkpoint=str(summary.best_checkpoint),
        jepa_checkpoint=str(jepa_checkpoint) if jepa_checkpoint else None,
        episodes=int(eval_out["episodes"]),
        best_score=float(eval_out["best_score"]),
        mean_score=float(eval_out["mean_score"]),
        median_score=float(eval_out["median_score"]),
        p95_score=float(eval_out["p95_score"]),
    )


def _checkpoint_for_variant(
    config: ProjectConfig,
    *,
    output_dir: Path,
    variant: str,
    seed: int,
    random_jepa_checkpoint: Path | None,
    passive_jepa_checkpoint: Path | None,
    random_jepa_steps: int | None,
    random_jepa_collect_steps: int | None,
    headless: bool | None,
    dashboard_every: int,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None,
    cache: dict[tuple[str, int], Path],
) -> Path | None:
    if variant == "pixel_dqn":
        return None
    if variant == "frozen_passive_jepa_dqn":
        return passive_jepa_checkpoint
    if variant != "frozen_random_jepa_dqn":
        raise ValueError(f"unknown Phase 7 variant: {variant}")
    if random_jepa_checkpoint is not None:
        return random_jepa_checkpoint

    cache_key = ("random", seed)
    if cache_key in cache:
        return cache[cache_key]

    cfg = _replace_seed_algorithm_output(config, seed, "dqn", output_dir)
    steps = (
        random_jepa_steps
        if random_jepa_steps is not None
        else max(1, config.training.passive_pretrain_steps)
    )
    summary = train_jepa_world(
        cfg,
        experiment=f"primary_random_jepa_pretrain_seed{seed}",
        steps=steps,
        collect_steps=random_jepa_collect_steps,
        headless=headless,
        dashboard_every=dashboard_every,
        env_factory=env_factory,
    )
    cache[cache_key] = summary.checkpoint
    return summary.checkpoint


def _replace_seed_algorithm_output(
    config: ProjectConfig,
    seed: int,
    algorithm: str,
    output_dir: Path,
) -> ProjectConfig:
    exp = dataclasses.replace(
        config.experiment,
        seed=seed,
        output_dir=str(output_dir),
    )
    agent = dataclasses.replace(config.agent, algorithm=algorithm)
    return dataclasses.replace(config, experiment=exp, agent=agent)


def _validate_variants(variants: Sequence[str]) -> None:
    unknown = [variant for variant in variants if variant not in PHASE7_VARIANTS]
    if unknown:
        raise ValueError(f"unknown Phase 7 variants: {', '.join(unknown)}")


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    return values


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    _validate_variants(values)
    return values


def _format_budget(budget: int) -> str:
    if budget >= 1_000_000:
        value = budget / 1_000_000
        return f"{value:g}M"
    if budget >= 1_000:
        value = budget / 1_000
        return f"{value:g}k"
    return str(budget)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 7 pixel-DQN vs frozen-JEPA+DQN sample-efficiency comparison."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/games/breakout.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--configs",
        type=_parse_csv_strings,
        default=("pixel_dqn", "frozen_random_jepa_dqn"),
        help=(
            "Comma-separated variants. Supported: "
            "pixel_dqn,frozen_random_jepa_dqn,frozen_passive_jepa_dqn"
        ),
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument(
        "--budgets",
        type=_parse_csv_ints,
        default=None,
        help="Comma-separated step budgets. Defaults to training.eval_budgets.",
    )
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--random-jepa-checkpoint",
        type=Path,
        default=None,
        help="Existing random-data JEPA checkpoint. If omitted, train-world creates one per seed.",
    )
    parser.add_argument(
        "--passive-jepa-checkpoint",
        type=Path,
        default=None,
        help="Passive-video JEPA checkpoint from Phase 13 for frozen_passive_jepa_dqn.",
    )
    parser.add_argument("--random-jepa-steps", type=int, default=None)
    parser.add_argument("--random-jepa-collect-steps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dashboard-every", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="Write the report but return exit code 0 even if frozen JEPA underperforms.",
    )
    parser.add_argument(
        "--transfer-config",
        type=Path,
        default=None,
        help="Optional second-game config for frozen-encoder transfer reporting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    report = run_phase7_comparison(
        config,
        output_dir=args.output_dir,
        variants=args.configs,
        seeds=args.seeds,
        budgets=args.budgets,
        eval_episodes=args.eval_episodes,
        random_jepa_checkpoint=args.random_jepa_checkpoint,
        passive_jepa_checkpoint=args.passive_jepa_checkpoint,
        random_jepa_steps=args.random_jepa_steps,
        random_jepa_collect_steps=args.random_jepa_collect_steps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        headless=not args.headed,
        dashboard_every=args.dashboard_every,
        tolerance=args.tolerance,
    )

    transfer_report = None
    if args.transfer_config is not None:
        if (
            "frozen_random_jepa_dqn" in args.configs
            and args.random_jepa_checkpoint is None
        ):
            print(
                "transfer comparison requires --random-jepa-checkpoint so the second game "
                "uses the same frozen encoder with a fresh policy head",
                file=sys.stderr,
            )
            return 2
        transfer_config = load_config(args.transfer_config)
        transfer_report = run_phase7_comparison(
            transfer_config,
            output_dir=args.output_dir / "transfer",
            variants=args.configs,
            seeds=args.seeds,
            budgets=args.budgets,
            eval_episodes=args.eval_episodes,
            random_jepa_checkpoint=args.random_jepa_checkpoint,
            passive_jepa_checkpoint=args.passive_jepa_checkpoint,
            random_jepa_steps=args.random_jepa_steps,
            random_jepa_collect_steps=args.random_jepa_collect_steps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            headless=not args.headed,
            dashboard_every=args.dashboard_every,
            tolerance=args.tolerance,
            suite="transfer",
        )

    print(f"comparison status={report.status}")
    print(f"summary={report.summary_path}")
    print(f"plot={report.plot_path}")
    if transfer_report is not None:
        print(f"transfer_status={transfer_report.status}")
        print(f"transfer_summary={transfer_report.summary_path}")
    if report.status == "fail" and not args.allow_regression:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

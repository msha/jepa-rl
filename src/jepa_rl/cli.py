from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.config import ConfigError, load_config, snapshot_config


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    print(f"config valid: {args.config}")
    print(
        "summary: "
        f"experiment={config.experiment.name} "
        f"game={config.game.name} "
        f"obs={config.observation.width}x{config.observation.height}x"
        f"{config.observation.input_channels} "
        f"actions={config.actions.num_actions} "
        f"agent={config.agent.algorithm}"
    )
    return 0


def _cmd_init_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        run_dir = create_run_dir(
            config.experiment.output_dir, args.experiment or config.experiment.name
        )
        snapshot_config(config, run_dir / "config.yaml")
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    print(f"created run directory: {run_dir}")
    return 0


def _cmd_collect_random(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import BrowserEnvError, PlaywrightBrowserGameEnv

    try:
        config = load_config(args.config)
        run_dir = create_run_dir(
            config.experiment.output_dir, args.experiment or f"{config.game.name}_random"
        )
        snapshot_config(config, run_dir / "config.yaml")
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    import json
    import random

    rng = random.Random(config.experiment.seed)
    metrics_path = run_dir / "metrics" / "random_events.jsonl"
    try:
        with (
            PlaywrightBrowserGameEnv(config, headless=not args.headed) as env,
            metrics_path.open("w", encoding="utf-8") as metrics,
        ):
            for episode in range(args.episodes):
                env.reset()
                episode_return = 0.0
                score = 0.0
                steps_taken = 0
                for step_index in range(1, args.max_steps + 1):
                    steps_taken = step_index
                    action = rng.randrange(config.actions.num_actions)
                    result = env.step(action)
                    episode_return += result.reward
                    score = result.score
                    if result.done:
                        break
                metrics.write(
                    json.dumps(
                        {
                            "type": "episode",
                            "episode": episode,
                            "return": episode_return,
                            "score": score,
                            "steps": steps_taken,
                        }
                    )
                    + "\n"
                )
    except BrowserEnvError as exc:
        print(f"browser error: {exc}", file=sys.stderr)
        return 2

    print(f"random collection complete: run_dir={run_dir} episodes={args.episodes}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import BrowserEnvError
    from jepa_rl.training.simple_q import train_linear_q

    try:
        config = load_config(args.config)
        summary = train_linear_q(
            config,
            experiment=args.experiment,
            steps=args.steps,
            learning_starts=args.learning_starts,
            lr=args.lr,
            headless=not args.headed,
        )
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2
    except (BrowserEnvError, ValueError) as exc:
        print(f"train failed: {exc}", file=sys.stderr)
        return 2

    print(
        "training complete: "
        f"run_dir={summary.run_dir} "
        f"checkpoint={summary.checkpoint} "
        f"steps={summary.steps} "
        f"episodes={summary.episodes} "
        f"best_score={summary.best_score:.2f} "
        f"mean_loss={summary.mean_loss:.6f}"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import BrowserEnvError
    from jepa_rl.training.simple_q import evaluate_linear_q

    try:
        config = load_config(args.config)
        summary = evaluate_linear_q(
            config,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            headless=not args.headed,
        )
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2
    except (BrowserEnvError, ValueError, FileNotFoundError) as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        return 2

    print(
        "evaluation complete: "
        f"episodes={summary['episodes']} "
        f"best_score={summary['best_score']:.2f} "
        f"mean_score={summary['mean_score']:.2f} "
        f"median_score={summary['median_score']:.2f} "
        f"p95_score={summary['p95_score']:.2f}"
    )
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"`jepa-rl {args.command}` is planned but not implemented yet. "
        "Use `jepa-rl validate-config` and `jepa-rl init-run` in the current scaffold.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jepa-rl",
        description="JEPA-RL browser-game research CLI.",
    )
    parser.add_argument("--version", action="version", version="jepa-rl 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-config", help="Load and validate a YAML config.")
    validate.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    validate.set_defaults(func=_cmd_validate_config)

    init_run = subparsers.add_parser("init-run", help="Create a run directory and config snapshot.")
    init_run.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    init_run.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to experiment.name from config.",
    )
    init_run.set_defaults(func=_cmd_init_run)

    collect = subparsers.add_parser("collect-random", help="Run random actions in a browser game.")
    collect.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    collect.add_argument(
        "--experiment", type=str, default=None, help="Optional run directory name."
    )
    collect.add_argument("--episodes", type=int, default=1, help="Number of random episodes.")
    collect.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    collect.add_argument("--headed", action="store_true", help="Show the browser window.")
    collect.set_defaults(func=_cmd_collect_random)

    train_world = subparsers.add_parser("train-world", help="Planned train-world command.")
    train_world.set_defaults(func=_cmd_not_implemented, command="train-world")

    train = subparsers.add_parser("train", help="Train the current minimal pixel Q model.")
    train.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    train.add_argument("--experiment", type=str, default=None, help="Optional run directory name.")
    train.add_argument(
        "--steps", type=int, default=200, help="Number of browser environment steps."
    )
    train.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help="Number of environment steps before model updates begin.",
    )
    train.add_argument("--lr", type=float, default=None, help="Optional learning rate override.")
    train.add_argument("--headed", action="store_true", help="Show the browser window.")
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained linear Q checkpoint.")
    evaluate.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="Path to .npz checkpoint.")
    evaluate.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    evaluate.add_argument("--headed", action="store_true", help="Show the browser window.")
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))

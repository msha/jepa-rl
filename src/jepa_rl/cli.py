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


def _cmd_open_game(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import (
        BrowserClosedError,
        BrowserEnvError,
        PlaywrightBrowserGameEnv,
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2
    if args.seconds < 0:
        print("open-game failed: --seconds must be nonnegative", file=sys.stderr)
        return 2
    if args.random_steps < 0:
        print("open-game failed: --random-steps must be nonnegative", file=sys.stderr)
        return 2

    import random

    rng = random.Random(config.experiment.seed)
    try:
        with PlaywrightBrowserGameEnv(config, headless=False) as env:
            score = env.read_score()
            print(
                "browser opened: "
                f"game={config.game.name} "
                f"url={config.game.url} "
                f"score={score:.2f}"
            )

            for step in range(args.random_steps):
                action = rng.randrange(config.actions.num_actions)
                result = env.step(action)
                print(
                    "step "
                    f"{step + 1}: action={action} "
                    f"score={result.score:.2f} "
                    f"reward={result.reward:.2f} "
                    f"done={result.done}"
                )
                if result.done:
                    break

            if args.hold:
                print("holding browser open; press Ctrl+C to close")
                while True:
                    env.wait(1.0)
            else:
                print(f"holding browser open for {args.seconds:.1f}s")
                env.wait(args.seconds)
    except (KeyboardInterrupt, BrowserClosedError):
        print("browser closed")
        return 0
    except BrowserEnvError as exc:
        print(f"browser error: {exc}", file=sys.stderr)
        return 2

    print("browser closed")
    return 0


def _cmd_ml_smoke(args: argparse.Namespace) -> int:
    from jepa_rl.training.simple_q import run_linear_q_ml_smoke

    try:
        summary = run_linear_q_ml_smoke(seed=args.seed, steps=args.steps, lr=args.lr)
    except ValueError as exc:
        print(f"ml-smoke failed: {exc}", file=sys.stderr)
        return 2

    print(
        "ml smoke complete: "
        f"steps={summary.steps} "
        f"initial_loss={summary.initial_loss:.6f} "
        f"final_loss={summary.final_loss:.6f} "
        f"improvement={summary.improvement:.6f} "
        f"weight_delta_norm={summary.weight_delta_norm:.6f}"
    )
    if summary.final_loss >= summary.initial_loss:
        print("ml-smoke failed: loss did not improve", file=sys.stderr)
        return 2
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from jepa_rl.ui.server import run_ui_server

    try:
        run_ui_server(
            config_path=args.config,
            run_dir=args.run,
            host=args.host,
            port=args.port,
            experiment=args.experiment,
            steps=args.steps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            dashboard_every=args.dashboard_every,
            open_browser=not args.no_open,
        )
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ui failed: {exc}", file=sys.stderr)
        return 2
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
            PlaywrightBrowserGameEnv(
                config,
                headless=not args.headed,
                run_dir=run_dir,
                record_video=config.recording.enabled,
            ) as env,
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

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    try:
        algorithm = config.agent.algorithm
        if algorithm == "dqn":
            from jepa_rl.training.pixel_dqn import train_dqn

            summary = train_dqn(
                config,
                experiment=args.experiment,
                steps=args.steps,
                learning_starts=args.learning_starts,
                headless=not args.headed,
                batch_size=args.batch_size,
                dashboard_every=args.dashboard_every,
            )
        else:
            from jepa_rl.training.simple_q import train_linear_q

            summary = train_linear_q(
                config,
                experiment=args.experiment,
                steps=args.steps,
                learning_starts=args.learning_starts,
                lr=args.lr,
                headless=not args.headed,
                batch_size=args.batch_size,
                dashboard_every=args.dashboard_every,
            )
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
        f"mean_loss={summary.mean_loss:.6f} "
        f"updates={summary.update_count} "
        f"replay_size={summary.replay_size} "
        f"target_updates={summary.target_update_count} "
        f"weight_delta_norm={summary.weight_delta_norm:.6f} "
        f"dashboard={summary.dashboard}"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import BrowserEnvError

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    try:
        algorithm = config.agent.algorithm
        if algorithm == "dqn":
            from jepa_rl.training.pixel_dqn import evaluate_dqn

            summary = evaluate_dqn(
                config,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                headless=not args.headed,
            )
        else:
            from jepa_rl.training.simple_q import evaluate_linear_q

            summary = evaluate_linear_q(
                config,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                headless=not args.headed,
            )
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


def _cmd_train_world(args: argparse.Namespace) -> int:
    from jepa_rl.browser.playwright_env import BrowserEnvError

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config invalid: {exc}", file=sys.stderr)
        return 2

    try:
        from jepa_rl.training.jepa_world import train_jepa_world

        summary = train_jepa_world(
            config,
            experiment=args.experiment,
            steps=args.steps,
            collect_steps=args.collect_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            headless=not args.headed,
            dashboard_every=args.dashboard_every,
        )
    except (BrowserEnvError, ValueError) as exc:
        print(f"train-world failed: {exc}", file=sys.stderr)
        return 2

    print(
        "train-world complete: "
        f"run_dir={summary.run_dir} "
        f"checkpoint={summary.checkpoint} "
        f"steps={summary.steps} "
        f"device={summary.device} "
        f"replay={summary.replay_size} "
        f"initial_loss={summary.initial_loss:.6f} "
        f"final_loss={summary.final_loss:.6f} "
        f"improvement={summary.improvement:.6f} "
        f"latent_std={summary.latent_std_mean:.4f} "
        f"eff_rank={summary.effective_rank:.2f} "
        f"dashboard={summary.dashboard}"
    )
    return 0


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

    open_game = subparsers.add_parser(
        "open-game", help="Open the configured browser game in visible Chromium."
    )
    open_game.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    open_game.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Seconds to keep the browser open unless --hold is set.",
    )
    open_game.add_argument(
        "--random-steps",
        type=int,
        default=0,
        help="Optional random actions to execute after opening.",
    )
    open_game.add_argument(
        "--hold",
        action="store_true",
        help="Keep the browser open until Ctrl+C.",
    )
    open_game.set_defaults(func=_cmd_open_game)

    ml_smoke = subparsers.add_parser(
        "ml-smoke", help="Verify the linear Q learner reduces loss on a synthetic task."
    )
    ml_smoke.add_argument("--steps", type=int, default=2000, help="Number of SGD updates.")
    ml_smoke.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    ml_smoke.add_argument("--seed", type=int, default=0, help="Random seed.")
    ml_smoke.set_defaults(func=_cmd_ml_smoke)

    ui = subparsers.add_parser(
        "ui",
        help="Live training dashboard and control panel.",
    )
    ui.add_argument(
        "--config", type=Path, default=None, help="Config YAML file."
    )
    ui.add_argument(
        "--run", type=Path, default=None, help="Run directory to view."
    )
    ui.add_argument("--host", type=str, default="127.0.0.1", help="HTTP bind host.")
    ui.add_argument("--port", type=int, default=8765, help="HTTP bind port.")
    ui.add_argument("--experiment", type=str, default="ui_train", help="Default run name.")
    ui.add_argument("--steps", type=int, default=500, help="Default training steps.")
    ui.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help="Default environment steps before updates begin.",
    )
    ui.add_argument("--batch-size", type=int, default=16, help="Default replay minibatch size.")
    ui.add_argument(
        "--dashboard-every",
        type=int,
        default=5,
        help="Rewrite dashboard every N training steps.",
    )
    ui.add_argument("--open", action="store_true", help="Open in browser (default behavior).")
    ui.add_argument("--no-open", action="store_true", help="Do not open the browser UI.")
    ui.set_defaults(func=_cmd_ui)

    collect = subparsers.add_parser("collect-random", help="Run random actions in a browser game.")
    collect.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    collect.add_argument(
        "--experiment", type=str, default=None, help="Optional run directory name."
    )
    collect.add_argument("--episodes", type=int, default=1, help="Number of random episodes.")
    collect.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    collect.add_argument("--headed", action="store_true", help="Show the browser window.")
    collect.set_defaults(func=_cmd_collect_random)

    train_world = subparsers.add_parser(
        "train-world", help="Train the JEPA action-conditioned world model offline."
    )
    train_world.add_argument(
        "--config", type=Path, required=True, help="Path to a config YAML file."
    )
    train_world.add_argument(
        "--experiment", type=str, default=None, help="Optional run directory name."
    )
    train_world.add_argument(
        "--steps", type=int, default=1000, help="Number of gradient updates."
    )
    train_world.add_argument(
        "--collect-steps",
        type=int,
        default=None,
        help="Browser env steps for random data collection before training.",
    )
    train_world.add_argument(
        "--batch-size", type=int, default=None, help="Sequence batch size override."
    )
    train_world.add_argument(
        "--lr", type=float, default=None, help="Learning rate override."
    )
    train_world.add_argument(
        "--dashboard-every",
        type=int,
        default=25,
        help="Rewrite dashboard every N gradient steps. 0 = only at end.",
    )
    train_world.add_argument("--headed", action="store_true", help="Show the browser window.")
    train_world.set_defaults(func=_cmd_train_world)

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
    train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Replay minibatch size. Defaults to min(config batch_size, 64).",
    )
    train.add_argument(
        "--dashboard-every",
        type=int,
        default=25,
        help="Rewrite dashboard every N steps. Use 0 to write only at the end.",
    )
    train.add_argument("--headed", action="store_true", help="Show the browser window.")
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate a trained linear Q checkpoint.")
    evaluate.add_argument("--config", type=Path, required=True, help="Path to a config YAML file.")
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint (.pt for DQN/JEPA, .npz for linear_q).",
    )
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

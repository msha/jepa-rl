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

    for name in ("collect-random", "train-world", "train", "eval"):
        planned = subparsers.add_parser(name, help=f"Planned {name} command.")
        planned.set_defaults(func=_cmd_not_implemented, command=name)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))

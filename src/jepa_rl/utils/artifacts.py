from __future__ import annotations

from pathlib import Path

RUN_SUBDIRS = ("metrics", "checkpoints", "videos", "replay", "diagnostics")


def create_run_dir(output_dir: str | Path, experiment_name: str) -> Path:
    """Create the standard run artifact directory layout."""

    if not experiment_name.strip():
        raise ValueError("experiment name cannot be empty")
    run_dir = Path(output_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(exist_ok=True)
    return run_dir

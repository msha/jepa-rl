from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jepa_rl.utils.config import ProjectConfig


class JsonlWriter:
    """Append-only JSONL event log compatible with the training dashboard."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._file.write(json.dumps(event) + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_run_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def step_event(
    *,
    step: int,
    episode: int,
    action: int,
    reward: float,
    score: float,
    done: bool,
    epsilon: float,
    loss: float | None,
    td_error: float | None,
    q_max: float,
    replay_size: int,
    updates: int,
    target_updates: int,
    weight_delta_norm: float | None = None,
    grad_norm: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "step",
        "step": step,
        "episode": episode,
        "action": action,
        "reward": reward,
        "score": score,
        "done": done,
        "epsilon": epsilon,
        "loss": loss,
        "td_error": td_error,
        "q_max": q_max,
        "replay_size": replay_size,
        "updates": updates,
        "target_updates": target_updates,
        "weight_delta_norm": weight_delta_norm,
    }
    if grad_norm is not None:
        event["grad_norm"] = grad_norm
    event.update(extra)
    return event


def episode_event(
    *,
    step: int,
    episode: int,
    return_: float,
    score: float,
    **extra: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "episode",
        "step": step,
        "episode": episode,
        "return": return_,
        "score": score,
    }
    event.update(extra)
    return event


def build_run_summary(
    *,
    algorithm: str,
    steps: int,
    requested_steps: int,
    status: str,
    episodes: int,
    num_actions: int,
    update_count: int,
    mean_loss: float,
    mean_td_error: float,
    replay_size: int,
    target_update_count: int,
    weight_delta_norm: float,
    best_score: float,
    started_at: float,
    **extra: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "steps": steps,
        "requested_steps": requested_steps,
        "status": status,
        "episodes": episodes,
        "num_actions": num_actions,
        "update_count": update_count,
        "mean_loss": mean_loss,
        "mean_td_error": mean_td_error,
        "replay_size": replay_size,
        "target_update_count": target_update_count,
        "weight_delta_norm": weight_delta_norm,
        "best_score": best_score,
        "wall_time_sec": time.time() - started_at,
    }
    summary.update(extra)
    return summary


def linear_epsilon(config: ProjectConfig, step: int) -> float:
    schedule = config.exploration
    progress = min(1.0, step / max(1, schedule.epsilon_decay_steps))
    return schedule.epsilon_start + progress * (schedule.epsilon_end - schedule.epsilon_start)

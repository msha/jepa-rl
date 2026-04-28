from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointPayload:
    step: int
    episode: int
    update_count: int
    target_update_count: int
    best_score: float
    config_dict: dict[str, Any]
    rng_python: Any
    rng_numpy: Any
    rng_torch: Any
    extra: dict[str, Any] = field(default_factory=dict)


def save_torch_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    target_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    payload: CheckpointPayload,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "target_model": target_model.state_dict() if target_model is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": payload.step,
            "episode": payload.episode,
            "update_count": payload.update_count,
            "target_update_count": payload.target_update_count,
            "best_score": payload.best_score,
            "config_dict": payload.config_dict,
            "rng_python": payload.rng_python,
            "rng_numpy": payload.rng_numpy,
            "rng_torch": payload.rng_torch,
            "extra": payload.extra,
        },
        path,
    )


def load_torch_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    target_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    map_location: str | torch.device = "cpu",
) -> CheckpointPayload:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if target_model is not None and state.get("target_model") is not None:
        target_model.load_state_dict(state["target_model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return CheckpointPayload(
        step=state["step"],
        episode=state["episode"],
        update_count=state["update_count"],
        target_update_count=state["target_update_count"],
        best_score=state["best_score"],
        config_dict=state["config_dict"],
        rng_python=state["rng_python"],
        rng_numpy=state["rng_numpy"],
        rng_torch=state["rng_torch"],
        extra=state.get("extra", {}),
    )

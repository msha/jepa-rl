from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """Environment observation after configured preprocessing."""

    data: Any
    width: int
    height: int
    channels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result returned by a browser-game environment step."""

    observation: Observation
    reward: float
    done: bool
    score: float
    info: dict[str, Any] = field(default_factory=dict)


class BrowserGameEnv(ABC):
    """Common browser-game interface used by collectors and trainers."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the game and return the first observation."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Apply a discrete action and return the resulting transition."""

    @abstractmethod
    def observe(self) -> Observation:
        """Read the current observation without changing game state."""

    @abstractmethod
    def read_score(self) -> float:
        """Read the current score from the configured score reader."""

    @abstractmethod
    def is_done(self) -> bool:
        """Return whether the current episode is terminal."""

    @abstractmethod
    def render_video_frame(self) -> Any:
        """Return an RGB frame suitable for video writing."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class ActionSpaceError(ValueError):
    """Raised when an action space specification is invalid."""


@dataclass(frozen=True)
class KeyboardAction:
    """A discrete keyboard action represented by zero or more simultaneous keys."""

    index: int
    name: str
    keys: tuple[str, ...]

    @property
    def is_noop(self) -> bool:
        return len(self.keys) == 0


def parse_key_combo(spec: str) -> tuple[str, ...]:
    """Parse an action string such as ``ArrowLeft+Space`` into key names."""

    normalized = spec.strip()
    if not normalized:
        raise ActionSpaceError("action spec cannot be empty")
    if normalized.lower() == "noop":
        return ()

    parts = tuple(part.strip() for part in normalized.split("+"))
    if any(not part for part in parts):
        raise ActionSpaceError(f"invalid key combination: {spec!r}")
    if len(set(parts)) != len(parts):
        raise ActionSpaceError(f"duplicate key in combination: {spec!r}")
    return parts


class DiscreteKeyboardActionSpace:
    """Fixed discrete keyboard action space for V1 browser games."""

    def __init__(self, specs: Sequence[str]):
        if not specs:
            raise ActionSpaceError("discrete keyboard action space requires at least one action")
        if len(set(specs)) != len(specs):
            raise ActionSpaceError("action specs must be unique")

        self._actions = tuple(
            KeyboardAction(index=index, name=spec.strip(), keys=parse_key_combo(spec))
            for index, spec in enumerate(specs)
        )

    @classmethod
    def from_iterable(cls, specs: Iterable[str]) -> DiscreteKeyboardActionSpace:
        return cls(tuple(specs))

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions)

    @property
    def actions(self) -> tuple[KeyboardAction, ...]:
        return self._actions

    def get(self, index: int) -> KeyboardAction:
        try:
            return self._actions[index]
        except IndexError as exc:
            raise ActionSpaceError(f"action index out of range: {index}") from exc

    def sample(self, rng: random.Random | None = None) -> KeyboardAction:
        generator = rng or random
        return generator.choice(self._actions)

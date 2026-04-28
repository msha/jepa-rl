from __future__ import annotations

import random

import pytest

from jepa_rl.browser.action_spaces import (
    ActionSpaceError,
    DiscreteKeyboardActionSpace,
    parse_key_combo,
)


def test_parse_key_combo_handles_noop_and_combinations() -> None:
    assert parse_key_combo("noop") == ()
    assert parse_key_combo("ArrowLeft+Space") == ("ArrowLeft", "Space")
    assert parse_key_combo(" ArrowRight + Space ") == ("ArrowRight", "Space")


def test_parse_key_combo_rejects_invalid_specs() -> None:
    with pytest.raises(ActionSpaceError, match="empty"):
        parse_key_combo("")
    with pytest.raises(ActionSpaceError, match="invalid"):
        parse_key_combo("ArrowLeft+")
    with pytest.raises(ActionSpaceError, match="duplicate"):
        parse_key_combo("Space+Space")


def test_discrete_keyboard_action_space_indexes_actions() -> None:
    space = DiscreteKeyboardActionSpace(["noop", "ArrowLeft", "ArrowRight+Space"])

    assert len(space) == 3
    assert space.get(0).is_noop
    assert space.get(2).keys == ("ArrowRight", "Space")
    assert [action.index for action in space] == [0, 1, 2]


def test_discrete_keyboard_action_space_sample_is_seedable() -> None:
    space = DiscreteKeyboardActionSpace(["noop", "ArrowLeft", "ArrowRight"])
    rng = random.Random(7)

    assert space.sample(rng).name == "ArrowLeft"


def test_discrete_keyboard_action_space_rejects_duplicate_specs() -> None:
    with pytest.raises(ActionSpaceError, match="unique"):
        DiscreteKeyboardActionSpace(["noop", "noop"])

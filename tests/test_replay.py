from __future__ import annotations

import random

import pytest

from jepa_rl.replay import ReplayBuffer, ReplayError, Transition


def make_transition(index: int, *, episode_id: str = "ep-1", done: bool = False) -> Transition:
    return Transition(
        obs=f"obs-{index}",
        action=index % 3,
        reward=float(index),
        next_obs=f"obs-{index + 1}",
        done=done,
        score=float(index * 10),
        timestamp=index,
        game_id="game",
        episode_id=episode_id,
        metadata={"index": index},
    )


def test_replay_add_and_capacity_eviction() -> None:
    replay = ReplayBuffer(capacity=3)
    for index in range(5):
        replay.add(make_transition(index))

    assert len(replay) == 3
    assert {transition.timestamp for transition in replay.transitions} == {2, 3, 4}


def test_replay_sample_is_seedable() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.extend([make_transition(index) for index in range(5)])

    batch = replay.sample(2, random.Random(2))

    assert [transition.timestamp for transition in batch] == [0, 4]


def test_replay_sample_rejects_oversized_batch() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.add(make_transition(0))

    with pytest.raises(ReplayError, match="cannot sample"):
        replay.sample(2)


def test_sequence_sampling_stays_within_episode() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.extend(
        [
            make_transition(0, episode_id="ep-1"),
            make_transition(1, episode_id="ep-1", done=True),
            make_transition(2, episode_id="ep-2"),
            make_transition(3, episode_id="ep-2"),
            make_transition(4, episode_id="ep-2"),
        ]
    )

    sequences = replay.sample_sequence(1, 3, random.Random(1))

    assert [[transition.timestamp for transition in seq] for seq in sequences] == [[2, 3, 4]]


def test_sequence_sampling_can_cross_episode_when_explicitly_allowed() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.extend(
        [
            make_transition(0, episode_id="ep-1"),
            make_transition(1, episode_id="ep-1", done=True),
            make_transition(2, episode_id="ep-2"),
        ]
    )

    sequences = replay.sample_sequence(1, 3, random.Random(1), allow_cross_episode=True)

    assert [transition.timestamp for transition in sequences[0]] == [0, 1, 2]


def test_sequence_sampling_reports_no_valid_starts() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.extend(
        [
            make_transition(0, episode_id="ep-1", done=True),
            make_transition(1, episode_id="ep-2", done=True),
        ]
    )

    with pytest.raises(ReplayError, match="only 0 valid starts"):
        replay.sample_sequence(1, 2)

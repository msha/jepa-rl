from __future__ import annotations

from typing import Any

import numpy as np

from jepa_rl.envs.browser_game_env import BrowserGameEnv, Observation, StepResult


class ScriptedEnv(BrowserGameEnv):
    """Deterministic fake environment for fast smoke training and testing.

    The correct action for each step is `step_in_episode % num_actions`.
    The observation encodes the correct action as a uniform fill value across
    all pixels, so a pooling-based encoder receives a strong, unambiguous signal.
    Reward is +1.0 when the agent picks the correct action, else 0.0.
    """

    def __init__(
        self,
        *,
        num_actions: int,
        height: int = 32,
        width: int = 32,
        channels: int = 1,
        episode_length: int = 64,
        seed: int = 0,
    ) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be positive")
        self.num_actions = num_actions
        self.height = height
        self.width = width
        self.channels = channels
        self.episode_length = episode_length
        self._rng = np.random.default_rng(seed)
        self._step_in_episode = 0
        self._episode_score = 0.0
        self._done = False

    def _correct_action(self) -> int:
        return self._step_in_episode % self.num_actions

    def _make_obs_data(self) -> np.ndarray:
        frame = np.zeros((self.channels, self.height, self.width), dtype=np.uint8)
        if self.num_actions > 1:
            fill_value = int(255 * self._correct_action() / (self.num_actions - 1))
            frame[:] = fill_value
        return frame

    def _obs(self) -> Observation:
        return Observation(
            data=self._make_obs_data(),
            width=self.width,
            height=self.height,
            channels=self.channels,
        )

    def reset(self) -> Observation:
        self._step_in_episode = 0
        self._episode_score = 0.0
        self._done = False
        return self._obs()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("call reset() before stepping a done episode")
        correct = self._correct_action()
        reward = 1.0 if action == correct else 0.0
        self._episode_score += reward
        self._step_in_episode += 1
        self._done = self._step_in_episode >= self.episode_length
        return StepResult(
            observation=self._obs(),
            reward=reward,
            done=self._done,
            score=self._episode_score,
        )

    def observe(self) -> Observation:
        return self._obs()

    def read_score(self) -> float:
        return self._episode_score

    def is_done(self) -> bool:
        return self._done

    def render_video_frame(self) -> Any:
        data = self._make_obs_data()
        if self.channels == 1:
            rgb = np.repeat(data[0:1], 3, axis=0)
        else:
            rgb = data[:3]
        return np.transpose(rgb, (1, 2, 0))

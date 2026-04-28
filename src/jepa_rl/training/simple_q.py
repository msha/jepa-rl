from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv
from jepa_rl.envs.browser_game_env import Observation
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.config import ProjectConfig, snapshot_config


@dataclass(frozen=True)
class TrainSummary:
    run_dir: Path
    checkpoint: Path
    steps: int
    episodes: int
    best_score: float
    mean_loss: float


class LinearQModel:
    """Small linear Q model used for browser smoke training before PyTorch lands."""

    def __init__(self, feature_dim: int, num_actions: int, rng: np.random.Generator):
        self.weights = rng.normal(0.0, 0.001, size=(feature_dim, num_actions)).astype(np.float32)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights

    def update(self, features: np.ndarray, action: int, target: float, lr: float) -> float:
        q_values = self.q_values(features)
        error = target - float(q_values[action])
        self.weights[:, action] += np.float32(lr * error) * features
        return error * error

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path: Path) -> LinearQModel:
        data = np.load(path, allow_pickle=False)
        weights = data["weights"].astype(np.float32)
        model = cls.__new__(cls)
        model.weights = weights
        return model


def featurize_observation(
    observation: Observation, *, grid_h: int = 12, grid_w: int = 16
) -> np.ndarray:
    """Convert stacked pixels into a compact feature vector plus bias."""

    data = observation.data.astype(np.float32) / 255.0
    channels, height, width = data.shape
    row_idx = np.linspace(0, height - 1, min(grid_h, height)).astype(np.int64)
    col_idx = np.linspace(0, width - 1, min(grid_w, width)).astype(np.int64)
    sampled = data[:, row_idx, :][:, :, col_idx]
    features = sampled.reshape(channels * len(row_idx) * len(col_idx))
    return np.concatenate([features, np.ones(1, dtype=np.float32)]).astype(np.float32)


def train_linear_q(
    config: ProjectConfig,
    *,
    experiment: str | None = None,
    steps: int,
    learning_starts: int = 0,
    lr: float | None = None,
    headless: bool | None = None,
) -> TrainSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")

    run_name = experiment or config.experiment.name
    run_dir = create_run_dir(config.experiment.output_dir, run_name)
    snapshot_config(config, run_dir / "config.yaml")
    metrics_path = run_dir / "metrics" / "train_events.jsonl"
    summary_path = run_dir / "metrics" / "train_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest.npz"

    py_rng = random.Random(config.experiment.seed)
    np_rng = np.random.default_rng(config.experiment.seed)
    learning_rate = lr if lr is not None else config.agent.optimizer.lr

    with PlaywrightBrowserGameEnv(config, headless=headless) as env:
        obs = env.reset()
        model = LinearQModel(
            feature_dim=featurize_observation(obs).shape[0],
            num_actions=config.actions.num_actions,
            rng=np_rng,
        )

        episode_id = 0
        episode_return = 0.0
        episode_score = 0.0
        best_score = float("-inf")
        losses: list[float] = []
        started_at = time.time()

        with metrics_path.open("w", encoding="utf-8") as metrics:
            for step in range(steps):
                features = featurize_observation(obs)
                epsilon = _linear_epsilon(config, step)
                if py_rng.random() < epsilon:
                    action = py_rng.randrange(config.actions.num_actions)
                else:
                    action = int(np.argmax(model.q_values(features)))

                result = env.step(action)
                next_features = featurize_observation(result.observation)
                if step >= learning_starts:
                    bootstrap = 0.0 if result.done else float(np.max(model.q_values(next_features)))
                    target = result.reward + config.agent.gamma * bootstrap
                    losses.append(model.update(features, action, target, learning_rate))

                episode_return += result.reward
                episode_score = result.score
                event = {
                    "type": "step",
                    "step": step + 1,
                    "episode": episode_id,
                    "action": action,
                    "reward": result.reward,
                    "score": result.score,
                    "done": result.done,
                    "epsilon": epsilon,
                }
                metrics.write(json.dumps(event) + "\n")

                obs = result.observation
                if result.done:
                    best_score = max(best_score, episode_score)
                    metrics.write(
                        json.dumps(
                            {
                                "type": "episode",
                                "step": step + 1,
                                "episode": episode_id,
                                "return": episode_return,
                                "score": episode_score,
                            }
                        )
                        + "\n"
                    )
                    episode_id += 1
                    episode_return = 0.0
                    episode_score = 0.0
                    obs = env.reset()

        if episode_score or episode_return:
            best_score = max(best_score, episode_score)
        if best_score == float("-inf"):
            best_score = episode_score

        mean_loss = float(np.mean(losses)) if losses else 0.0
        metadata = {
            "algorithm": "linear_q_pixel_smoke",
            "steps": steps,
            "episodes": episode_id + 1,
            "num_actions": config.actions.num_actions,
            "feature_grid": [12, 16],
            "mean_loss": mean_loss,
            "best_score": best_score,
            "wall_time_sec": time.time() - started_at,
        }
        model.save(checkpoint_path, metadata)
        summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return TrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        steps=steps,
        episodes=metadata["episodes"],
        best_score=float(best_score),
        mean_loss=mean_loss,
    )


def evaluate_linear_q(
    config: ProjectConfig,
    *,
    checkpoint: Path,
    episodes: int,
    headless: bool | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    model = LinearQModel.load(checkpoint)
    scores: list[float] = []
    returns: list[float] = []
    with PlaywrightBrowserGameEnv(config, headless=headless) as env:
        for _ in range(episodes):
            obs = env.reset()
            episode_return = 0.0
            score = 0.0
            for _step in range(config.game.max_steps_per_episode):
                features = featurize_observation(obs)
                action = int(np.argmax(model.q_values(features)))
                result = env.step(action)
                episode_return += result.reward
                score = result.score
                obs = result.observation
                if result.done:
                    break
            scores.append(score)
            returns.append(episode_return)

    return {
        "episodes": episodes,
        "scores": scores,
        "returns": returns,
        "best_score": max(scores),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "p95_score": float(np.percentile(scores, 95)),
    }


def _linear_epsilon(config: ProjectConfig, step: int) -> float:
    schedule = config.exploration
    progress = min(1.0, step / max(1, schedule.epsilon_decay_steps))
    return schedule.epsilon_start + progress * (schedule.epsilon_end - schedule.epsilon_start)

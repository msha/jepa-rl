from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv
from jepa_rl.envs.browser_game_env import Observation
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.config import ProjectConfig, snapshot_config
from jepa_rl.utils.dashboard import write_training_dashboard


@dataclass(frozen=True)
class TrainSummary:
    run_dir: Path
    checkpoint: Path
    best_checkpoint: Path
    dashboard: Path
    steps: int
    episodes: int
    best_score: float
    mean_loss: float
    update_count: int
    weight_delta_norm: float
    replay_size: int
    target_update_count: int


@dataclass(frozen=True)
class MlSmokeSummary:
    steps: int
    initial_loss: float
    final_loss: float
    improvement: float
    weight_delta_norm: float


@dataclass(frozen=True)
class FeatureTransition:
    features: np.ndarray
    action: int
    reward: float
    next_features: np.ndarray
    done: bool


class LinearReplay:
    """In-memory replay storage for the temporary linear DQN trainer."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._storage: list[FeatureTransition] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: FeatureTransition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int, rng: random.Random) -> list[FeatureTransition]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not self._storage:
            raise ValueError("cannot sample from empty replay")
        size = min(batch_size, len(self._storage))
        return rng.sample(self._storage, size)


class LinearQModel:
    """Small linear Q model used for browser smoke training before PyTorch lands."""

    def __init__(self, feature_dim: int, num_actions: int, rng: np.random.Generator):
        self.weights = rng.normal(0.0, 0.001, size=(feature_dim, num_actions)).astype(np.float32)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights

    def copy(self) -> LinearQModel:
        model = self.__class__.__new__(self.__class__)
        model.weights = self.weights.copy()
        return model

    def sync_from(self, other: LinearQModel) -> None:
        self.weights[...] = other.weights

    def update(self, features: np.ndarray, action: int, target: float, lr: float) -> float:
        q_values = self.q_values(features)
        error = target - float(q_values[action])
        self.weights[:, action] += np.float32(lr * error) * features
        return error * error

    def update_batch(
        self,
        transitions: list[FeatureTransition],
        *,
        target_model: LinearQModel,
        gamma: float,
        lr: float,
    ) -> tuple[float, float]:
        losses: list[float] = []
        abs_td_errors: list[float] = []
        for transition in transitions:
            bootstrap = 0.0
            if not transition.done:
                bootstrap = float(np.max(target_model.q_values(transition.next_features)))
            target = transition.reward + gamma * bootstrap
            current = float(self.q_values(transition.features)[transition.action])
            abs_td_errors.append(abs(target - current))
            losses.append(self.update(transition.features, transition.action, target, lr))
        return float(np.mean(losses)), float(np.mean(abs_td_errors))

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
    batch_size: int | None = None,
    dashboard_every: int = 25,
    stop_event: Event | None = None,
) -> TrainSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_starts < 0:
        raise ValueError("learning_starts must be nonnegative")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dashboard_every < 0:
        raise ValueError("dashboard_every must be nonnegative")

    run_name = experiment or config.experiment.name
    run_dir = create_run_dir(config.experiment.output_dir, run_name)
    snapshot_config(config, run_dir / "config.yaml")
    metrics_path = run_dir / "metrics" / "train_events.jsonl"
    summary_path = run_dir / "metrics" / "train_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest.npz"
    best_checkpoint_path = run_dir / "checkpoints" / "best.npz"

    py_rng = random.Random(config.experiment.seed)
    np_rng = np.random.default_rng(config.experiment.seed)
    learning_rate = lr if lr is not None else config.agent.optimizer.lr
    effective_batch_size = batch_size or min(config.agent.batch_size, 64)
    replay = LinearReplay(capacity=config.replay.capacity)

    with PlaywrightBrowserGameEnv(config, headless=headless) as env:
        obs = env.reset()
        model = LinearQModel(
            feature_dim=featurize_observation(obs).shape[0],
            num_actions=config.actions.num_actions,
            rng=np_rng,
        )
        target_model = model.copy()
        initial_weights = model.weights.copy()

        episode_id = 0
        episode_return = 0.0
        episode_score = 0.0
        best_score = float("-inf")
        losses: list[float] = []
        td_errors: list[float] = []
        update_count = 0
        target_update_count = 0
        actual_steps = 0
        status = "completed"
        started_at = time.time()

        with metrics_path.open("w", encoding="utf-8") as metrics:
            for step in range(steps):
                if stop_event is not None and stop_event.is_set():
                    status = "stopped"
                    break
                actual_steps = step + 1
                features = featurize_observation(obs)
                epsilon = _linear_epsilon(config, step)
                if py_rng.random() < epsilon:
                    action = py_rng.randrange(config.actions.num_actions)
                else:
                    action = int(np.argmax(model.q_values(features)))

                result = env.step(action)
                next_features = featurize_observation(result.observation)
                replay.add(
                    FeatureTransition(
                        features=features,
                        action=action,
                        reward=result.reward,
                        next_features=next_features,
                        done=result.done,
                    )
                )

                loss = None
                td_error = None
                if step >= learning_starts and len(replay) > 0:
                    for _ in range(config.agent.gradient_steps):
                        batch = replay.sample(effective_batch_size, py_rng)
                        loss, td_error = model.update_batch(
                            batch,
                            target_model=target_model,
                            gamma=config.agent.gamma,
                            lr=learning_rate,
                        )
                        losses.append(loss)
                        td_errors.append(td_error)
                        update_count += 1

                if (step + 1) % config.agent.target_update_interval == 0:
                    target_model.sync_from(model)
                    target_update_count += 1

                episode_return += result.reward
                episode_score = result.score
                weight_delta_norm = float(np.linalg.norm(model.weights - initial_weights))
                event = {
                    "type": "step",
                    "step": step + 1,
                    "episode": episode_id,
                    "action": action,
                    "reward": result.reward,
                    "score": result.score,
                    "done": result.done,
                    "epsilon": epsilon,
                    "loss": loss,
                    "td_error": td_error,
                    "q_max": float(np.max(model.q_values(features))),
                    "replay_size": len(replay),
                    "updates": update_count,
                    "target_updates": target_update_count,
                    "weight_delta_norm": weight_delta_norm,
                }
                metrics.write(json.dumps(event) + "\n")

                obs = result.observation
                if result.done:
                    best_score = max(best_score, episode_score)
                    if episode_score >= best_score:
                        model.save(best_checkpoint_path, {"score": episode_score, "step": step + 1})
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

                if dashboard_every and (step + 1) % dashboard_every == 0:
                    metrics.flush()
                    _write_intermediate_summary(
                        summary_path=summary_path,
                        algorithm="linear_dqn_replay_smoke",
                        steps=step + 1,
                        requested_steps=steps,
                        status="running",
                        episodes=episode_id + 1,
                        num_actions=config.actions.num_actions,
                        update_count=update_count,
                        mean_loss=float(np.mean(losses)) if losses else 0.0,
                        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
                        replay_size=len(replay),
                        target_update_count=target_update_count,
                        weight_delta_norm=weight_delta_norm,
                        best_score=best_score if best_score != float("-inf") else episode_score,
                        started_at=started_at,
                    )
                    write_training_dashboard(run_dir)

        if episode_score or episode_return:
            best_score = max(best_score, episode_score)
        if best_score == float("-inf"):
            best_score = episode_score

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_error = float(np.mean(td_errors)) if td_errors else 0.0
        weight_delta_norm = float(np.linalg.norm(model.weights - initial_weights))
        metadata = {
            "algorithm": "linear_dqn_replay_smoke",
            "steps": actual_steps,
            "requested_steps": steps,
            "status": status,
            "episodes": episode_id + 1,
            "num_actions": config.actions.num_actions,
            "feature_grid": [12, 16],
            "batch_size": effective_batch_size,
            "replay_capacity": config.replay.capacity,
            "replay_size": len(replay),
            "target_update_interval": config.agent.target_update_interval,
            "target_update_count": target_update_count,
            "gradient_steps": config.agent.gradient_steps,
            "learning_starts": learning_starts,
            "update_count": update_count,
            "mean_loss": mean_loss,
            "mean_td_error": mean_td_error,
            "weight_delta_norm": weight_delta_norm,
            "best_score": best_score,
            "wall_time_sec": time.time() - started_at,
        }
        model.save(checkpoint_path, metadata)
        if not best_checkpoint_path.exists():
            model.save(best_checkpoint_path, metadata)
        summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        dashboard_path = write_training_dashboard(run_dir)

    return TrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        dashboard=dashboard_path,
        steps=actual_steps,
        episodes=metadata["episodes"],
        best_score=float(best_score),
        mean_loss=mean_loss,
        update_count=update_count,
        weight_delta_norm=weight_delta_norm,
        replay_size=len(replay),
        target_update_count=target_update_count,
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


def _write_intermediate_summary(
    *,
    summary_path: Path,
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
) -> None:
    metadata = {
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
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _linear_epsilon(config: ProjectConfig, step: int) -> float:
    schedule = config.exploration
    progress = min(1.0, step / max(1, schedule.epsilon_decay_steps))
    return schedule.epsilon_start + progress * (schedule.epsilon_end - schedule.epsilon_start)


def run_linear_q_ml_smoke(*, seed: int = 0, steps: int = 2000, lr: float = 0.03) -> MlSmokeSummary:
    """Train the linear Q model on a deterministic synthetic Q-function."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    rng = np.random.default_rng(seed)
    feature_dim = 9
    num_actions = 4
    sample_count = 128
    features = rng.normal(0.0, 0.25, size=(sample_count, feature_dim)).astype(np.float32)
    features[:, -1] = 1.0
    target_weights = rng.normal(0.0, 0.2, size=(feature_dim, num_actions)).astype(np.float32)
    targets = features @ target_weights

    model = LinearQModel(feature_dim=feature_dim, num_actions=num_actions, rng=rng)
    initial_weights = model.weights.copy()

    def mse() -> float:
        error = features @ model.weights - targets
        return float(np.mean(error * error))

    initial_loss = mse()
    for _ in range(steps):
        row = int(rng.integers(0, sample_count))
        action = int(rng.integers(0, num_actions))
        model.update(features[row], action, float(targets[row, action]), lr)
    final_loss = mse()
    return MlSmokeSummary(
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        improvement=initial_loss - final_loss,
        weight_delta_norm=float(np.linalg.norm(model.weights - initial_weights)),
    )

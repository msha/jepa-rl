"""Training loop for frozen JEPA encoder + DQN Q-head."""
from __future__ import annotations

import copy
import os
import random
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jepa_rl.envs.browser_game_env import BrowserGameEnv
from jepa_rl.models.device import resolve_torch_device
from jepa_rl.models.dqn import double_dqn_target
from jepa_rl.models.frozen_jepa_dqn import build_frozen_jepa_q_network
from jepa_rl.replay.replay_buffer import ReplayBuffer, Transition
from jepa_rl.training.pixel_dqn import (
    _build_optimizer,
    _env_context,
    _save_render_frame,
)
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.checkpoint import CheckpointPayload, load_torch_checkpoint, save_torch_checkpoint
from jepa_rl.utils.config import ProjectConfig, snapshot_config
from jepa_rl.utils.dashboard import write_training_dashboard
from jepa_rl.utils.metrics import (
    JsonlWriter,
    build_run_summary,
    episode_event,
    linear_epsilon,
    step_event,
    write_run_summary,
)

ALGORITHM = "frozen_jepa_dqn"


@dataclass
class FrozenJepaDqnTrainSummary:
    run_dir: Path
    checkpoint: Path
    best_checkpoint: Path
    dashboard: Path
    steps: int
    episodes: int
    best_score: float
    mean_loss: float
    mean_td_error: float
    update_count: int
    target_update_count: int
    replay_size: int
    jepa_checkpoint: Path
    weight_delta_norm: float


def train_frozen_jepa_dqn(
    config: ProjectConfig,
    *,
    jepa_checkpoint: Path,
    experiment: str | None = None,
    steps: int,
    learning_starts: int | None = None,
    headless: bool | None = None,
    batch_size: int | None = None,
    dashboard_every: int = 25,
    stop_event: Event | None = None,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None = None,
    resume_checkpoint: Path | None = None,
    screenshot_path: Path | None = None,
    live_step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> FrozenJepaDqnTrainSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if dashboard_every < 0:
        raise ValueError("dashboard_every must be nonnegative")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    run_dir = create_run_dir(config.experiment.output_dir, experiment or config.experiment.name)
    snapshot_config(config, run_dir / "config.yaml")

    if env_factory is None:
        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        def env_factory(cfg: ProjectConfig, hd: bool | None) -> BrowserGameEnv:
            return PlaywrightBrowserGameEnv(
                cfg, headless=hd, run_dir=run_dir, record_video=config.recording.enabled
            )

    device = resolve_torch_device(config.experiment.device)
    if config.experiment.precision != "fp32" and device.type == "mps":
        warnings.warn(
            "fp16/bf16 on MPS is unsupported; using fp32 instead",
            stacklevel=2,
        )

    seed = config.experiment.seed
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    metrics_path = run_dir / "metrics" / "train_events.jsonl"
    summary_path = run_dir / "metrics" / "train_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"

    online_net = build_frozen_jepa_q_network(
        config, num_actions=config.actions.num_actions, jepa_checkpoint_path=jepa_checkpoint
    ).to(device)
    target_net = copy.deepcopy(online_net)
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = _build_optimizer(config, online_net.trainable_parameters)
    replay = ReplayBuffer(capacity=config.replay.capacity)

    effective_learning_starts = (
        learning_starts if learning_starts is not None else config.agent.learning_starts
    )
    effective_batch_size = batch_size if batch_size is not None else config.agent.batch_size
    checkpoint_every = config.training.checkpoint_interval_steps

    start_step = 0
    episode_id = 0
    best_score = float("-inf")
    update_count = 0
    target_update_count = 0

    if resume_checkpoint is not None:
        payload = load_torch_checkpoint(
            resume_checkpoint,
            model=online_net,
            target_model=target_net,
            optimizer=optimizer,
        )
        py_rng.setstate(payload.rng_python)
        np_rng.bit_generator.state = payload.rng_numpy
        torch.set_rng_state(payload.rng_torch)
        start_step = payload.step
        episode_id = payload.episode
        best_score = payload.best_score
        update_count = payload.update_count
        target_update_count = payload.target_update_count
        online_net = online_net.to(device)
        target_net = target_net.to(device)

    # Track Q-head weight drift (encoder is frozen)
    q_head_params = [p.detach().clone() for p in online_net.trainable_parameters]

    def _weight_delta() -> float:
        total = 0.0
        for orig, cur in zip(q_head_params, online_net.trainable_parameters, strict=False):
            total += (cur.detach().cpu() - orig).norm().item()
        return total

    losses: list[float] = []
    td_errors: list[float] = []
    status = "completed"
    started_at = time.time()

    env_instance = env_factory(config, headless)
    with _env_context(env_instance) as env:
        obs = env.reset()

        with JsonlWriter(metrics_path) as writer:
            episode_return = 0.0
            episode_score = 0.0

            for step in range(start_step, start_step + steps):
                if stop_event is not None and stop_event.is_set():
                    status = "stopped"
                    break

                actual_step = step + 1
                eps = linear_epsilon(config, step)

                if py_rng.random() < eps:
                    action = py_rng.randrange(config.actions.num_actions)
                else:
                    obs_t = torch.from_numpy(obs.data[None]).to(device)
                    with torch.no_grad():
                        action = int(online_net(obs_t).argmax(dim=1).item())

                result = env.step(action)
                replay.add(
                    Transition(
                        obs=obs.data,
                        action=action,
                        reward=result.reward,
                        next_obs=result.observation.data,
                        done=result.done,
                        score=result.score,
                        timestamp=actual_step,
                        game_id=config.game.name,
                        episode_id=str(episode_id),
                    )
                )

                loss_val: float | None = None
                td_error_val: float | None = None
                grad_norm_val: float | None = None
                q_max_val = 0.0

                if (
                    step >= effective_learning_starts
                    and step % config.agent.train_every == 0
                    and len(replay) >= effective_batch_size
                ):
                    for _ in range(config.agent.gradient_steps):
                        batch = replay.sample(effective_batch_size, py_rng)
                        obs_b = torch.from_numpy(
                            np.stack([t.obs for t in batch])
                        ).to(device)
                        next_obs_b = torch.from_numpy(
                            np.stack([t.next_obs for t in batch])
                        ).to(device)
                        actions_b = torch.tensor(
                            [t.action for t in batch], dtype=torch.long, device=device
                        )
                        rewards_b = torch.tensor(
                            [t.reward for t in batch], dtype=torch.float32, device=device
                        )
                        dones_b = torch.tensor(
                            [t.done for t in batch], dtype=torch.bool, device=device
                        )

                        q_pred = online_net(obs_b).gather(1, actions_b[:, None]).squeeze(1)
                        target = double_dqn_target(
                            online_net=online_net,
                            target_net=target_net,
                            next_obs=next_obs_b,
                            rewards=rewards_b,
                            dones=dones_b,
                            gamma=config.agent.gamma,
                        )

                        with torch.no_grad():
                            q_max_val = float(online_net(obs_b).max(dim=1).values.mean().item())

                        loss = F.smooth_l1_loss(q_pred, target)
                        td_err = float((q_pred.detach() - target).abs().mean().item())

                        optimizer.zero_grad()
                        loss.backward()
                        grad_norm_val = float(
                            torch.nn.utils.clip_grad_norm_(
                                online_net.trainable_parameters, 10.0
                            ).item()
                        )
                        optimizer.step()

                        loss_val = float(loss.item())
                        td_error_val = td_err
                        losses.append(loss_val)
                        td_errors.append(td_error_val)
                        update_count += 1

                if (step + 1) % config.agent.target_update_interval == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    target_update_count += 1

                episode_return += result.reward
                episode_score = result.score
                wdn = _weight_delta()

                event = step_event(
                    step=actual_step,
                    episode=episode_id,
                    action=action,
                    reward=result.reward,
                    score=result.score,
                    done=result.done,
                    epsilon=eps,
                    loss=loss_val,
                    td_error=td_error_val,
                    q_max=q_max_val,
                    replay_size=len(replay),
                    updates=update_count,
                    target_updates=target_update_count,
                    weight_delta_norm=wdn,
                    grad_norm=grad_norm_val,
                )
                writer.write(event)
                if live_step_callback is not None:
                    live_step_callback(event)

                obs = result.observation
                if screenshot_path is not None:
                    _save_render_frame(env, obs.data, screenshot_path)
                if result.done:
                    if episode_score > best_score:
                        best_score = episode_score
                        _save_ckpt(
                            best_checkpoint_path,
                            online_net,
                            target_net,
                            optimizer,
                            py_rng,
                            np_rng,
                            actual_step,
                            episode_id,
                            update_count,
                            target_update_count,
                            best_score,
                            config,
                            replay_size=len(replay),
                        )

                    writer.write(
                        episode_event(
                            step=actual_step,
                            episode=episode_id,
                            return_=episode_return,
                            score=episode_score,
                        )
                    )
                    episode_id += 1
                    episode_return = 0.0
                    episode_score = 0.0
                    obs = env.reset()

                if dashboard_every and (step + 1 - start_step) % dashboard_every == 0:
                    writer.flush()
                    _update_summary(
                        summary_path,
                        config,
                        status="running",
                        steps=actual_step,
                        requested_steps=start_step + steps,
                        episodes=episode_id,
                        update_count=update_count,
                        losses=losses,
                        td_errors=td_errors,
                        replay_size=len(replay),
                        target_update_count=target_update_count,
                        weight_delta_norm=wdn,
                        best_score=best_score if best_score != float("-inf") else episode_score,
                        started_at=started_at,
                        jepa_checkpoint=jepa_checkpoint,
                    )
                    write_training_dashboard(run_dir)

                if checkpoint_every and (step + 1 - start_step) % checkpoint_every == 0:
                    _save_ckpt(
                        checkpoint_path,
                        online_net,
                        target_net,
                        optimizer,
                        py_rng,
                        np_rng,
                        actual_step,
                        episode_id,
                        update_count,
                        target_update_count,
                        best_score if best_score != float("-inf") else 0.0,
                        config,
                        replay_size=len(replay),
                    )

    final_step = start_step + steps
    final_score = best_score if best_score != float("-inf") else episode_score
    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_td_error = float(np.mean(td_errors)) if td_errors else 0.0
    wdn = _weight_delta()

    _save_ckpt(
        checkpoint_path,
        online_net,
        target_net,
        optimizer,
        py_rng,
        np_rng,
        final_step,
        episode_id,
        update_count,
        target_update_count,
        final_score,
        config,
        replay_size=len(replay),
    )
    if not best_checkpoint_path.exists():
        _save_ckpt(
            best_checkpoint_path,
            online_net,
            target_net,
            optimizer,
            py_rng,
            np_rng,
            final_step,
            episode_id,
            update_count,
            target_update_count,
            final_score,
            config,
            replay_size=len(replay),
        )

    _update_summary(
        summary_path,
        config,
        status=status,
        steps=final_step,
        requested_steps=final_step,
        episodes=episode_id,
        update_count=update_count,
        losses=losses,
        td_errors=td_errors,
        replay_size=len(replay),
        target_update_count=target_update_count,
        weight_delta_norm=wdn,
        best_score=final_score,
        started_at=started_at,
        jepa_checkpoint=jepa_checkpoint,
    )
    dashboard_path = write_training_dashboard(run_dir)

    return FrozenJepaDqnTrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        dashboard=dashboard_path,
        steps=final_step,
        episodes=episode_id,
        best_score=final_score,
        mean_loss=mean_loss,
        mean_td_error=mean_td_error,
        update_count=update_count,
        target_update_count=target_update_count,
        replay_size=len(replay),
        jepa_checkpoint=jepa_checkpoint,
        weight_delta_norm=wdn,
    )


def evaluate_frozen_jepa_dqn(
    config: ProjectConfig,
    *,
    jepa_checkpoint: Path,
    checkpoint: Path,
    episodes: int,
    headless: bool | None = None,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None = None,
    run_dir: Path | None = None,
    screenshot_path: Path | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if env_factory is None:
        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        def env_factory(cfg: ProjectConfig, hd: bool | None) -> BrowserGameEnv:
            return PlaywrightBrowserGameEnv(
                cfg, headless=hd, run_dir=run_dir, record_video=config.evaluation.record_video
            )

    device = resolve_torch_device(config.experiment.device)
    online_net = build_frozen_jepa_q_network(
        config, num_actions=config.actions.num_actions, jepa_checkpoint_path=jepa_checkpoint
    ).to(device)
    load_torch_checkpoint(checkpoint, model=online_net, target_model=None, optimizer=None)
    online_net.eval()

    record_eval = config.evaluation.record_video and run_dir is not None
    eval_video_dir = run_dir / "videos" / "eval" if record_eval and run_dir else None

    scores: list[float] = []
    returns_: list[float] = []

    env_instance = env_factory(config, headless)
    with _env_context(env_instance) as env:
        for ep_idx in range(episodes):
            obs = env.reset()
            episode_return = 0.0
            score = 0.0
            for _ in range(config.game.max_steps_per_episode):
                obs_t = torch.from_numpy(obs.data[None]).to(device)
                with torch.no_grad():
                    action = int(online_net(obs_t).argmax(dim=1).item())
                result = env.step(action)
                episode_return += result.reward
                score = result.score
                obs = result.observation
                if screenshot_path is not None:
                    _save_render_frame(env, obs.data, screenshot_path)
                if result.done:
                    break
            scores.append(score)
            returns_.append(episode_return)

            if eval_video_dir is not None and hasattr(env, "save_recording"):
                env.save_recording(eval_video_dir / f"episode_{ep_idx:06d}")

    return {
        "episodes": episodes,
        "scores": scores,
        "returns": returns_,
        "best_score": max(scores),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "p95_score": float(np.percentile(scores, 95)),
    }


def _save_ckpt(
    path: Path,
    online_net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    step: int,
    episode: int,
    update_count: int,
    target_update_count: int,
    best_score: float,
    config: ProjectConfig,
    replay_size: int = 0,
) -> None:
    save_torch_checkpoint(
        path,
        model=online_net,
        target_model=target_net,
        optimizer=optimizer,
        payload=CheckpointPayload(
            step=step,
            episode=episode,
            update_count=update_count,
            target_update_count=target_update_count,
            best_score=best_score,
            config_dict=config.to_dict(),
            rng_python=py_rng.getstate(),
            rng_numpy=np_rng.bit_generator.state,
            rng_torch=torch.get_rng_state(),
            extra={"replay_size": replay_size},
        ),
    )


def _update_summary(
    path: Path,
    config: ProjectConfig,
    *,
    status: str,
    steps: int,
    requested_steps: int,
    episodes: int,
    update_count: int,
    losses: list[float],
    td_errors: list[float],
    replay_size: int,
    target_update_count: int,
    weight_delta_norm: float,
    best_score: float,
    started_at: float,
    jepa_checkpoint: Path,
) -> None:
    summary = build_run_summary(
        algorithm=ALGORITHM,
        steps=steps,
        requested_steps=requested_steps,
        status=status,
        episodes=episodes,
        num_actions=config.actions.num_actions,
        update_count=update_count,
        mean_loss=float(np.mean(losses)) if losses else 0.0,
        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
        replay_size=replay_size,
        target_update_count=target_update_count,
        weight_delta_norm=weight_delta_norm,
        best_score=best_score,
        started_at=started_at,
        jepa_checkpoint=str(jepa_checkpoint),
        encoder_frozen=True,
        latent_path="on_the_fly",
    )
    write_run_summary(path, summary)

"""Phase 8: Joint JEPA + DQN training loop.

The JEPA encoder is shared between the world-model loss and the DQN Q-head loss.
Both losses backpropagate through the encoder, so the encoder learns to represent
observations that are both predictable across time and useful for policy learning.

Leakage invariant (see docs/leakage_audit.md):
  - Action selection: obs_t -> online_encoder -> z_t -> q_head(z_t)  [no_grad]
  - JEPA target: obs_{t+k} -> stop_grad(target_encoder) -> z_target  [EMA-delayed]
  - DQN target: next_obs stored in replay -> online_encoder -> best_action
                                           -> target_encoder -> target_q_net -> Q
  Neither path reads from future observations not yet taken.
"""
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
from jepa_rl.models.jepa import JepaWorldModel, JepaWorldModelConfig
from jepa_rl.models.joint_jepa_dqn import LatentQHead
from jepa_rl.replay.replay_buffer import ReplayBuffer, Transition
from jepa_rl.training.jepa_world import _count_valid_sequences, _ema_tau, _make_batch_from_replay
from jepa_rl.training.pixel_dqn import _build_optimizer, _env_context, _save_render_frame
from jepa_rl.utils.artifacts import create_run_dir
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

ALGORITHM = "joint_jepa_dqn"


@dataclass
class JointJepaDqnTrainSummary:
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
    jepa_update_count: int
    target_update_count: int
    replay_size: int
    jepa_loss: float
    latent_std: float
    effective_rank: float


def _save_joint_ckpt(
    path: Path,
    jepa: JepaWorldModel,
    q_head: LatentQHead,
    q_target: LatentQHead,
    jepa_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    *,
    step: int,
    episode: int,
    update_count: int,
    jepa_update_count: int,
    target_update_count: int,
    best_score: float,
    replay_size: int,
    config: ProjectConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "jepa_model": jepa.state_dict(),
            "q_net": q_head.state_dict(),
            "q_target_net": q_target.state_dict(),
            "jepa_optimizer": jepa_opt.state_dict(),
            "q_optimizer": q_opt.state_dict(),
            "step": step,
            "episode": episode,
            "update_count": update_count,
            "jepa_update_count": jepa_update_count,
            "target_update_count": target_update_count,
            "best_score": best_score,
            "replay_size": replay_size,
            "config_dict": config.to_dict(),
            "rng_python": py_rng.getstate(),
            "rng_numpy": np_rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
        },
        path,
    )


def train_joint_jepa_dqn(
    config: ProjectConfig,
    *,
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
) -> JointJepaDqnTrainSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if dashboard_every < 0:
        raise ValueError("dashboard_every must be nonnegative")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if env_factory is None:
        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        def env_factory(cfg: ProjectConfig, hd: bool | None) -> BrowserGameEnv:
            return PlaywrightBrowserGameEnv(
                cfg, headless=hd, run_dir=run_dir, record_video=config.recording.enabled
            )

    device = resolve_torch_device(config.experiment.device)
    if config.experiment.precision != "fp32" and device.type == "mps":
        warnings.warn(
            f"fp16/bf16 on MPS is unsupported; using fp32 instead "
            f"(config.experiment.precision={config.experiment.precision!r})",
            stacklevel=2,
        )

    seed = config.experiment.seed
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    run_name = experiment or config.experiment.name
    run_dir = create_run_dir(config.experiment.output_dir, run_name)
    snapshot_config(config, run_dir / "config.yaml")

    metrics_path = run_dir / "metrics" / "train_events.jsonl"
    summary_path = run_dir / "metrics" / "train_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"

    # Build models
    jepa_cfg = JepaWorldModelConfig.from_project_config(config)
    jepa = JepaWorldModel(jepa_cfg).to(device)
    q_head = LatentQHead(
        latent_dim=config.world_model.latent_dim,
        num_actions=config.actions.num_actions,
        hidden_dims=config.agent.q_network.hidden_dims,
        dueling=config.agent.q_network.dueling,
    ).to(device)
    q_target = copy.deepcopy(q_head)
    for p in q_target.parameters():
        p.requires_grad_(False)

    # Two optimizers: JEPA opt updates encoder+predictor; Q opt updates Q-head only.
    # During DQN backward, gradients flow through jepa.encode() into jepa_opt params.
    jepa_trainable = [p for p in jepa.parameters() if p.requires_grad]
    jepa_opt = _build_optimizer(config, jepa_trainable)
    q_opt = _build_optimizer(config, q_head.parameters())

    replay = ReplayBuffer(capacity=config.replay.capacity)
    effective_learning_starts = (
        learning_starts if learning_starts is not None else config.agent.learning_starts
    )
    effective_batch_size = batch_size if batch_size is not None else config.agent.batch_size
    checkpoint_every = config.training.checkpoint_interval_steps
    action_chunk_size = config.world_model.predictor.action_chunk_size
    max_horizon = max(jepa_cfg.horizons)
    sequence_length = max_horizon * action_chunk_size + 1
    world_updates_per_step = config.training.world_updates_per_env_step
    policy_updates_per_step = config.training.policy_updates_per_env_step
    freeze_encoder = config.training.freeze_encoder
    intrinsic_cfg = config.exploration.intrinsic_reward

    # Running stats for intrinsic reward normalization (EMA)
    _ir_ema_mean = 0.0
    _ir_ema_var = 1.0
    _ir_ema_alpha = 0.01  # EMA coefficient

    # Resumable state
    start_step = 0
    episode_id = 0
    best_score = float("-inf")
    update_count = 0
    jepa_update_count = 0
    target_update_count = 0

    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        jepa.load_state_dict(state["jepa_model"])
        q_head.load_state_dict(state["q_net"])
        q_target.load_state_dict(state["q_target_net"])
        jepa.to(device)
        q_head.to(device)
        q_target.to(device)
        jepa_opt.load_state_dict(state["jepa_optimizer"])
        q_opt.load_state_dict(state["q_optimizer"])
        py_rng.setstate(state["rng_python"])
        np_rng.bit_generator.state = state["rng_numpy"]
        torch.set_rng_state(state["rng_torch"])
        start_step = state["step"]
        episode_id = state["episode"]
        best_score = state["best_score"]
        update_count = state["update_count"]
        jepa_update_count = state["jepa_update_count"]
        target_update_count = state["target_update_count"]

    losses: list[float] = []
    td_errors: list[float] = []
    jepa_losses: list[float] = []
    latent_stds: list[float] = []
    effective_ranks: list[float] = []
    status = "completed"
    started_at = time.time()

    # Fractional update counters for configurable update ratios
    jepa_update_accum = 0.0
    q_update_accum = 0.0

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

                # Action selection: online encoder only, no target encoder touched
                if py_rng.random() < eps:
                    action = py_rng.randrange(config.actions.num_actions)
                else:
                    obs_t = torch.from_numpy(obs.data[None]).float().to(device)
                    with torch.no_grad():
                        z = jepa.encode(obs_t)
                        action = int(q_head(z).argmax(dim=1).item())

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
                jepa_loss_val: float | None = None
                grad_norm_val: float | None = None
                intrinsic_reward_val: float | None = None
                q_max_val = 0.0

                past_warmup = step >= effective_learning_starts
                enough_data = len(replay) >= effective_batch_size
                enough_seq = _count_valid_sequences(replay, sequence_length) >= effective_batch_size

                if past_warmup and enough_data:
                    jepa_update_accum += world_updates_per_step
                    q_update_accum += policy_updates_per_step

                    # JEPA updates
                    while jepa_update_accum >= 1.0 and enough_seq:
                        jbatch = _make_batch_from_replay(
                            replay,
                            effective_batch_size,
                            sequence_length,
                            jepa_cfg,
                            device,
                            py_rng,
                            action_chunk_size=action_chunk_size,
                        )
                        metrics = jepa(jbatch)
                        j_loss = metrics["loss"]
                        jepa_opt.zero_grad()
                        j_loss.backward()
                        torch.nn.utils.clip_grad_norm_(jepa_trainable, 1.0)
                        jepa_opt.step()
                        jepa_loss_val = float(j_loss.item())
                        jepa_losses.append(jepa_loss_val)
                        latent_stds.append(float(metrics["latent_std_mean"].item()))
                        effective_ranks.append(float(metrics["effective_rank"].item()))
                        jepa_update_count += 1
                        jepa_update_accum -= 1.0

                    # Update JEPA target encoder (EMA)
                    tau = _ema_tau(config, step, steps)
                    jepa.update_target_encoder(tau)

                    # DQN updates
                    while q_update_accum >= 1.0:
                        batch = replay.sample(effective_batch_size, py_rng)
                        obs_b = torch.from_numpy(
                            np.stack([t.obs for t in batch])
                        ).float().to(device)
                        next_obs_b = torch.from_numpy(
                            np.stack([t.next_obs for t in batch])
                        ).float().to(device)
                        actions_b = torch.tensor(
                            [t.action for t in batch], dtype=torch.long, device=device
                        )
                        rewards_b = torch.tensor(
                            [t.reward for t in batch], dtype=torch.float32, device=device
                        )
                        dones_b = torch.tensor(
                            [t.done for t in batch], dtype=torch.bool, device=device
                        )

                        # Double DQN target: action selected by online, evaluated by target
                        with torch.no_grad():
                            z_next_online = jepa.encode(next_obs_b)
                            best_actions = q_head(z_next_online).argmax(dim=1, keepdim=True)
                            z_next_target = jepa.encode_target(next_obs_b)
                            next_q_values = (
                                q_target(z_next_target).gather(1, best_actions).squeeze(1)
                            )

                            # Intrinsic reward: JEPA prediction error for observed transition
                            if intrinsic_cfg.enabled:
                                z_obs_ir = jepa.encode(obs_b)
                                # Repeat action across all horizon slots for single-step prediction
                                actions_ir = actions_b[:, None].expand(
                                    -1, max_horizon
                                ).contiguous()
                                pred_dict = jepa.predictor(z_obs_ir, actions_ir)
                                z_pred_next = pred_dict[min(jepa_cfg.horizons)]
                                z_next_actual = jepa.encode(next_obs_b)
                                ir_raw = (
                                    (z_next_actual - z_pred_next).pow(2).sum(dim=-1).sqrt()
                                )
                                # EMA normalize
                                batch_mean = float(ir_raw.mean().item())
                                batch_var = float(ir_raw.var().item()) + 1e-8
                                _ir_ema_mean = (
                                    (1 - _ir_ema_alpha) * _ir_ema_mean
                                    + _ir_ema_alpha * batch_mean
                                )
                                _ir_ema_var = (
                                    (1 - _ir_ema_alpha) * _ir_ema_var
                                    + _ir_ema_alpha * batch_var
                                )
                                if intrinsic_cfg.normalize:
                                    ir = (ir_raw - _ir_ema_mean) / (_ir_ema_var ** 0.5 + 1e-8)
                                else:
                                    ir = ir_raw
                                rewards_b = rewards_b + intrinsic_cfg.beta * ir
                                intrinsic_reward_val = float(ir.mean().item())

                            td_targets = rewards_b + config.agent.gamma * next_q_values * (~dones_b)
                            q_max_val = float(q_head(z_next_online).max(dim=1).values.mean().item())

                        # Forward with grad: DQN loss flows through encoder unless frozen
                        z_obs = jepa.encode(obs_b)
                        if freeze_encoder:
                            z_obs = z_obs.detach()
                        q_pred = q_head(z_obs).gather(1, actions_b[:, None]).squeeze(1)
                        dqn_loss = F.smooth_l1_loss(q_pred, td_targets)
                        td_err = float((q_pred.detach() - td_targets).abs().mean().item())

                        jepa_opt.zero_grad()
                        q_opt.zero_grad()
                        dqn_loss.backward()
                        clip_params = (
                            list(q_head.parameters())
                            if freeze_encoder
                            else jepa_trainable + list(q_head.parameters())
                        )
                        grad_norm_val = float(
                            torch.nn.utils.clip_grad_norm_(clip_params, 10.0).item()
                        )
                        if not freeze_encoder:
                            jepa_opt.step()
                        q_opt.step()

                        loss_val = float(dqn_loss.item())
                        td_error_val = td_err
                        losses.append(loss_val)
                        td_errors.append(td_error_val)
                        update_count += 1
                        q_update_accum -= 1.0

                    if (step + 1) % config.agent.target_update_interval == 0:
                        q_target.load_state_dict(q_head.state_dict())
                        target_update_count += 1

                episode_return += result.reward
                episode_score = result.score

                extra_kwargs: dict[str, Any] = dict(
                    jepa_loss=jepa_loss_val,
                    jepa_updates=jepa_update_count,
                )
                if intrinsic_cfg.enabled:
                    extra_kwargs["intrinsic_reward"] = intrinsic_reward_val
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
                    grad_norm=grad_norm_val,
                    **extra_kwargs,
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
                        _save_joint_ckpt(
                            best_checkpoint_path,
                            jepa, q_head, q_target, jepa_opt, q_opt,
                            py_rng, np_rng,
                            step=actual_step,
                            episode=episode_id,
                            update_count=update_count,
                            jepa_update_count=jepa_update_count,
                            target_update_count=target_update_count,
                            best_score=best_score,
                            replay_size=len(replay),
                            config=config,
                        )
                    ev = episode_event(
                        step=actual_step,
                        episode=episode_id,
                        return_=episode_return,
                        score=episode_score,
                    )
                    writer.write(ev)
                    episode_id += 1
                    episode_return = 0.0
                    episode_score = 0.0
                    obs = env.reset()

                if (step + 1) % checkpoint_every == 0:
                    _save_joint_ckpt(
                        checkpoint_path,
                        jepa, q_head, q_target, jepa_opt, q_opt,
                        py_rng, np_rng,
                        step=actual_step,
                        episode=episode_id,
                        update_count=update_count,
                        jepa_update_count=jepa_update_count,
                        target_update_count=target_update_count,
                        best_score=best_score,
                        replay_size=len(replay),
                        config=config,
                    )
                    if dashboard_every > 0:
                        write_training_dashboard(run_dir)

    # Save final checkpoint
    _save_joint_ckpt(
        checkpoint_path,
        jepa, q_head, q_target, jepa_opt, q_opt,
        py_rng, np_rng,
        step=start_step + steps,
        episode=episode_id,
        update_count=update_count,
        jepa_update_count=jepa_update_count,
        target_update_count=target_update_count,
        best_score=best_score,
        replay_size=len(replay),
        config=config,
    )

    mean_jepa_loss = float(np.mean(jepa_losses)) if jepa_losses else 0.0
    mean_latent_std = float(np.mean(latent_stds)) if latent_stds else 0.0
    mean_eff_rank = float(np.mean(effective_ranks)) if effective_ranks else 0.0

    _update_joint_summary(
        summary_path,
        config,
        status=status,
        steps=steps,
        episodes=episode_id,
        update_count=update_count,
        jepa_update_count=jepa_update_count,
        losses=losses,
        td_errors=td_errors,
        jepa_losses=jepa_losses,
        replay_size=len(replay),
        target_update_count=target_update_count,
        best_score=best_score,
        started_at=started_at,
    )
    if dashboard_every > 0:
        write_training_dashboard(run_dir)

    return JointJepaDqnTrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        dashboard=run_dir / "dashboard.html",
        steps=steps,
        episodes=episode_id,
        best_score=best_score,
        mean_loss=float(np.mean(losses)) if losses else 0.0,
        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
        update_count=update_count,
        jepa_update_count=jepa_update_count,
        target_update_count=target_update_count,
        replay_size=len(replay),
        jepa_loss=mean_jepa_loss,
        latent_std=mean_latent_std,
        effective_rank=mean_eff_rank,
    )


def _update_joint_summary(
    path: Path,
    config: ProjectConfig,
    *,
    status: str,
    steps: int,
    episodes: int,
    update_count: int,
    jepa_update_count: int,
    losses: list[float],
    td_errors: list[float],
    jepa_losses: list[float],
    replay_size: int,
    target_update_count: int,
    best_score: float,
    started_at: float,
) -> None:
    summary = build_run_summary(
        algorithm=ALGORITHM,
        steps=steps,
        requested_steps=steps,
        status=status,
        episodes=episodes,
        num_actions=config.actions.num_actions,
        update_count=update_count,
        mean_loss=float(np.mean(losses)) if losses else 0.0,
        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
        replay_size=replay_size,
        target_update_count=target_update_count,
        best_score=best_score,
        weight_delta_norm=0.0,
        started_at=started_at,
    )
    summary["jepa_update_count"] = jepa_update_count
    summary["mean_jepa_loss"] = float(np.mean(jepa_losses)) if jepa_losses else 0.0
    write_run_summary(path, summary)


def evaluate_joint_jepa_dqn(
    config: ProjectConfig,
    checkpoint_path: Path,
    *,
    episodes: int | None = None,
    headless: bool | None = None,
    run_dir: Path | None = None,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None = None,
) -> dict[str, Any]:
    """Deterministic evaluation of a joint JEPA + DQN checkpoint."""
    device = resolve_torch_device(config.experiment.device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    jepa_cfg = JepaWorldModelConfig.from_project_config(config)
    jepa = JepaWorldModel(jepa_cfg).to(device)
    jepa.load_state_dict(state["jepa_model"])
    jepa.eval()

    q_head = LatentQHead(
        latent_dim=config.world_model.latent_dim,
        num_actions=config.actions.num_actions,
        hidden_dims=config.agent.q_network.hidden_dims,
        dueling=config.agent.q_network.dueling,
    ).to(device)
    q_head.load_state_dict(state["q_net"])
    q_head.eval()

    num_episodes = episodes if episodes is not None else config.evaluation.episodes
    record_eval = config.evaluation.record_video and run_dir is not None
    eval_video_dir = run_dir / "videos" / "eval" if record_eval else None

    if env_factory is None:
        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        def env_factory(cfg: ProjectConfig, hd: bool | None) -> BrowserGameEnv:
            return PlaywrightBrowserGameEnv(
                cfg, headless=hd, record_video=config.evaluation.record_video
            )

    env_instance = env_factory(config, headless)
    scores: list[float] = []
    returns_: list[float] = []
    with _env_context(env_instance) as env:
        for ep_idx in range(num_episodes):
            obs = env.reset()
            episode_return = 0.0
            for _ in range(config.game.max_steps_per_episode):
                obs_t = torch.from_numpy(obs.data[None]).float().to(device)
                with torch.no_grad():
                    z = jepa.encode(obs_t)
                    action = int(q_head(z).argmax(dim=1).item())
                result = env.step(action)
                episode_return += result.reward
                obs = result.observation
                if result.done:
                    break
            scores.append(env.read_score())
            returns_.append(episode_return)
            if eval_video_dir is not None and hasattr(env, "save_recording"):
                env.save_recording(eval_video_dir / f"episode_{ep_idx:06d}")

    scores_arr = np.array(scores)
    return {
        "episodes": num_episodes,
        "scores": scores,
        "returns": returns_,
        "mean_score": float(scores_arr.mean()),
        "median_score": float(np.median(scores_arr)),
        "best_score": float(scores_arr.max()),
        "p95_score": float(np.percentile(scores_arr, 95)),
    }

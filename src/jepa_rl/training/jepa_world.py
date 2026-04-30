from __future__ import annotations

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

from jepa_rl.envs.browser_game_env import BrowserGameEnv
from jepa_rl.models.device import resolve_torch_device
from jepa_rl.models.jepa import JepaBatch, JepaWorldModel, JepaWorldModelConfig
from jepa_rl.replay.replay_buffer import ReplayBuffer, Transition
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.checkpoint import CheckpointPayload, load_torch_checkpoint, save_torch_checkpoint
from jepa_rl.utils.config import ProjectConfig, snapshot_config
from jepa_rl.utils.dashboard import write_training_dashboard
from jepa_rl.utils.metrics import JsonlWriter, write_run_summary


@dataclass(frozen=True)
class JepaWorldTrainSummary:
    run_dir: Path
    checkpoint: Path
    dashboard: Path
    device: str
    steps: int
    collect_steps: int
    replay_size: int
    initial_loss: float
    final_loss: float
    improvement: float
    latent_std_mean: float
    effective_rank: float


def train_jepa_world(
    config: ProjectConfig,
    *,
    experiment: str | None = None,
    steps: int,
    collect_steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    dashboard_every: int = 25,
    stop_event: Event | None = None,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv] | None = None,
    headless: bool | None = None,
    resume_checkpoint: Path | None = None,
    live_step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JepaWorldTrainSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")

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
            f"fp16/bf16 on MPS is unsupported; forcing fp32 "
            f"(config.experiment.precision={config.experiment.precision!r})",
            stacklevel=2,
        )

    seed = config.experiment.seed
    py_rng = random.Random(seed)
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    run_name = experiment or f"{config.experiment.name}_jepa"
    run_dir = create_run_dir(config.experiment.output_dir, run_name)
    snapshot_config(config, run_dir / "config.yaml")

    metrics_path = run_dir / "metrics" / "train_events.jsonl"
    summary_path = run_dir / "metrics" / "train_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"

    model_config = JepaWorldModelConfig.from_project_config(config)
    model = JepaWorldModel(model_config).to(device)
    effective_lr = lr if lr is not None else config.world_model.optimizer.lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=config.world_model.optimizer.weight_decay,
        betas=(
            tuple(config.world_model.optimizer.betas)
            if config.world_model.optimizer.betas
            else (0.9, 0.95)
        ),
    )

    start_step = 0
    if resume_checkpoint is not None:
        payload = load_torch_checkpoint(
            resume_checkpoint,
            model=model,
            target_model=None,
            optimizer=optimizer,
        )
        start_step = payload.step

    effective_batch_size = batch_size if batch_size is not None else config.agent.batch_size
    action_chunk_size = config.world_model.predictor.action_chunk_size
    sequence_length = max(model_config.horizons) * action_chunk_size + 1

    # Collect random data into replay buffer
    effective_collect_steps = (
        collect_steps if collect_steps is not None
        else max(config.replay.capacity // 10, sequence_length * effective_batch_size * 4)
    )
    replay = _collect_random_data(
        config, effective_collect_steps, py_rng, env_factory, headless,
        stop_event=stop_event,
        progress_callback=live_step_callback,
    )

    if len(replay) < sequence_length:
        raise ValueError(
            f"replay has only {len(replay)} transitions; need at least {sequence_length} "
            f"(max_horizon={max(model_config.horizons)})"
        )

    initial_loss: float | None = None
    final_metrics: dict[str, float] = {}
    started_at = time.time()

    with JsonlWriter(metrics_path) as writer:
        for grad_step in range(start_step, start_step + steps):
            if stop_event is not None and stop_event.is_set():
                break

            actual_step = grad_step + 1
            batch = _make_batch_from_replay(
                replay, effective_batch_size, sequence_length, model_config, device, py_rng,
                action_chunk_size=action_chunk_size,
            )

            model.train()
            output = model(batch)
            loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tau = _ema_tau(config, grad_step, start_step + steps)
            model.update_target_encoder(tau)

            step_metrics = {k: float(v.detach().cpu()) for k, v in output.items()}
            step_metrics["step"] = actual_step
            step_metrics["tau"] = tau

            if initial_loss is None:
                initial_loss = step_metrics["loss"]
            final_metrics = step_metrics

            writer.write({"type": "step", **step_metrics})

            if live_step_callback is not None:
                live_step_callback({"type": "step", **step_metrics})

            if dashboard_every and actual_step % dashboard_every == 0:
                writer.flush()
                _write_summary(
                    summary_path,
                    config,
                    step_metrics=step_metrics,
                    steps=actual_step,
                    requested_steps=start_step + steps,
                    replay_size=len(replay),
                    started_at=started_at,
                )
                write_training_dashboard(run_dir)

    assert initial_loss is not None
    final_loss = final_metrics.get("loss", initial_loss)

    save_torch_checkpoint(
        checkpoint_path,
        model=model,
        target_model=None,
        optimizer=optimizer,
        payload=CheckpointPayload(
            step=start_step + steps,
            episode=0,
            update_count=steps,
            target_update_count=steps,
            best_score=0.0,
            config_dict=config.to_dict(),
            rng_python=py_rng.getstate(),
            rng_numpy={},
            rng_torch=torch.get_rng_state(),
        ),
    )

    _write_summary(
        summary_path,
        config,
        step_metrics=final_metrics,
        steps=start_step + steps,
        requested_steps=start_step + steps,
        replay_size=len(replay),
        started_at=started_at,
    )
    dashboard_path = write_training_dashboard(run_dir)

    return JepaWorldTrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        dashboard=dashboard_path,
        device=str(device),
        steps=start_step + steps,
        collect_steps=len(replay),
        replay_size=len(replay),
        initial_loss=initial_loss,
        final_loss=final_loss,
        improvement=initial_loss - final_loss,
        latent_std_mean=final_metrics.get("latent_std_mean", 0.0),
        effective_rank=final_metrics.get("effective_rank", 0.0),
    )


class _NullContext:
    def __init__(self, env: BrowserGameEnv) -> None:
        self._env = env

    def __enter__(self) -> BrowserGameEnv:
        return self._env

    def __exit__(self, *exc: object) -> None:
        pass


def _env_context(env: BrowserGameEnv) -> Any:
    return env if hasattr(env, "__enter__") else _NullContext(env)


def _collect_random_data(
    config: ProjectConfig,
    collect_steps: int,
    py_rng: random.Random,
    env_factory: Callable[[ProjectConfig, bool | None], BrowserGameEnv],
    headless: bool | None,
    stop_event: Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=max(config.replay.capacity, collect_steps))
    env_instance = env_factory(config, headless)
    episode_id = 0

    with _env_context(env_instance) as env:
        obs = env.reset()
        for step in range(collect_steps):
            if stop_event is not None and stop_event.is_set():
                break
            action = py_rng.randrange(config.actions.num_actions)
            result = env.step(action)
            replay.add(
                Transition(
                    obs=obs.data,
                    action=action,
                    reward=result.reward,
                    next_obs=result.observation.data,
                    done=result.done,
                    score=result.score,
                    timestamp=step,
                    game_id=config.game.name,
                    episode_id=str(episode_id),
                )
            )
            obs = result.observation
            if result.done:
                episode_id += 1
                obs = env.reset()
            if progress_callback is not None and step % 10 == 0:
                progress_callback({
                    "type": "step",
                    "step": 0,
                    "phase": "collecting",
                    "collect_step": step + 1,
                    "collect_total": collect_steps,
                    "episodes": episode_id,
                    "replay_size": len(replay),
                    "latest_reward": result.reward,
                    "latest_score": result.score,
                    "latest_done": result.done,
                })

    return replay


def _make_batch_from_replay(
    replay: ReplayBuffer,
    batch_size: int,
    sequence_length: int,
    model_config: JepaWorldModelConfig,
    device: torch.device,
    py_rng: random.Random,
    *,
    action_chunk_size: int = 1,
) -> JepaBatch:
    max_horizon = max(model_config.horizons)
    sequences = replay.sample_sequence(
        min(batch_size, _count_valid_sequences(replay, sequence_length)),
        sequence_length,
        py_rng,
        allow_cross_episode=False,
    )
    if len(sequences) < batch_size:
        sequences = replay.sample_sequence(
            batch_size, sequence_length, py_rng, allow_cross_episode=True
        )

    context_obs = torch.from_numpy(
        np.stack([seq[0].obs for seq in sequences])
    ).to(device)
    # Each predicted step h corresponds to primitive step h * action_chunk_size.
    # The action for step h is the action taken at the start of that chunk,
    # i.e. at primitive index h * action_chunk_size.
    actions = torch.tensor(
        [[seq[h * action_chunk_size].action for h in range(max_horizon)] for seq in sequences],
        dtype=torch.long,
        device=device,
    )
    target_obs: dict[int, torch.Tensor] = {}
    for horizon in model_config.horizons:
        target_obs[horizon] = torch.from_numpy(
            np.stack([seq[horizon * action_chunk_size].obs for seq in sequences])
        ).to(device)

    return JepaBatch(context_obs=context_obs, target_obs=target_obs, actions=actions)


def _count_valid_sequences(replay: ReplayBuffer, sequence_length: int) -> int:
    try:
        return len(replay._valid_sequence_starts(sequence_length, allow_cross_episode=False))
    except Exception:
        return 0


def _ema_tau(config: ProjectConfig, step: int, total_steps: int) -> float:
    start = config.world_model.target_encoder.ema_tau_start
    end = config.world_model.target_encoder.ema_tau_end
    progress = min(1.0, step / max(1, total_steps - 1))
    return start + progress * (end - start)


def _write_summary(
    path: Path,
    config: ProjectConfig,
    *,
    step_metrics: dict[str, Any],
    steps: int,
    requested_steps: int,
    replay_size: int,
    started_at: float,
) -> None:
    summary = {
        "algorithm": "jepa_world",
        "steps": steps,
        "requested_steps": requested_steps,
        "status": "running" if steps < requested_steps else "completed",
        "replay_size": replay_size,
        "wall_time_sec": time.time() - started_at,
        "loss": step_metrics.get("loss", 0.0),
        "prediction_loss": step_metrics.get("prediction_loss", 0.0),
        "variance_loss": step_metrics.get("variance_loss", 0.0),
        "covariance_loss": step_metrics.get("covariance_loss", 0.0),
        "latent_std_mean": step_metrics.get("latent_std_mean", 0.0),
        "effective_rank": step_metrics.get("effective_rank", 0.0),
        # Dashboard back-compat keys
        "episodes": 0,
        "best_score": 0.0,
        "mean_loss": step_metrics.get("loss", 0.0),
        "mean_td_error": 0.0,
        "update_count": steps,
        "target_update_count": steps,
        "weight_delta_norm": 0.0,
    }
    # Per-horizon losses
    for k, v in step_metrics.items():
        if k.startswith("prediction_loss_h"):
            summary[k] = v
    write_run_summary(path, summary)

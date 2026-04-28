from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from jepa_rl.models.device import resolve_torch_device
from jepa_rl.models.jepa import JepaBatch, JepaWorldModel, JepaWorldModelConfig
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.config import ProjectConfig, snapshot_config


@dataclass(frozen=True)
class JepaSmokeSummary:
    run_dir: Path
    checkpoint: Path
    device: str
    steps: int
    initial_loss: float
    final_loss: float
    improvement: float
    latent_std_mean: float
    effective_rank: float


def run_jepa_smoke(
    config: ProjectConfig,
    *,
    experiment: str = "jepa_smoke",
    steps: int = 20,
    batch_size: int = 8,
    device_name: str = "auto",
    lr: float | None = None,
) -> JepaSmokeSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 1:
        raise ValueError("batch_size must be > 1")

    device = resolve_torch_device(device_name)
    torch.manual_seed(config.experiment.seed)
    run_dir = create_run_dir(config.experiment.output_dir, experiment)
    snapshot_config(config, run_dir / "config.yaml")
    metrics_path = run_dir / "metrics" / "jepa_smoke.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "jepa_smoke.pt"

    model_config = JepaWorldModelConfig.from_project_config(config)
    model = JepaWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr if lr is not None else config.world_model.optimizer.lr,
        weight_decay=config.world_model.optimizer.weight_decay,
    )

    initial_loss = None
    final_metrics = None
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step in range(steps):
            batch = _make_synthetic_batch(config, model_config, batch_size, device)
            model.train()
            output = model(batch)
            loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder(_ema_tau(config, step, steps))
            if initial_loss is None:
                initial_loss = float(loss.detach().cpu())
            final_metrics = {key: float(value.detach().cpu()) for key, value in output.items()}
            final_metrics["step"] = step + 1
            metrics.write(json.dumps(final_metrics) + "\n")

    assert initial_loss is not None and final_metrics is not None
    torch.save(
        {
            "model": model.state_dict(),
            "config": model_config,
            "final_metrics": final_metrics,
        },
        checkpoint_path,
    )
    final_loss = final_metrics["loss"]
    return JepaSmokeSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        device=str(device),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        improvement=initial_loss - final_loss,
        latent_std_mean=final_metrics["latent_std_mean"],
        effective_rank=final_metrics["effective_rank"],
    )


def _make_synthetic_batch(
    config: ProjectConfig,
    model_config: JepaWorldModelConfig,
    batch_size: int,
    device: torch.device,
) -> JepaBatch:
    height = config.observation.height
    width = config.observation.width
    channels = config.observation.input_channels
    max_horizon = max(model_config.horizons)
    context = torch.rand(batch_size, channels, height, width, device=device)
    actions = torch.randint(
        low=0,
        high=model_config.num_actions,
        size=(batch_size, max_horizon),
        device=device,
    )
    targets: dict[int, torch.Tensor] = {}
    for horizon in model_config.horizons:
        shift = horizon % max(1, width)
        action_signal = actions[:, horizon - 1].float().view(batch_size, 1, 1, 1)
        action_signal = action_signal / max(1, model_config.num_actions - 1)
        target = torch.roll(context, shifts=shift, dims=-1)
        target = (0.85 * target + 0.15 * action_signal).clamp(0, 1)
        targets[horizon] = target
    return JepaBatch(context_obs=context, target_obs=targets, actions=actions)


def _ema_tau(config: ProjectConfig, step: int, steps: int) -> float:
    start = config.world_model.target_encoder.ema_tau_start
    end = config.world_model.target_encoder.ema_tau_end
    progress = min(1.0, step / max(1, steps - 1))
    return start + progress * (end - start)


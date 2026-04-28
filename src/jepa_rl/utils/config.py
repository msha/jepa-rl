from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa_rl.browser.action_spaces import ActionSpaceError, DiscreteKeyboardActionSpace
from jepa_rl.utils.simple_yaml import dump_simple_yaml, load_simple_yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def _as_dict(data: Any, *, source: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must contain a mapping at the top level")
    return data


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _as_dict(load_simple_yaml(path), source=path)

    with path.open("r", encoding="utf-8") as handle:
        return _as_dict(yaml.safe_load(handle) or {}, source=path)


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load a config and recursively merge its ``extends`` list."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")

    raw = _load_yaml_file(config_path)
    extends = raw.get("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if extends is None:
        extends = []
    if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
        raise ConfigError(f"{config_path}: extends must be a string or list of strings")

    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = (config_path.parent / parent).resolve()
        merged = _merge_dicts(merged, load_config_dict(parent_path))
    return _merge_dicts(merged, raw)


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"missing required mapping: {key}")
    return value


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be a nonnegative integer")
    return value


def _positive_float(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ConfigError(f"{path} must be a positive number")
    return float(value)


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{path} cannot be empty")
    return value


def _str_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{path} must be a list of strings")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    device: str
    precision: str
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        precision = _str(data.get("precision", "fp32"), "experiment.precision")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ConfigError("experiment.precision must be one of fp32, fp16, bf16")
        return cls(
            name=_str(data.get("name"), "experiment.name"),
            seed=_nonnegative_int(data.get("seed", 0), "experiment.seed"),
            device=_str(data.get("device", "cpu"), "experiment.device"),
            precision=precision,
            output_dir=_str(data.get("output_dir", "runs"), "experiment.output_dir"),
        )


@dataclass(frozen=True)
class GameConfig:
    name: str
    url: str
    browser: str
    headless: bool
    fps: int
    action_repeat: int
    max_steps_per_episode: int
    reset_timeout_sec: float
    done_selector: str | None
    reset_key: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameConfig:
        browser = _str(data.get("browser", "chromium"), "game.browser")
        if browser not in {"chromium", "chrome"}:
            raise ConfigError("game.browser must be chromium or chrome")
        done_selector = data.get("done_selector")
        if done_selector is not None:
            done_selector = _str(done_selector, "game.done_selector")

        reset_key = data.get("reset_key")
        if reset_key is not None:
            reset_key = _str(reset_key, "game.reset_key")

        return cls(
            name=_str(data.get("name"), "game.name"),
            url=_str(data.get("url"), "game.url"),
            browser=browser,
            headless=_bool(data.get("headless", True), "game.headless"),
            fps=_positive_int(data.get("fps", 30), "game.fps"),
            action_repeat=_positive_int(data.get("action_repeat", 1), "game.action_repeat"),
            max_steps_per_episode=_positive_int(
                data.get("max_steps_per_episode", 10000), "game.max_steps_per_episode"
            ),
            reset_timeout_sec=_positive_float(
                data.get("reset_timeout_sec", 10), "game.reset_timeout_sec"
            ),
            done_selector=done_selector,
            reset_key=reset_key,
        )


@dataclass(frozen=True)
class ObservationConfig:
    mode: str
    width: int
    height: int
    grayscale: bool
    frame_stack: int
    crop: tuple[int, int, int, int] | None
    normalize: bool

    @property
    def input_channels(self) -> int:
        color_channels = 1 if self.grayscale else 3
        return color_channels * self.frame_stack

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationConfig:
        mode = _str(data.get("mode", "screenshot"), "observation.mode")
        if mode not in {"screenshot", "canvas", "dom_assisted", "hybrid"}:
            raise ConfigError(
                "observation.mode must be screenshot, canvas, dom_assisted, or hybrid"
            )

        crop_raw = data.get("crop")
        crop = None
        if crop_raw is not None:
            if (
                not isinstance(crop_raw, list)
                or len(crop_raw) != 4
                or not all(isinstance(item, int) and item >= 0 for item in crop_raw)
            ):
                raise ConfigError("observation.crop must be null or [x, y, width, height]")
            crop = tuple(crop_raw)  # type: ignore[assignment]
            if crop[2] <= 0 or crop[3] <= 0:
                raise ConfigError("observation.crop width and height must be positive")

        return cls(
            mode=mode,
            width=_positive_int(data.get("width", 84), "observation.width"),
            height=_positive_int(data.get("height", 84), "observation.height"),
            grayscale=_bool(data.get("grayscale", False), "observation.grayscale"),
            frame_stack=_positive_int(data.get("frame_stack", 4), "observation.frame_stack"),
            crop=crop,
            normalize=_bool(data.get("normalize", True), "observation.normalize"),
        )


@dataclass(frozen=True)
class ActionsConfig:
    type: str
    keys: tuple[str, ...] = ()
    repeat: int | None = None

    @property
    def num_actions(self) -> int:
        return len(self.keys)

    def build_discrete_keyboard_space(self) -> DiscreteKeyboardActionSpace:
        if self.type != "discrete_keyboard":
            raise ConfigError("only discrete_keyboard action spaces can be built in V1")
        try:
            return DiscreteKeyboardActionSpace(self.keys)
        except ActionSpaceError as exc:
            raise ConfigError(str(exc)) from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionsConfig:
        action_type = _str(data.get("type", "discrete_keyboard"), "actions.type")
        if action_type not in {
            "discrete_keyboard",
            "discrete_mouse",
            "hybrid_discrete",
            "continuous_mouse",
        }:
            raise ConfigError("actions.type is not supported")
        if action_type != "discrete_keyboard":
            raise ConfigError("V1 scaffold currently supports only discrete_keyboard actions")

        keys = tuple(_str_list(data.get("keys"), "actions.keys"))
        config = cls(
            type=action_type,
            keys=keys,
            repeat=(
                _positive_int(data["repeat"], "actions.repeat")
                if "repeat" in data and data["repeat"] is not None
                else None
            ),
        )
        config.build_discrete_keyboard_space()
        return config


@dataclass(frozen=True)
class RewardConfig:
    type: str
    score_reader: str
    score_selector: str | None
    score_region: tuple[int, int, int, int] | None
    survival_bonus: float
    idle_penalty: float
    death_penalty: float
    clip_rewards: bool
    privileged: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardConfig:
        reward_type = _str(data.get("type", "score_delta"), "reward.type")
        if reward_type != "score_delta":
            raise ConfigError("reward.type must be score_delta")

        score_reader = _str(data.get("score_reader", "dom"), "reward.score_reader")
        if score_reader not in {"dom", "ocr", "canvas_ocr", "visual_template", "javascript"}:
            raise ConfigError("reward.score_reader is not supported")

        score_selector = data.get("score_selector")
        if score_selector is not None:
            score_selector = _str(score_selector, "reward.score_selector")

        if score_reader == "dom" and score_selector is None:
            raise ConfigError("reward.score_selector is required when score_reader is dom")

        privileged = _bool(data.get("privileged", False), "reward.privileged")
        if score_reader == "javascript" and not privileged:
            raise ConfigError("javascript score readers must set reward.privileged: true")

        score_region_raw = data.get("score_region")
        score_region = None
        if score_region_raw is not None:
            if (
                not isinstance(score_region_raw, list)
                or len(score_region_raw) != 4
                or not all(isinstance(item, int) and item >= 0 for item in score_region_raw)
            ):
                raise ConfigError("reward.score_region must be null or [x, y, width, height]")
            score_region = tuple(score_region_raw)  # type: ignore[assignment]

        return cls(
            type=reward_type,
            score_reader=score_reader,
            score_selector=score_selector,
            score_region=score_region,
            survival_bonus=_number(data.get("survival_bonus", 0.0), "reward.survival_bonus"),
            idle_penalty=_number(data.get("idle_penalty", 0.0), "reward.idle_penalty"),
            death_penalty=_number(data.get("death_penalty", 0.0), "reward.death_penalty"),
            clip_rewards=_bool(data.get("clip_rewards", False), "reward.clip_rewards"),
            privileged=privileged,
        )


@dataclass(frozen=True)
class EncoderConfig:
    type: str
    hidden_channels: tuple[int, ...] = ()
    image_size: tuple[int, int] | None = None
    patch_size: int | None = None
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    mlp_ratio: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderConfig:
        encoder_type = _str(data.get("type", "conv"), "world_model.encoder.type")
        if encoder_type not in {"conv", "vit"}:
            raise ConfigError("world_model.encoder.type must be conv or vit")
        hidden_channels = tuple(
            _positive_int(item, "world_model.encoder.hidden_channels[]")
            for item in data.get("hidden_channels", [])
        )
        if encoder_type == "conv" and not hidden_channels:
            raise ConfigError("conv encoder requires hidden_channels")
        return cls(type=encoder_type, hidden_channels=hidden_channels)


@dataclass(frozen=True)
class PredictorConditioningConfig:
    action_sequence: bool
    policy_embedding: bool
    task_or_goal_embedding: bool
    value_guidance: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PredictorConditioningConfig:
        data = data or {}
        config = cls(
            action_sequence=_bool(
                data.get("action_sequence", True), "conditioning.action_sequence"
            ),
            policy_embedding=_bool(
                data.get("policy_embedding", False), "conditioning.policy_embedding"
            ),
            task_or_goal_embedding=_bool(
                data.get("task_or_goal_embedding", False), "conditioning.task_or_goal_embedding"
            ),
            value_guidance=_bool(data.get("value_guidance", False), "conditioning.value_guidance"),
        )
        if not config.action_sequence:
            raise ConfigError("V1 predictor requires action_sequence conditioning")
        if config.policy_embedding or config.task_or_goal_embedding or config.value_guidance:
            raise ConfigError("V2/V3 JEPA conditioning is not implemented in the V1 scaffold")
        return config


@dataclass(frozen=True)
class PredictorConfig:
    type: str
    hidden_dim: int
    depth: int
    num_heads: int
    horizons: tuple[int, ...]
    action_embed_dim: int
    action_chunk_size: int
    conditioning: PredictorConditioningConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictorConfig:
        predictor_type = _str(data.get("type", "transformer"), "world_model.predictor.type")
        if predictor_type not in {"transformer", "gru"}:
            raise ConfigError("world_model.predictor.type must be transformer or gru")
        horizons_raw = data.get("horizons", [1, 2, 4, 8])
        if (
            not isinstance(horizons_raw, list)
            or not horizons_raw
            or not all(isinstance(item, int) and item > 0 for item in horizons_raw)
        ):
            raise ConfigError("world_model.predictor.horizons must be a non-empty list of integers")
        return cls(
            type=predictor_type,
            hidden_dim=_positive_int(
                data.get("hidden_dim", 512), "world_model.predictor.hidden_dim"
            ),
            depth=_positive_int(data.get("depth", 4), "world_model.predictor.depth"),
            num_heads=_positive_int(data.get("num_heads", 8), "world_model.predictor.num_heads"),
            horizons=tuple(horizons_raw),
            action_embed_dim=_positive_int(
                data.get("action_embed_dim", 64), "world_model.predictor.action_embed_dim"
            ),
            action_chunk_size=_positive_int(
                data.get("action_chunk_size", 1), "world_model.predictor.action_chunk_size"
            ),
            conditioning=PredictorConditioningConfig.from_dict(data.get("conditioning")),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float
    betas: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> OptimizerConfig:
        optimizer_type = _str(data.get("type", "adamw"), f"{path}.type")
        if optimizer_type not in {"adamw", "adam"}:
            raise ConfigError(f"{path}.type must be adamw or adam")
        betas_raw = data.get("betas")
        betas = None
        if betas_raw is not None:
            if (
                not isinstance(betas_raw, list)
                or len(betas_raw) != 2
                or not all(isinstance(item, (int, float)) for item in betas_raw)
            ):
                raise ConfigError(f"{path}.betas must be [beta1, beta2]")
            betas = (float(betas_raw[0]), float(betas_raw[1]))
        return cls(
            type=optimizer_type,
            lr=_positive_float(data.get("lr", 0.0003), f"{path}.lr"),
            weight_decay=_number(data.get("weight_decay", 0.0), f"{path}.weight_decay"),
            betas=betas,
        )


@dataclass(frozen=True)
class WorldLossConfig:
    prediction: str
    lambda_var: float
    lambda_cov: float
    latent_norm: bool
    variance_floor: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldLossConfig:
        prediction = _str(data.get("prediction", "cosine_mse"), "world_model.loss.prediction")
        if prediction not in {"cosine_mse", "mse"}:
            raise ConfigError("world_model.loss.prediction must be cosine_mse or mse")
        return cls(
            prediction=prediction,
            lambda_var=_number(data.get("lambda_var", 1.0), "world_model.loss.lambda_var"),
            lambda_cov=_number(data.get("lambda_cov", 0.04), "world_model.loss.lambda_cov"),
            latent_norm=_bool(data.get("latent_norm", True), "world_model.loss.latent_norm"),
            variance_floor=_positive_float(
                data.get("variance_floor", 1.0), "world_model.loss.variance_floor"
            ),
        )


@dataclass(frozen=True)
class TargetEncoderConfig:
    ema_tau_start: float
    ema_tau_end: float
    stop_gradient: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetEncoderConfig:
        start = _positive_float(data.get("ema_tau_start", 0.996), "target_encoder.ema_tau_start")
        end = _positive_float(data.get("ema_tau_end", 0.9999), "target_encoder.ema_tau_end")
        if start >= 1.0 or end >= 1.0:
            raise ConfigError("target encoder EMA tau values must be < 1.0")
        if end < start:
            raise ConfigError("target encoder ema_tau_end must be >= ema_tau_start")
        return cls(
            ema_tau_start=start,
            ema_tau_end=end,
            stop_gradient=_bool(data.get("stop_gradient", True), "target_encoder.stop_gradient"),
        )


@dataclass(frozen=True)
class WorldModelConfig:
    enabled: bool
    latent_dim: int
    encoder: EncoderConfig
    predictor: PredictorConfig
    optimizer: OptimizerConfig
    loss: WorldLossConfig
    target_encoder: TargetEncoderConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldModelConfig:
        return cls(
            enabled=_bool(data.get("enabled", True), "world_model.enabled"),
            latent_dim=_positive_int(data.get("latent_dim", 512), "world_model.latent_dim"),
            encoder=EncoderConfig.from_dict(_require_mapping(data, "encoder")),
            predictor=PredictorConfig.from_dict(_require_mapping(data, "predictor")),
            optimizer=OptimizerConfig.from_dict(
                _require_mapping(data, "optimizer"), "world_model.optimizer"
            ),
            loss=WorldLossConfig.from_dict(_require_mapping(data, "loss")),
            target_encoder=TargetEncoderConfig.from_dict(_require_mapping(data, "target_encoder")),
        )


@dataclass(frozen=True)
class QNetworkConfig:
    hidden_dims: tuple[int, ...]
    dueling: bool
    distributional: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QNetworkConfig:
        hidden_dims = data.get("hidden_dims", [512, 512])
        if (
            not isinstance(hidden_dims, list)
            or not hidden_dims
            or not all(isinstance(item, int) and item > 0 for item in hidden_dims)
        ):
            raise ConfigError("agent.q_network.hidden_dims must be a non-empty list of integers")
        return cls(
            hidden_dims=tuple(hidden_dims),
            dueling=_bool(data.get("dueling", True), "agent.q_network.dueling"),
            distributional=_bool(
                data.get("distributional", False), "agent.q_network.distributional"
            ),
        )


@dataclass(frozen=True)
class AgentConfig:
    algorithm: str
    gamma: float
    n_step: int
    batch_size: int
    target_update_interval: int
    learning_starts: int
    train_every: int
    gradient_steps: int
    q_network: QNetworkConfig
    optimizer: OptimizerConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        algorithm = _str(data.get("algorithm", "dqn"), "agent.algorithm")
        if algorithm not in {"dqn", "ppo"}:
            raise ConfigError("agent.algorithm must be dqn or ppo")
        if algorithm != "dqn":
            raise ConfigError("V1 scaffold currently supports only dqn")
        gamma = _positive_float(data.get("gamma", 0.997), "agent.gamma")
        if gamma >= 1.0:
            raise ConfigError("agent.gamma must be < 1.0")
        return cls(
            algorithm=algorithm,
            gamma=gamma,
            n_step=_positive_int(data.get("n_step", 5), "agent.n_step"),
            batch_size=_positive_int(data.get("batch_size", 256), "agent.batch_size"),
            target_update_interval=_positive_int(
                data.get("target_update_interval", 2000), "agent.target_update_interval"
            ),
            learning_starts=_nonnegative_int(
                data.get("learning_starts", 10000), "agent.learning_starts"
            ),
            train_every=_positive_int(data.get("train_every", 4), "agent.train_every"),
            gradient_steps=_positive_int(data.get("gradient_steps", 1), "agent.gradient_steps"),
            q_network=QNetworkConfig.from_dict(_require_mapping(data, "q_network")),
            optimizer=OptimizerConfig.from_dict(
                _require_mapping(data, "optimizer"), "agent.optimizer"
            ),
        )


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int
    prioritized: bool
    priority_alpha: float
    priority_beta_start: float
    priority_beta_end: float
    sequence_length: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayConfig:
        beta_start = _number(data.get("priority_beta_start", 0.4), "replay.priority_beta_start")
        beta_end = _number(data.get("priority_beta_end", 1.0), "replay.priority_beta_end")
        if not 0.0 <= beta_start <= 1.0 or not 0.0 <= beta_end <= 1.0:
            raise ConfigError("replay priority beta values must be in [0, 1]")
        if beta_end < beta_start:
            raise ConfigError("replay.priority_beta_end must be >= priority_beta_start")
        return cls(
            capacity=_positive_int(data.get("capacity", 1_000_000), "replay.capacity"),
            prioritized=_bool(data.get("prioritized", True), "replay.prioritized"),
            priority_alpha=_number(data.get("priority_alpha", 0.6), "replay.priority_alpha"),
            priority_beta_start=beta_start,
            priority_beta_end=beta_end,
            sequence_length=_positive_int(
                data.get("sequence_length", 16), "replay.sequence_length"
            ),
        )


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    enabled: bool
    source: str
    beta: float
    normalize: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> IntrinsicRewardConfig:
        data = data or {}
        source = _str(
            data.get("source", "jepa_prediction_error"), "exploration.intrinsic_reward.source"
        )
        if source != "jepa_prediction_error":
            raise ConfigError("only jepa_prediction_error intrinsic reward is supported")
        return cls(
            enabled=_bool(data.get("enabled", False), "exploration.intrinsic_reward.enabled"),
            source=source,
            beta=_number(data.get("beta", 0.0), "exploration.intrinsic_reward.beta"),
            normalize=_bool(data.get("normalize", True), "exploration.intrinsic_reward.normalize"),
        )


@dataclass(frozen=True)
class ExplorationConfig:
    type: str
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    intrinsic_reward: IntrinsicRewardConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplorationConfig:
        exploration_type = _str(data.get("type", "epsilon_greedy"), "exploration.type")
        if exploration_type != "epsilon_greedy":
            raise ConfigError("exploration.type must be epsilon_greedy")
        epsilon_start = _number(data.get("epsilon_start", 1.0), "exploration.epsilon_start")
        epsilon_end = _number(data.get("epsilon_end", 0.05), "exploration.epsilon_end")
        if not 0.0 <= epsilon_start <= 1.0 or not 0.0 <= epsilon_end <= 1.0:
            raise ConfigError("epsilon values must be in [0, 1]")
        if epsilon_end > epsilon_start:
            raise ConfigError("exploration.epsilon_end must be <= epsilon_start")
        return cls(
            type=exploration_type,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=_positive_int(
                data.get("epsilon_decay_steps", 500_000), "exploration.epsilon_decay_steps"
            ),
            intrinsic_reward=IntrinsicRewardConfig.from_dict(data.get("intrinsic_reward")),
        )


@dataclass(frozen=True)
class TrainingConfig:
    passive_pretrain_steps: int
    total_env_steps: int
    learning_starts: int
    train_every: int
    world_updates_per_env_step: float
    policy_updates_per_env_step: float
    eval_interval_steps: int
    checkpoint_interval_steps: int
    planning_start_step: int
    planning_eval_only_until: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        return cls(
            passive_pretrain_steps=_nonnegative_int(
                data.get("passive_pretrain_steps", 0), "training.passive_pretrain_steps"
            ),
            total_env_steps=_positive_int(
                data.get("total_env_steps", 1), "training.total_env_steps"
            ),
            learning_starts=_nonnegative_int(
                data.get("learning_starts", 0), "training.learning_starts"
            ),
            train_every=_positive_int(data.get("train_every", 1), "training.train_every"),
            world_updates_per_env_step=_number(
                data.get("world_updates_per_env_step", 1), "training.world_updates_per_env_step"
            ),
            policy_updates_per_env_step=_number(
                data.get("policy_updates_per_env_step", 1), "training.policy_updates_per_env_step"
            ),
            eval_interval_steps=_positive_int(
                data.get("eval_interval_steps", 50_000), "training.eval_interval_steps"
            ),
            checkpoint_interval_steps=_positive_int(
                data.get("checkpoint_interval_steps", 50_000), "training.checkpoint_interval_steps"
            ),
            planning_start_step=_nonnegative_int(
                data.get("planning_start_step", 0), "training.planning_start_step"
            ),
            planning_eval_only_until=_nonnegative_int(
                data.get("planning_eval_only_until", 0), "training.planning_eval_only_until"
            ),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int
    deterministic: bool
    record_video: bool
    save_best_by: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        save_best_by = _str(data.get("save_best_by", "mean_score"), "evaluation.save_best_by")
        if save_best_by not in {"mean_score", "median_score", "best_score"}:
            raise ConfigError(
                "evaluation.save_best_by must be mean_score, median_score, or best_score"
            )
        return cls(
            episodes=_positive_int(data.get("episodes", 20), "evaluation.episodes"),
            deterministic=_bool(data.get("deterministic", True), "evaluation.deterministic"),
            record_video=_bool(data.get("record_video", True), "evaluation.record_video"),
            save_best_by=save_best_by,
        )


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    game: GameConfig
    observation: ObservationConfig
    actions: ActionsConfig
    reward: RewardConfig
    world_model: WorldModelConfig
    agent: AgentConfig
    replay: ReplayConfig
    exploration: ExplorationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        config = cls(
            experiment=ExperimentConfig.from_dict(_require_mapping(data, "experiment")),
            game=GameConfig.from_dict(_require_mapping(data, "game")),
            observation=ObservationConfig.from_dict(_require_mapping(data, "observation")),
            actions=ActionsConfig.from_dict(_require_mapping(data, "actions")),
            reward=RewardConfig.from_dict(_require_mapping(data, "reward")),
            world_model=WorldModelConfig.from_dict(_require_mapping(data, "world_model")),
            agent=AgentConfig.from_dict(_require_mapping(data, "agent")),
            replay=ReplayConfig.from_dict(_require_mapping(data, "replay")),
            exploration=ExplorationConfig.from_dict(_require_mapping(data, "exploration")),
            training=TrainingConfig.from_dict(_require_mapping(data, "training")),
            evaluation=EvaluationConfig.from_dict(_require_mapping(data, "evaluation")),
        )
        config.validate_cross_fields()
        return config

    def validate_cross_fields(self) -> None:
        if self.actions.repeat is not None and self.actions.repeat != self.game.action_repeat:
            raise ConfigError("actions.repeat must match game.action_repeat when set")
        if self.replay.sequence_length <= max(self.world_model.predictor.horizons):
            raise ConfigError(
                "replay.sequence_length must be greater than the largest JEPA horizon"
            )
        if self.agent.learning_starts > self.training.total_env_steps:
            raise ConfigError("agent.learning_starts cannot exceed training.total_env_steps")
        if self.training.learning_starts > self.training.total_env_steps:
            raise ConfigError("training.learning_starts cannot exceed training.total_env_steps")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))


def load_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.from_dict(load_config_dict(path))


def snapshot_config(config: ProjectConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        output_path.write_text(dump_simple_yaml(data), encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json.loads(json.dumps(data)), handle, sort_keys=False)

"""Tests for config field name consistency between _build_config_detail and dataclasses."""

from __future__ import annotations

import dataclasses

from jepa_rl.ui.server import _apply_config_override, _build_config_detail
from jepa_rl.utils.config import load_config

# Fields where the key is a logical label, not a dataclass field name.
# These are rendered as read-only via meta.type == "readonly" in the UI.
READ_ONLY_KEYS = {
    "input_channels",   # ObservationConfig property
    "actions",          # display alias for num_actions
    "keys",             # ActionsConfig tuple
    "encoder",          # nested dataclass .type
    "predictor",        # nested dataclass .type
    "horizons",         # predictor sub-field
    "loss",             # nested dataclass value
    "ema_tau",          # composite display string
    "reward",           # reward.type display
    "score_reader",     # reward.score_reader display
    "clip_rewards",     # reward.clip_rewards shown in game info section
}

CONFIG_PATH = "configs/games/breakout.yaml"


def _load_test_config():
    return load_config(CONFIG_PATH)


def _unpack(field):
    """Unpack a 4-element field tuple: (key, value, tip, meta)."""
    key = field[0]
    value = field[1]
    tip = field[2] if len(field) > 2 else ""
    meta = field[3] if len(field) > 3 else {}
    return key, value, tip, meta


class TestConfigFieldConsistency:
    """Every editable field key in _build_config_detail must resolve to a real dataclass field."""

    def test_all_field_keys_are_real_attributes(self):
        config = _load_test_config()
        groups = _build_config_detail(config)

        group_map = {
            "experiment": config.experiment,
            "game": config.game,
            "observation": config.observation,
            "actions": config.actions,
            "reward": config.reward,
            "agent": config.agent,
            "exploration": config.exploration,
            "replay": config.replay,
            "world_model": config.world_model,
            "training": config.training,
        }

        errors = []
        for group in groups:
            title = group["title"]
            for field in group["fields"]:
                key, value, tip, meta = _unpack(field)
                if key in READ_ONLY_KEYS or meta.get("type") == "readonly":
                    continue

                sub = group_map.get(title)
                if sub is None:
                    errors.append(f"group '{title}' not found in group_map")
                    continue
                if not hasattr(sub, key):
                    errors.append(f"{title}.{key} not found on {type(sub).__name__}")
                else:
                    field_names = {f.name for f in dataclasses.fields(sub)}
                    if key not in field_names:
                        errors.append(
                            f"{title}.{key} exists as attr but is not a dataclass field "
                            f"on {type(sub).__name__} (missing from read-only set?)"
                        )

        assert not errors, "Field mismatches:\n" + "\n".join(errors)

    def test_all_group_titles_match_group_map(self):
        config = _load_test_config()
        groups = _build_config_detail(config)

        valid_groups = {
            "experiment", "game", "observation", "actions", "reward",
            "agent", "exploration", "replay", "world_model", "training",
        }
        for group in groups:
            assert group["title"] in valid_groups, (
                f"group title '{group['title']}' not in group_map keys"
            )

    def test_editable_fields_have_type_meta(self):
        """Every editable field should have a meta dict with a 'type' key."""
        config = _load_test_config()
        groups = _build_config_detail(config)
        for group in groups:
            for field in group["fields"]:
                key, value, tip, meta = _unpack(field)
                assert "type" in meta, f"{group['title']}.{key} missing 'type' in meta"

    def test_readonly_fields_are_marked(self):
        """Fields in READ_ONLY_KEYS should have type 'readonly' in meta."""
        config = _load_test_config()
        groups = _build_config_detail(config)
        for group in groups:
            for field in group["fields"]:
                key, value, tip, meta = _unpack(field)
                if key in READ_ONLY_KEYS:
                    assert meta.get("type") == "readonly", (
                        f"{group['title']}.{key} in READ_ONLY_KEYS but meta type is "
                        f"'{meta.get('type')}'"
                    )


class TestApplyConfigOverride:
    """Smoke-test _apply_config_override for every overridable field type."""

    @staticmethod
    def _cfg():
        return _load_test_config()

    # --- experiment ---

    def test_experiment_name(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "experiment", "name", "test-run")
        assert new.experiment.name == "test-run"

    def test_experiment_seed(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "experiment", "seed", "42")
        assert new.experiment.seed == 42

    def test_experiment_device(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "experiment", "device", "mps")
        assert new.experiment.device == "mps"

    def test_experiment_precision(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "experiment", "precision", "fp16")
        assert new.experiment.precision == "fp16"

    # --- observation ---

    def test_observation_mode(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "mode", "canvas")
        assert new.observation.mode == "canvas"

    def test_observation_width(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "width", "128")
        assert new.observation.width == 128

    def test_observation_height(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "height", "128")
        assert new.observation.height == 128

    def test_observation_grayscale(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "grayscale", "true")
        assert new.observation.grayscale is True

    def test_observation_frame_stack(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "frame_stack", "2")
        assert new.observation.frame_stack == 2

    def test_observation_normalize(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "observation", "normalize", "true")
        assert new.observation.normalize is True

    # --- agent ---

    def test_agent_algorithm(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "algorithm", "linear_q")
        assert new.agent.algorithm == "linear_q"

    def test_agent_gamma(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "gamma", "0.99")
        assert new.agent.gamma == 0.99

    def test_agent_n_step(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "n_step", "3")
        assert new.agent.n_step == 3

    def test_agent_batch_size(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "batch_size", "64")
        assert new.agent.batch_size == 64

    def test_agent_target_update_interval(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "target_update_interval", "1000")
        assert new.agent.target_update_interval == 1000

    def test_agent_learning_starts(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "learning_starts", "5000")
        assert new.agent.learning_starts == 5000

    def test_agent_train_every(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "agent", "train_every", "2")
        assert new.agent.train_every == 2

    # --- exploration ---

    def test_exploration_epsilon_start(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "exploration", "epsilon_start", "0.5")
        assert new.exploration.epsilon_start == 0.5

    def test_exploration_epsilon_end(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "exploration", "epsilon_end", "0.01")
        assert new.exploration.epsilon_end == 0.01

    def test_exploration_epsilon_decay_steps(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "exploration", "epsilon_decay_steps", "200000")
        assert new.exploration.epsilon_decay_steps == 200000

    # --- replay ---

    def test_replay_capacity(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "replay", "capacity", "500000")
        assert new.replay.capacity == 500000

    def test_replay_prioritized(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "replay", "prioritized", "false")
        assert new.replay.prioritized is False

    def test_replay_sequence_length(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "replay", "sequence_length", "32")
        assert new.replay.sequence_length == 32

    # --- world_model ---

    def test_world_model_enabled(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "world_model", "enabled", "false")
        assert new.world_model.enabled is False

    def test_world_model_latent_dim(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "world_model", "latent_dim", "256")
        assert new.world_model.latent_dim == 256

    # --- game ---

    def test_game_fps(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "game", "fps", "60")
        assert new.game.fps == 60

    def test_game_action_repeat(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "game", "action_repeat", "4")
        assert new.game.action_repeat == 4

    def test_game_max_steps_per_episode(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "game", "max_steps_per_episode", "5000")
        assert new.game.max_steps_per_episode == 5000

    def test_game_headless(self):
        cfg = self._cfg()
        new = _apply_config_override(cfg, "game", "headless", "false")
        assert new.game.headless is False

    # --- error cases ---

    def test_unknown_field_raises(self):
        cfg = self._cfg()
        raised = False
        try:
            _apply_config_override(cfg, "game", "max_steps", "1000")
        except ValueError as e:
            raised = True
            assert "unknown config field" in str(e)
        assert raised, "expected ValueError for unknown field"

    def test_unknown_group_raises(self):
        cfg = self._cfg()
        raised = False
        try:
            _apply_config_override(cfg, "nonexistent", "foo", "bar")
        except ValueError as e:
            raised = True
            assert "unknown config field" in str(e)
        assert raised, "expected ValueError for unknown group"

    def test_observation_size_raises(self):
        """The old composite key 'size' should fail."""
        cfg = self._cfg()
        raised = False
        try:
            _apply_config_override(cfg, "observation", "size", "84x84")
        except ValueError as e:
            raised = True
            assert "unknown config field" in str(e)
        assert raised, "expected ValueError for composite 'size' key"

# JEPA-RL Browser Game Agent

JEPA-RL is a research project for training reinforcement learning agents to play arbitrary web browser games from visual observations. The core idea is to decouple representation learning from reward optimization: a Joint-Embedding Predictive Architecture (JEPA) learns compact action-conditioned latent states from gameplay, and a policy/value learner uses those latents to maximize game score.

The first target is a Breakout-like browser game benchmark. Version 1 is complete when the system can train from screenshots, run a DQN baseline, train a JEPA+DQN agent, log scores and model diagnostics, and demonstrate that JEPA matches or exceeds pixel-DQN sample efficiency.

## Status

This repository now has an initial runnable browser-game scaffold: package metadata, a `jepa-rl` CLI, typed configuration loading and validation, starter configs, a local Breakout-like HTML game, Playwright browser control, screenshot observations, DOM score reading, discrete keyboard action parsing, an in-memory replay buffer, run artifact directory creation, a minimal NumPy pixel-Q smoke trainer, and unit tests.

Implemented CLI commands:

- `jepa-rl validate-config --config configs/games/breakout.yaml`
- `jepa-rl init-run --config configs/games/breakout.yaml --experiment <name>`
- `jepa-rl open-game --config configs/games/breakout.yaml`
- `jepa-rl ml-smoke`
- `jepa-rl dashboard --run runs/<experiment>`
- `jepa-rl ui --config configs/games/breakout.yaml`
- `jepa-rl collect-random --config configs/games/breakout.yaml --experiment <name>`
- `jepa-rl train --config configs/games/breakout.yaml --experiment <name>`
- `jepa-rl eval --config configs/games/breakout.yaml --checkpoint <path>`

The current `train` command is a deliberately small linear Q-learning smoke path over downsampled pixels. It proves that the browser, actions, rewards, metrics, and checkpoint path work. It is not the planned DQN/JEPA implementation yet. `train-world` remains a planned stub.

The canonical specification lives in:

- [docs/design_doc.md](docs/design_doc.md) — full design document with model details, configuration contract, training phases, evaluation metrics, risks, and the V1/V2/V3 JEPA roadmap.
- [TODO.md](TODO.md) — phase-by-phase implementation checklist derived from the design doc.
- [DEVELOPMENT.md](DEVELOPMENT.md) — development methodology: per-phase verification gates, dependency graph, parallel-track decomposition, agent-dispatch playbook, and standard verification recipes.
- [CLAUDE.md](CLAUDE.md) — guidance for AI-assisted development in this repo.

Reference papers are listed in the [Reference Papers](#reference-papers) section below. PDFs are stored locally in `docs/` but are not committed to the repository.

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install uv once (e.g. `brew install uv`, `pipx install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
uv sync                                                            # creates .venv, installs project + dev deps from uv.lock
uv run jepa-rl validate-config --config configs/games/breakout.yaml
uv run jepa-rl validate-config --config configs/presets/tiny.yaml
uv run jepa-rl validate-config --config configs/presets/small.yaml
uv run jepa-rl validate-config --config configs/presets/base.yaml
uv run pytest
uv run ruff check src tests
```

If Playwright reports that Chromium is missing, install it once:

```bash
uv run playwright install chromium
```

Run the local browser game and train the current smoke model:

```bash
uv run jepa-rl ml-smoke
uv run jepa-rl open-game --config configs/games/breakout.yaml --seconds 10
uv run jepa-rl open-game --config configs/games/breakout.yaml --seconds 10 --random-steps 5
uv run jepa-rl ui --config configs/games/breakout.yaml
uv run jepa-rl collect-random --config configs/games/breakout.yaml --experiment smoke_random --episodes 1 --max-steps 25
uv run jepa-rl train --config configs/games/breakout.yaml --experiment smoke_train --steps 200 --learning-starts 0
uv run jepa-rl dashboard --run runs/smoke_train --open
uv run jepa-rl eval --config configs/games/breakout.yaml --checkpoint runs/smoke_train/checkpoints/latest.npz --episodes 3
```

`open-game` always opens visible Chromium. Add `--hold` to keep it open until Ctrl+C, or close the browser window directly; both paths exit cleanly. `ui` opens a local control panel with the game embedded, Start/Stop/Eval buttons, and live charts for score, loss, epsilon, TD error, replay size, updates, and progress. Add `--headed` to `collect-random`, `train`, or `eval` to watch Chromium play during those commands. `ml-smoke` is a fast numeric check that the current linear Q learner reduces loss and changes weights before involving the browser. `train` writes `runs/<experiment>/dashboard.html`, and `dashboard --open` regenerates and opens that static UI panel. `uv.lock` is committed; CI installs from it with `uv sync --frozen`.

## Project Goals

### Primary
- Train agents to play arbitrary browser games from screenshots or canvas captures.
- Use high score as the primary optimization target.
- Support arbitrary games through configurable YAML game adapters.
- Learn reusable visual and temporal representations with JEPA-style latent prediction.
- Allow model scale and hyperparameters to be configured without code changes.
- Support both keyboard and mouse action spaces.
- Produce reproducible experiments with logged configs, metrics, checkpoints, and replay videos.

### Secondary
- Offline pretraining from recorded gameplay (random, scripted, human, or previous-agent runs).
- Online self-play and autonomous exploration.
- Latent-space planning once the JEPA model is sufficiently accurate.
- Multi-game training for general representations.
- Curriculum learning from simple games to more complex games.

## Non-Goals for Version 1

- No solving every arbitrary website without per-game adapter configuration.
- No reading personal browser sessions or private user data.
- No use of game source code as privileged state.
- No mutation of game score, memory, local storage, or network requests for advantage.
- No general-purpose web automation agent beyond score-driven games.
- No foundation-scale pretraining from scratch.

## Version 1 Scope

- Playwright-controlled Chromium environment.
- Screenshot and canvas observation modes.
- Discrete keyboard action spaces with key combinations and action repeat.
- DOM score reader with OCR fallback.
- Random policy baseline runner.
- Replay buffer with both transition sampling (DQN) and contiguous sequence sampling (JEPA).
- Pixel DQN baseline (Double DQN, dueling, n-step returns, prioritized replay).
- Action-conditioned JEPA pretraining with EMA target encoder and variance/covariance regularization.
- Frozen JEPA + DQN.
- Joint JEPA + DQN training.
- Deterministic evaluation episodes.
- Video recording of selected episodes.
- Metrics, checkpoints, and config snapshots.

## System Architecture

```text
Browser Game (Chromium)
  |
  v
Browser Adapter
  - Playwright / CDP
  - screenshot or canvas frames
  - keyboard and mouse actions
  - score, reset, done detection
  |
  v
Environment Wrapper
  - resize, crop, normalize
  - frame stacking
  - reward shaping
  - action repeat
  |
  v
Replay Buffer
  - transitions for DQN
  - contiguous sequences for JEPA
  |
  +--> JEPA World Model
  |      - online encoder
  |      - EMA target encoder (stop-gradient)
  |      - action-conditioned predictor
  |      - variance/covariance regularization
  |      - latent collapse metrics
  |
  v
RL Agent
  - DQN family first (Double DQN, dueling, n-step, prioritized replay)
  - PPO later (actor-critic, GAE, optional recurrent)
  - optional intrinsic reward from JEPA prediction error
  |
  v
Evaluator and Logger
  - scores, videos, losses
  - representation diagnostics
  - operational metrics
  - checkpoints
```

## Key Design Choice: JEPA for Browser Game RL

Pixel-based RL often learns directly from rewards, which can be sparse, noisy, delayed, or game-specific. A JEPA-style model learns useful latent representations by predicting future or masked target embeddings rather than reconstructing raw pixels. For browser games, this matters because many visual details are irrelevant to gameplay: background art, animation effects, browser chrome, ads, decorative UI, or particle effects.

The proposed system uses an **action-conditioned temporal JEPA**:

```text
z_t      = Encoder(o_{t-L+1:t})
y_{t+k}  = TargetEncoder(o_{t+k})            # stop-gradient, EMA copy of Encoder
p_{t+k}  = Predictor(z_t, a_t, ..., a_{t+k-1}, k)

L_jepa   = mean_k || normalize(p_{t+k}) - normalize(y_{t+k}) ||_2^2
L_world  = L_jepa + lambda_var * L_variance + lambda_cov * L_covariance
```

Prediction is in latent space, not pixel space. The policy receives `z_t` (and optional uncertainty / scalar features) instead of raw pixels.

## Recommended Implementation Stack

- Python 3.11 or newer.
- PyTorch for models and optimization.
- Playwright for browser control (Chromium).
- NumPy, Pillow, and OpenCV for image handling.
- Pydantic or OmegaConf for typed configuration.
- TensorBoard, Weights & Biases, or MLflow for experiment logging.
- Pytest for tests.
- Ruff and mypy for code quality once the package structure exists.
- uv for environment management, dependency resolution, and the locked `uv.lock` file.

These are defaults to start with, not hard requirements.

## Planned Repository Layout

```text
jepa-rl/
  configs/
    base.yaml
    presets/
      tiny.yaml          # CPU smoke tests
      small.yaml         # first serious Breakout-like experiments
      base.yaml          # multi-game / visually richer games (ViT)
    games/
      breakout.yaml
      snake.yaml
  docs/
    design_doc.md
    *.pdf                # reference papers (see below)
  scripts/
    collect_random.py
    eval.py
    train.py
  src/
    jepa_rl/
      browser/
        playwright_env.py
        score_readers.py
        action_spaces.py
      envs/
        browser_game_env.py
        wrappers.py
      models/
        encoders.py      # conv + ViT
        jepa.py          # online + target encoder + losses
        predictors.py    # transformer + GRU
        q_networks.py
        policies.py
      replay/
        replay_buffer.py
        sequence_sampler.py
      training/
        train_world.py
        train_dqn.py
        train_joint.py
        evaluate.py
      utils/
        checkpointing.py
        config.py
        logging.py
        video.py
  tests/
    test_action_spaces.py
    test_config.py
    test_env_reset.py
    test_jepa_shapes.py
    test_replay.py
    test_score_reader.py
```

## Target CLI

The CLI exists, but only config validation and run initialization are implemented so far. The intended full workflow is:

```bash
# Phase 1: Random data collection (also gives the random-policy baseline number)
jepa-rl collect-random --config configs/games/breakout.yaml --experiment breakout_random_v1

# Phase 2: Offline JEPA pretraining from collected replay
jepa-rl train-world --config configs/games/breakout.yaml --replay runs/breakout_random_v1/replay

# Phase 3: Joint JEPA + DQN training
jepa-rl train --config configs/games/breakout.yaml --experiment breakout_jepa_dqn_v1

# Evaluation with deterministic action selection
jepa-rl eval --checkpoint runs/breakout_jepa_dqn_v1/latest.pt --episodes 50

# Config validation without launching the browser
jepa-rl validate-config --config configs/games/breakout.yaml
```

Expected training outputs:

- Best score achieved.
- Mean, median, and p95 evaluation score over deterministic episodes.
- Score at fixed environment-step budgets: 100k, 500k, 1M, and 5M.
- Learning curves.
- JEPA prediction losses by horizon.
- Latent variance, covariance, and collapse indicators.
- Q-learning losses, TD error, Q-value scale, and exploration rate.
- Browser reset and score-reader failure rates.
- Replay videos for best and representative episodes.
- Model checkpoints and config snapshots.

## Configuration Contract

Every important experiment setting is declarative. A run is reproducible from a saved config snapshot plus code version. Configs are composable: base config + game config + experiment override. Invalid combinations should fail fast before browser launch.

Example game config:

```yaml
experiment:
  name: breakout_jepa_dqn_small
  seed: 42
  device: cuda
  precision: bf16
  output_dir: runs/

game:
  name: breakout
  url: "https://example.com/breakout"
  browser: chromium
  headless: true
  fps: 30
  action_repeat: 4
  max_steps_per_episode: 10000

observation:
  mode: screenshot              # screenshot | canvas | dom_assisted | hybrid
  width: 160
  height: 120
  grayscale: false
  frame_stack: 4
  crop: null
  normalize: true

actions:
  type: discrete_keyboard       # discrete_keyboard | discrete_mouse | hybrid_discrete | continuous_mouse (V2+)
  keys:
    - noop
    - ArrowLeft
    - ArrowRight
    - Space
    - ArrowLeft+Space
    - ArrowRight+Space

reward:
  type: score_delta             # reward_t = score_t - score_{t-1}
  score_reader: dom             # dom | ocr | template | js_callback (privileged)
  score_selector: "#score"
  survival_bonus: 0.0
  idle_penalty: 0.0
  death_penalty: 0.0
  clip_rewards: false

world_model:
  enabled: true
  latent_dim: 512
  encoder:
    type: conv                  # conv | vit
    hidden_channels: [32, 64, 128, 256]
  predictor:
    type: transformer           # transformer | gru
    hidden_dim: 512
    depth: 4
    num_heads: 8
    horizons: [1, 2, 4, 8]
    action_embed_dim: 64
    action_chunk_size: 4
    conditioning:
      action_sequence: true
      policy_embedding: false   # Version 2
      task_or_goal_embedding: false  # Version 2
      value_guidance: false     # Version 3
  target_encoder:
    ema_tau_start: 0.996
    ema_tau_end: 0.9999
    stop_gradient: true
  loss:
    prediction: cosine_mse
    lambda_var: 1.0
    lambda_cov: 0.04
    latent_norm: true
    variance_floor: 1.0
  planning:                     # Version 3 (off in V1)
    enabled: false
    method: latent_mpc
    horizon: 8
    num_candidates: 64

agent:
  algorithm: dqn                # dqn | ppo (later)
  gamma: 0.997
  n_step: 5
  batch_size: 256
  q_network:
    hidden_dims: [512, 512]
    dueling: true
    distributional: false
  target_update_interval: 2000

replay:
  capacity: 1000000
  prioritized: true
  priority_alpha: 0.6
  priority_beta_start: 0.4
  priority_beta_end: 1.0
  sequence_length: 16

exploration:
  type: epsilon_greedy
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 500000
  intrinsic_reward:
    enabled: true
    source: jepa_prediction_error
    beta: 0.01
    normalize: true

training:
  passive_pretrain_steps: 100000
  total_env_steps: 5000000
  learning_starts: 10000
  train_every: 4
  world_updates_per_env_step: 1
  policy_updates_per_env_step: 1
  eval_interval_steps: 50000
  checkpoint_interval_steps: 50000

evaluation:
  episodes: 20
  deterministic: true
  record_video: true
  save_best_by: mean_score
```

## Model Size Presets

| Preset | Use Case | Input | Latent | Encoder | Predictor | Replay |
|---|---|---|---|---|---|---|
| `tiny` | CPU/GPU smoke tests | 84x84 grayscale | 128 | conv [16,32,64] | depth 2, 4 heads | 100k |
| `small` | First serious Breakout-like runs | 160x120 RGB | 512 | conv [32,64,128,256] | depth 4, 8 heads | 1M |
| `base` | Multi-game / visually rich | 224x224 RGB | 768 | ViT depth 8, 8 heads | depth 6, 12 heads | 2M |

## Training Phases

0. **Passive gameplay collection** — Collect or import gameplay videos before online RL. Sources: random play, scripted play, human demonstration, or previous agent runs. Seeds the visual temporal representation cheaply.

1. **Random or scripted trajectory collection** — Action-labeled trajectories. Populates replay, validates environment stepping and reset behavior, and measures the random-policy baseline score.

2. **JEPA pretraining** — Train encoder, target encoder, and predictor on replay sequences. Track prediction loss by horizon and collapse diagnostics.

3. **Baseline DQN training** — Pixel DQN without JEPA. Establishes the minimum serious baseline.

4. **Frozen JEPA + DQN training** — Freeze a pretrained JEPA encoder and train a DQN head on latent states.

5. **Joint JEPA + DQN training** — Alternate online environment rollout, replay insertion, JEPA updates, and Q-learning updates.

6. **High-score fine-tuning** — Lower exploration, select top checkpoints by deterministic evaluation, and record representative videos.

7. **Optional latent planning fine-tuning** — Later versions can use latent action search or value-guided planning after the world model is reliable.

## JEPA Roadmap

The JEPA component evolves across three versions. Each version adds new conditioning to the predictor.

### Version 1: Action-Conditioned JEPA (MVP)

Predict future target latents from current visual context and a future action sequence:

```text
p_{t+k} = Predictor(z_t, a_t, a_{t+1}, ..., a_{t+k-1}, k)
```

This is the MVP for Breakout-like games. Best matched to: ACT-JEPA-style action chunking; DQN as the policy.

### Version 2: Policy-Conditioned JEPA

Add policy/task/goal embeddings for multi-game transfer and reward-free pretraining:

```text
p_{t+k} = Predictor(z_t, pi_id, a_{t:t+k-1}, goal, k)
```

Useful for: multi-game training, offline reward-free data, fast adaptation to new games or scoring rules. Best matched to: TD-JEPA, V-JEPA 2 stage-1 video pretraining + stage-2 action post-training.

### Version 3: Value-Guided JEPA Planning

Shape latent geometry so high-value states are easier to search for during planning:

```text
L_world = L_jepa
        + lambda_var * L_variance
        + lambda_cov * L_covariance
        + lambda_action * L_action_chunk
        + lambda_value_shape * L_value_shape
```

Use latent action search (latent MPC) at evaluation time and during difficult states. Best matched to: value-guided JEPA planning, TD-MPC2-style latent planning for hybrid keyboard/mouse spaces.

## Evaluation Baselines

Minimum baselines:

- Random policy.
- Pixel DQN without JEPA.
- DQN with frozen JEPA pretrained on random action-labeled data.
- DQN with frozen JEPA pretrained on passive gameplay video.
- DQN with jointly trained action-conditioned JEPA.

Optional baselines:

- PPO from pixels.
- Autoencoder representation + DQN.
- Contrastive representation + DQN.
- Dreamer-style latent actor-critic (DreamerV2/V3).
- EfficientZero or MuZero-style planning baseline.
- TD-MPC-style latent planner for mouse-heavy games.
- Human demonstration warm start.

## Evaluation Metrics

### Primary
- Best score achieved.
- Mean, median, and p95 score over deterministic evaluation episodes.
- Score at fixed budgets: 100k, 500k, 1M, and 5M environment steps.
- Time to target score.
- Sample efficiency relative to pixel DQN.
- Wall-clock training time.

### Representation
- JEPA prediction loss by horizon.
- Latent variance and covariance.
- Collapse indicators.
- Action-sequence prediction accuracy.
- Prediction error on high-score versus low-score trajectories.
- Transfer performance with frozen encoder and a new policy head.

### RL
- Episode return.
- TD error.
- Q-value scale.
- Action entropy.
- Exploration rate.
- Replay priority distribution.

### Operational
- Environment steps per second.
- Browser reset failure rate.
- Score-reader failure rate.
- GPU utilization.
- Checkpoint save and restore success.

### Planning (later versions)
- Planner-improved score versus policy-only score.
- Planning latency per decision.
- Candidate action sequence success rate.
- Value-prediction calibration.

## Risks and Mitigations

| Risk | Mitigations |
|---|---|
| **JEPA collapse** | EMA target encoder, stop-gradient target branch, latent normalization, variance/covariance regularization, collapse metrics in logging |
| **Score reader fragility** | Prefer DOM selectors, validate readers before training, track reader confidence, save screenshots on failure, allow user-supplied parser |
| **Browser timing instability** | Fixed viewport, frame pacing, action repeat, environment step synchronization, timing diagnostics |
| **Sparse rewards** | Random/scripted warm-start, intrinsic reward from JEPA prediction error, demos, curriculum, exploration schedules |
| **Overfitting to one game** | Multi-game training, domain randomization, gameplay-preserving visual augmentations, held-out game evaluation |

## Safety and Isolation

Browser runs should use:

- Dedicated browser profiles.
- No personal logged-in sessions.
- Fixed viewport.
- Headless mode by default.
- Network allowlist where practical.
- No extension access.
- Optional JavaScript callbacks only when explicitly configured (and marked as privileged in run metadata).
- No mutation of game source, score, memory, local storage, or network requests for advantage.

## Open Questions

These remain open from the design doc and should be resolved during implementation:

1. Should the first implementation use DQN only, or support PPO from the beginning?
2. Should score extraction be treated as privileged state, or only as reward signal?
3. How much DOM assistance is acceptable for the target use case?
4. Should JEPA pretraining use only random data, or include early policy rollouts?
5. How should continuous mouse control be represented in the JEPA predictor?
6. What browser games should define the initial benchmark suite?
7. Should the world model support latent-space planning in version 1, or only representation learning?

## Reference Papers

These papers directly motivate the design. Each informs a specific part of the roadmap. PDFs are stored locally in `docs/` but are **not committed to the repository** — download or locate them separately.

| Paper | Informs |
|---|---|
| **V-JEPA 2**: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2506.09985) | Phase 0 passive video pretraining + Phase 2 action-conditioned post-training |
| **ACT-JEPA**: Joint-Embedding Predictive Architecture for Efficient Policy Representation Learning (2501.14622) | Action chunking, imitation warm-start, action-sequence prediction diagnostics |
| **TD-JEPA**: Latent-Predictive Representations for Zero-Shot Reinforcement Learning (2510.00739) | V2 policy-conditioned predictor, multi-step horizons, reward-free pretraining |
| **Value-Guided Action Planning with JEPA World Models** (2601.00844) | V3 value-shaped latent loss, high-score goal embeddings, latent action search |
| **DreamerV3**: Mastering Diverse Control Tasks through World Models | Robust default hyperparameters across games, latent actor-critic baseline |
| **TD-MPC2**: Scalable, Robust World Models for Continuous Control (2310.16828) | Future decoder-free latent MPC backend, hybrid/continuous action planning |

EfficientZero and DreamerV2 are also referenced in [docs/design_doc.md §23](docs/design_doc.md) as comparison baselines.

## Version 1 Acceptance Criteria

Version 1 is done when:

- A user can configure a browser game through YAML.
- The agent can train from visual observations.
- Configurable model sizes and hyperparameters are supported.
- Score, loss, diagnostic, video, checkpoint, and config artifacts are logged.
- Random policy, pixel DQN, frozen JEPA+DQN, and joint JEPA+DQN are available.
- JEPA collapse and unstable representation learning are detectable from metrics.
- At least one Breakout-like game shows JEPA+DQN outperforming random and matching or exceeding pixel-DQN sample efficiency.

## Recommended Initial Experiment

Start with a Breakout clone because it has:

- Clear score signal.
- Small discrete action space.
- Simple visual dynamics.
- Known RL benchmark behavior.
- Easy replay video interpretation.

Recommended starting setup:

```yaml
preset: small
algorithm: dqn
world_model: jepa_conv_transformer
total_env_steps: 2000000
observation: 160x120_rgb_frame_stack_4
action_repeat: 4
horizons: [1, 2, 4, 8]
latent_dim: 512
intrinsic_reward_beta: 0.01
```

The first success milestone is not "superhuman Breakout." It is:

1. Stable browser control.
2. Reliable score extraction.
3. DQN beats random.
4. JEPA representation remains non-collapsed (variance floor maintained, no rank degeneration).
5. JEPA+DQN reaches the same score as pixel DQN in fewer environment steps.

## Development Notes

- Keep environment code separate from model and training code.
- Make score readers testable without launching long training runs.
- Validate every config before starting a browser.
- Save enough diagnostics to debug failed score extraction, failed reset, and blank observations.
- Prefer small deterministic smoke tests before GPU-heavy training.
- Treat every training run as an experiment artifact: config, git revision when available, environment metadata, metrics, videos, and checkpoints should be saved together.
- Privileged score readers (JavaScript callbacks, direct DOM mutation hooks) must be marked as such in run metadata so reports clearly state when an evaluation used assistance beyond pure visual play.

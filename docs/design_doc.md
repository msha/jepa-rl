# Design Document: JEPA-Based Reinforcement Learning Agent for Arbitrary Browser Games

## 1. Summary

This document proposes a reinforcement learning system that can play arbitrary web browser games, such as Breakout, Snake, Flappy Bird, simple platformers, clicker games, and other score-driven browser environments. The agent observes the browser screen, learns a compact predictive world representation using a Joint-Embedding Predictive Architecture (JEPA), and optimizes gameplay policy to maximize high score.

The system is designed to be configurable. Model sizes, context windows, action spaces, reward shaping, replay sizes, optimization settings, exploration behavior, and environment wrappers should all be controlled from declarative configuration files.

The core idea is to decouple representation learning from reward optimization:

1. A browser automation layer captures visual observations and executes keyboard, mouse, or touch-like actions.
2. A JEPA world model learns predictive latent representations from screen observations and action-conditioned transitions.
3. A policy/value learner uses those latent states to choose actions that maximize score.
4. An evaluator tracks high score, learning stability, generalization, and sample efficiency across games.

This design targets a research-grade implementation first, with a path toward a robust general browser game agent.

---

## 2. Goals

### 2.1 Primary Goals

- Train an agent to play web browser games from pixels or DOM-assisted visual state.
- Maximize game score, with high score as the primary optimization metric.
- Support arbitrary games through a configurable game adapter interface.
- Learn reusable visual and temporal representations using a JEPA-style predictive latent objective.
- Allow model scale and hyperparameters to be configured without code changes.
- Support both keyboard and mouse action spaces.
- Provide reproducible experiments, logging, checkpoints, and evaluation reports.

### 2.2 Secondary Goals

- Support offline pretraining from recorded gameplay.
- Support online self-play / autonomous exploration.
- Support human demonstration data as optional warm-start input.
- Allow planning in latent space once the JEPA model is sufficiently accurate.
- Support multi-game training to encourage general representations.
- Support curriculum learning from simple games to more complex games.

### 2.3 Non-Goals for Version 1

- Solving every arbitrary website or game without adapter configuration.
- Reading private user data from the browser.
- Using game source code directly as privileged state.
- Cheating by modifying score, memory, network requests, or game code.
- Building a full general-purpose web automation agent beyond game-playing.
- Training a foundation-scale model from scratch.

---

## 3. Product Requirements

### 3.1 User Workflow

A user should be able to define a game configuration like this:

```yaml
game:
  name: breakout_demo
  url: "https://example.com/breakout"
  observation:
    mode: screenshot
    width: 160
    height: 120
    grayscale: false
    frame_stack: 4
  actions:
    type: discrete
    keys:
      - noop
      - ArrowLeft
      - ArrowRight
      - Space
      - ArrowLeft+Space
      - ArrowRight+Space
  reward:
    score_reader: ocr
    score_region: [8, 8, 120, 32]
    terminal_detection: visual_template
  training:
    max_steps_per_episode: 10000
    target_score: 500
```

Then run:

```bash
jepa-rl train --config configs/games/breakout_demo.yaml --experiment breakout_jepa_v1
jepa-rl eval --checkpoint runs/breakout_jepa_v1/latest.pt --episodes 50
```

The output should include:

- Best score achieved.
- Mean, median, and p95 score over evaluation episodes.
- Learning curves.
- Replay videos of best episodes.
- Model checkpoints.
- Hyperparameter snapshot.
- Failure diagnostics.

---

## 4. System Architecture

```text
+-----------------------+
| Browser Game           |
| Chrome / Chromium      |
+-----------+-----------+
            |
            v
+-----------------------+
| Browser Adapter        |
| Playwright / CDP       |
| Screenshot, Actions,   |
| Score, Reset, Done     |
+-----------+-----------+
            |
            v
+-----------------------+
| Environment Wrapper    |
| Frame Stack, Resize,   |
| Reward, Termination    |
+-----------+-----------+
            |
            v
+-----------------------+
| Replay Buffer          |
| Observations, Actions, |
| Rewards, Latents       |
+-----------+-----------+
            |
            v
+-----------------------+       +-----------------------+
| JEPA World Model       |<----->| Representation Trainer |
| Encoder, Predictor,    |       | Self-supervised Losses |
| Target Encoder         |       +-----------------------+
+-----------+-----------+
            |
            v
+-----------------------+       +-----------------------+
| RL Agent               |<----->| Policy Trainer         |
| Policy, Value, Q,      |       | PPO / DQN / SAC-style  |
| Exploration Controller |       | Updates                |
+-----------+-----------+       +-----------------------+
            |
            v
+-----------------------+
| Evaluator + Logger     |
| Scores, Videos,        |
| Metrics, Checkpoints   |
+-----------------------+
```

---

## 5. Key Design Choice: JEPA for Browser Game RL

Traditional pixel-based RL often learns directly from rewards, which can be sparse, noisy, delayed, or game-specific. A JEPA-style model learns useful latent representations by predicting future or masked target embeddings rather than reconstructing raw pixels. For browser games, this is attractive because many visual details are irrelevant to gameplay: background art, animation effects, browser chrome, ads, decorative UI, or particle effects.

The proposed system uses an action-conditioned temporal JEPA:

- Encode current observation sequence into latent state `z_t`.
- Encode future target observation into target latent `y_{t+k}` using a momentum or stop-gradient target encoder.
- Predict `y_{t+k}` from current latent state plus action sequence.
- Train prediction in latent space, not pixel space.
- Feed learned latent states to the RL policy/value model.

This gives the policy a representation that emphasizes controllable, score-relevant game dynamics.

---

## 6. Environment Interface

### 6.1 Game Adapter API

Each game should implement or configure a common interface:

```python
class BrowserGameEnv:
    def reset(self) -> Observation:
        ...

    def step(self, action: Action) -> StepResult:
        ...

    def observe(self) -> Observation:
        ...

    def read_score(self) -> float:
        ...

    def is_done(self) -> bool:
        ...

    def render_video_frame(self) -> np.ndarray:
        ...
```

### 6.2 Observation Modes

Supported observation modes:

1. `screenshot`: raw browser/game canvas screenshot.
2. `canvas`: direct capture of HTML canvas if available.
3. `dom_assisted`: screenshot plus DOM-extracted text such as score, timer, lives.
4. `hybrid`: screenshot, DOM text, and optional OCR output.

Version 1 should prioritize `screenshot` and `canvas` because they are closest to general game-playing from vision.

### 6.3 Action Modes

Supported action types:

- `discrete_keyboard`: fixed list of key combinations.
- `discrete_mouse`: fixed click zones or drag primitives.
- `hybrid_discrete`: keyboard plus mouse primitives.
- `continuous_mouse`: normalized pointer movement and click probability.

Version 1 should use discrete action spaces. Continuous mouse control can come later.

Example discrete actions:

```yaml
actions:
  type: discrete
  repeat: 4
  keys:
    - noop
    - ArrowLeft
    - ArrowRight
    - ArrowUp
    - ArrowDown
    - Space
    - ArrowLeft+Space
    - ArrowRight+Space
```

### 6.4 Score and Reward Extraction

Reward can be derived from score deltas:

```text
reward_t = score_t - score_{t-1}
```

Additional shaping is optional:

```text
reward_t = score_delta
         + survival_bonus
         - idle_penalty
         - death_penalty
```

Score readers:

- DOM selector: preferred when score is represented in HTML.
- Canvas OCR: useful for canvas games.
- Visual template matching: useful for digits/lives icons.
- User-supplied JavaScript callback: optional but should be marked as privileged.

The default should avoid privileged game internals unless explicitly configured.

---

## 7. JEPA World Model

### 7.1 Inputs

The JEPA model receives:

- Frame stack or short video clip: `o_{t-L+1:t}`.
- Optional action history: `a_{t-L+1:t}`.
- Optional scalar features: score delta, lives, timer, terminal flag.

### 7.2 Encoders

Recommended default encoder for version 1:

- Small convolutional encoder for low-resolution games.
- Optional Vision Transformer encoder for larger experiments.

Convolutional default:

```yaml
world_model:
  encoder:
    type: conv
    input_channels: 12        # RGB x frame_stack=4
    hidden_channels: [32, 64, 128, 256]
    kernel_sizes: [8, 4, 3, 3]
    strides: [4, 2, 1, 1]
    activation: gelu
    norm: layer_norm
    latent_dim: 512
```

ViT option:

```yaml
world_model:
  encoder:
    type: vit
    image_size: [160, 120]
    patch_size: 8
    embed_dim: 384
    depth: 6
    num_heads: 6
    mlp_ratio: 4
    latent_dim: 512
```

### 7.3 Predictor

The predictor maps context latent plus action sequence to future target latent:

```text
p_{t+k} = Predictor(z_t, a_t, a_{t+1}, ..., a_{t+k-1}, k)
```

Recommended design:

- Action embedding table for discrete actions.
- Temporal transformer or GRU over future action sequence.
- MLP projection head.
- Optional horizon embedding.

```yaml
world_model:
  predictor:
    type: transformer
    action_embed_dim: 64
    hidden_dim: 512
    depth: 4
    num_heads: 8
    mlp_ratio: 4
    dropout: 0.1
    horizons: [1, 2, 4, 8]
```

### 7.4 Target Encoder

Use an exponential moving average target encoder:

```text
theta_target = tau * theta_target + (1 - tau) * theta_online
```

Default:

```yaml
world_model:
  target_encoder:
    ema_tau_start: 0.996
    ema_tau_end: 0.9999
    stop_gradient: true
```

### 7.5 Losses

Primary JEPA prediction loss:

```text
L_jepa = mean_horizon || normalize(p_{t+k}) - normalize(y_{t+k}) ||_2^2
```

Collapse prevention options:

- EMA target encoder.
- Latent normalization.
- Variance/covariance regularization.
- Predictor bottleneck.
- Batch-level embedding variance floor.

Recommended combined loss:

```text
L_world = L_jepa + lambda_var * L_variance + lambda_cov * L_covariance
```

Default:

```yaml
world_model:
  loss:
    prediction: cosine_mse
    lambda_var: 1.0
    lambda_cov: 0.04
    latent_norm: true
    variance_floor: 1.0
```

---

## 8. Reinforcement Learning Agent

### 8.1 Policy Inputs

The policy receives:

- `z_t`: current JEPA latent state.
- Optional recurrent hidden state.
- Optional score/lives/timer scalar features.
- Optional uncertainty estimates from JEPA prediction error.

### 8.2 RL Algorithms

Version 1 should support two policy learner options:

#### Option A: DQN-Family Agent

Best for discrete browser games and Atari-like settings.

Recommended components:

- Double DQN.
- Dueling Q-network.
- Prioritized replay.
- N-step returns.
- Distributional value head as optional extension.

Good default for Breakout-like games.

#### Option B: PPO Agent

Best for broader action spaces and simpler implementation with recurrent policies.

Recommended components:

- Actor-critic policy.
- Generalized Advantage Estimation.
- Entropy bonus.
- Optional recurrent GRU/LSTM.

Good default for mixed game types and online rollouts.

### 8.3 Recommended Version 1 Choice

Use DQN-family for the first implementation because most target games can be represented with discrete actions and score-delta rewards. Add PPO once the browser adapter and JEPA training are stable.

### 8.4 Q-Network Example

```yaml
agent:
  algorithm: dqn
  q_network:
    input_dim: 512
    hidden_dims: [512, 512]
    dueling: true
    distributional: false
  gamma: 0.997
  n_step: 5
  batch_size: 256
  target_update_interval: 2000
  learning_starts: 10000
  train_every: 4
  gradient_steps: 1
```

### 8.5 Exploration

Exploration should be configurable:

```yaml
exploration:
  type: epsilon_greedy
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 500000
```

Advanced exploration options:

- Intrinsic reward from JEPA prediction error.
- Random network distillation.
- Episodic novelty bonus.
- Meta-controller over exploration policies.

For version 1, implement epsilon-greedy and optional JEPA prediction-error intrinsic reward:

```text
reward_total = reward_env + beta_intrinsic * prediction_error
```

---

## 9. Training Strategy

### 9.1 Training Phases

#### Phase 1: Random Data Collection

Collect initial replay data using random or scripted actions.

Purpose:

- Populate replay buffer.
- Bootstrap JEPA representation.
- Validate score extraction and reset logic.

#### Phase 2: JEPA Pretraining

Train world model on collected trajectories.

Purpose:

- Learn visual dynamics.
- Stabilize policy learning.
- Reduce sample complexity.

#### Phase 3: Joint Online Training

Alternate between:

- Actor rollout.
- Replay insertion.
- JEPA updates.
- RL updates.
- Evaluation episodes.

#### Phase 4: High-Score Fine-Tuning

Once the agent achieves non-trivial play:

- Reduce exploration.
- Increase exploitation.
- Maintain small novelty bonus to avoid local optima.
- Save top-k policies by evaluation score.

### 9.2 Training Loop

```python
for global_step in range(total_steps):
    obs = env.observe()
    z = world_model.encode(obs)
    action = agent.act(z, exploration=True)

    next_obs, reward, done, info = env.step(action)
    replay.add(obs, action, reward, next_obs, done, info)

    if global_step > learning_starts:
        for _ in range(world_updates_per_step):
            batch = replay.sample_sequence(batch_size, sequence_length)
            world_loss = train_jepa(world_model, batch)

        for _ in range(policy_updates_per_step):
            batch = replay.sample_transition(batch_size)
            latent_batch = world_model.encode_batch(batch)
            rl_loss = train_agent(agent, latent_batch)

    if done:
        obs = env.reset()

    if global_step % eval_interval == 0:
        evaluate_and_checkpoint()
```

### 9.3 Joint Training Ratio

Default:

```yaml
training:
  total_env_steps: 5000000
  learning_starts: 10000
  world_updates_per_env_step: 1
  policy_updates_per_env_step: 1
  eval_interval_steps: 50000
  checkpoint_interval_steps: 50000
```

For compute-constrained training:

```yaml
training:
  world_updates_per_env_step: 0.25
  policy_updates_per_env_step: 1
```

For representation-heavy research:

```yaml
training:
  world_updates_per_env_step: 2
  policy_updates_per_env_step: 1
```

---

## 10. Configuration System

### 10.1 Design Principles

- Every important hyperparameter should be declarative.
- Configs should be composable: base config + game config + experiment override.
- Configs should be logged with every run.
- Invalid config combinations should fail fast.

### 10.2 Top-Level Config

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

observation:
  mode: screenshot
  width: 160
  height: 120
  grayscale: false
  frame_stack: 4
  crop: null
  normalize: true

reward:
  type: score_delta
  score_reader: dom
  score_selector: "#score"
  survival_bonus: 0.0
  idle_penalty: 0.0
  death_penalty: 0.0
  clip_rewards: false

world_model:
  enabled: true
  latent_dim: 512
  encoder:
    type: conv
    hidden_channels: [32, 64, 128, 256]
  predictor:
    type: transformer
    hidden_dim: 512
    depth: 4
    num_heads: 8
    horizons: [1, 2, 4, 8]
  optimizer:
    type: adamw
    lr: 0.0003
    weight_decay: 0.05
    betas: [0.9, 0.95]
  loss:
    prediction: cosine_mse
    lambda_var: 1.0
    lambda_cov: 0.04
  target_encoder:
    ema_tau_start: 0.996
    ema_tau_end: 0.9999

agent:
  algorithm: dqn
  gamma: 0.997
  n_step: 5
  batch_size: 256
  q_network:
    hidden_dims: [512, 512]
    dueling: true
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.01

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

---

## 11. Hyperparameter Groups

### 11.1 Game Parameters

| Parameter | Purpose | Typical Values |
|---|---|---|
| `fps` | Browser/game stepping rate | 15, 30, 60 |
| `action_repeat` | Repeat chosen action for N frames | 1, 2, 4, 8 |
| `max_steps_per_episode` | Prevent infinite episodes | 1k-50k |
| `reset_timeout_sec` | Detect failed reset | 5-30 |

### 11.2 Observation Parameters

| Parameter | Purpose | Typical Values |
|---|---|---|
| `width`, `height` | Model input size | 84x84, 160x120, 224x224 |
| `frame_stack` | Motion information | 1, 4, 8 |
| `grayscale` | Reduce input size | true/false |
| `crop` | Remove browser/UI noise | optional |

### 11.3 JEPA Parameters

| Parameter | Purpose | Typical Values |
|---|---|---|
| `latent_dim` | Representation size | 128, 256, 512, 1024 |
| `horizons` | Future prediction steps | [1,2,4], [1,2,4,8] |
| `ema_tau` | Target encoder smoothing | 0.99-0.9999 |
| `lambda_var` | Collapse prevention | 0.1-2.0 |
| `lambda_cov` | Redundancy reduction | 0.01-0.1 |
| `world_lr` | JEPA learning rate | 1e-4 to 1e-3 |

### 11.4 RL Parameters

| Parameter | Purpose | Typical Values |
|---|---|---|
| `gamma` | Discount factor | 0.99-0.999 |
| `n_step` | Multi-step return length | 1, 3, 5, 10 |
| `batch_size` | RL update batch size | 64-512 |
| `agent_lr` | Policy/Q learning rate | 1e-5 to 3e-4 |
| `epsilon_end` | Final random action rate | 0.01-0.1 |
| `replay_capacity` | Experience memory | 100k-5M |

---

## 12. Model Size Presets

### 12.1 Tiny Preset

Use for local CPU/GPU smoke tests.

```yaml
preset: tiny
observation:
  width: 84
  height: 84
  grayscale: true
world_model:
  latent_dim: 128
  encoder:
    hidden_channels: [16, 32, 64]
  predictor:
    hidden_dim: 128
    depth: 2
    num_heads: 4
agent:
  q_network:
    hidden_dims: [256]
replay:
  capacity: 100000
```

### 12.2 Small Preset

Use for first serious Breakout-like experiments.

```yaml
preset: small
observation:
  width: 160
  height: 120
  grayscale: false
world_model:
  latent_dim: 512
  encoder:
    hidden_channels: [32, 64, 128, 256]
  predictor:
    hidden_dim: 512
    depth: 4
    num_heads: 8
agent:
  q_network:
    hidden_dims: [512, 512]
replay:
  capacity: 1000000
```

### 12.3 Base Preset

Use for multi-game or visually richer games.

```yaml
preset: base
observation:
  width: 224
  height: 224
  grayscale: false
world_model:
  latent_dim: 768
  encoder:
    type: vit
    embed_dim: 512
    depth: 8
    num_heads: 8
  predictor:
    hidden_dim: 768
    depth: 6
    num_heads: 12
agent:
  q_network:
    hidden_dims: [1024, 1024]
replay:
  capacity: 2000000
```

---

## 13. Evaluation

### 13.1 Metrics

Primary:

- Best score achieved.
- Mean score over deterministic evaluation episodes.
- Median score.
- p95 score.
- Time to target score.

Representation metrics:

- JEPA prediction loss by horizon.
- Latent variance and covariance.
- Collapse indicators.
- Prediction error vs. novelty.

RL metrics:

- Episode return.
- TD error.
- Q-value scale.
- Action entropy.
- Exploration rate.
- Replay priority distribution.

Operational metrics:

- Environment steps per second.
- Browser reset failure rate.
- Score reader failure rate.
- GPU utilization.
- Training wall-clock time.

### 13.2 Baselines

Minimum baselines:

1. Random policy.
2. DQN from pixels without JEPA.
3. DQN with frozen JEPA pretrained on random data.
4. DQN with jointly trained JEPA.

Optional baselines:

- PPO from pixels.
- Autoencoder representation + DQN.
- Contrastive representation + DQN.
- Human demonstration warm-start.

---

## 14. Data and Replay

### 14.1 Replay Storage

Replay records should include:

```python
Transition = {
    "obs": uint8_image,
    "action": int_or_vector,
    "reward": float,
    "next_obs": uint8_image,
    "done": bool,
    "score": float,
    "timestamp": int,
    "game_id": str,
    "episode_id": str,
    "metadata": dict,
}
```

For JEPA sequence training, replay must support contiguous sequence sampling:

```python
SequenceBatch = {
    "obs": [B, T, C, H, W],
    "actions": [B, T],
    "rewards": [B, T],
    "dones": [B, T],
}
```

### 14.2 Offline Datasets

Optional sources:

- Random gameplay.
- Scripted gameplay.
- Human demonstration.
- Previous agent checkpoints.
- Synthetic curriculum environments.

---

## 15. Browser Automation

### 15.1 Recommended Stack

Use Playwright with Chromium.

Rationale:

- Reliable browser control.
- Screenshot capture.
- Keyboard and mouse input.
- Headless mode.
- JavaScript injection for optional score/reset helpers.
- Video recording support.

### 15.2 Isolation and Safety

Browser runs should use:

- Dedicated browser profile.
- No personal logged-in sessions.
- Network allowlist if needed.
- Fixed viewport.
- Deterministic random seed where supported.
- No extension access.

### 15.3 Reset Strategy

Reset methods in order of preference:

1. Configured reset key or button.
2. Page reload.
3. JavaScript callback.
4. Full browser context restart.

---

## 16. Implementation Plan

### Milestone 1: Browser Environment MVP

Deliverables:

- Playwright-based browser runner.
- Screenshot observation.
- Discrete keyboard actions.
- Configurable reset.
- Score extraction through DOM selector and OCR fallback.
- Random-policy evaluation.

Success criteria:

- Can run 100 episodes of a Breakout-like game without crashing.
- Produces score logs and replay videos.

### Milestone 2: Replay + Baseline DQN

Deliverables:

- Replay buffer.
- DQN-family agent.
- Training loop.
- Checkpointing.
- Evaluation loop.

Success criteria:

- DQN beats random baseline on at least one simple browser game.

### Milestone 3: JEPA Pretraining

Deliverables:

- Encoder, target encoder, predictor.
- Sequence sampling.
- JEPA loss and collapse monitoring.
- Offline pretraining from random replay.

Success criteria:

- JEPA loss decreases.
- Latent variance remains healthy.
- Frozen JEPA + DQN beats or matches pixel DQN sample efficiency.

### Milestone 4: Joint JEPA + RL Training

Deliverables:

- Alternating world model and policy updates.
- Intrinsic reward from prediction error.
- Hyperparameter sweeps.

Success criteria:

- Joint system improves high score faster than baseline DQN on Breakout-like games.

### Milestone 5: Multi-Game Generalization

Deliverables:

- Multi-game config loader.
- Shared JEPA encoder.
- Game-specific action heads or action abstraction layer.
- Cross-game evaluation.

Success criteria:

- Shared representation improves adaptation to a new simple browser game.

---

## 17. Repository Structure

```text
jepa-browser-rl/
  configs/
    base.yaml
    presets/
      tiny.yaml
      small.yaml
      base.yaml
    games/
      breakout.yaml
      snake.yaml
  src/
    browser/
      playwright_env.py
      score_readers.py
      action_spaces.py
    envs/
      browser_game_env.py
      wrappers.py
    models/
      encoders.py
      jepa.py
      predictors.py
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
      config.py
      logging.py
      checkpointing.py
      video.py
  scripts/
    train.py
    eval.py
    collect_random.py
  tests/
    test_env_reset.py
    test_score_reader.py
    test_replay.py
    test_jepa_shapes.py
```

---

## 18. Risks and Mitigations

### 18.1 JEPA Collapse

Risk: The representation collapses to constant embeddings.

Mitigations:

- EMA target encoder.
- Stop-gradient target branch.
- Latent normalization.
- Variance/covariance regularization.
- Collapse metrics in logging.

### 18.2 Score Reader Fragility

Risk: OCR or DOM score extraction fails.

Mitigations:

- Prefer DOM selector when possible.
- Validate score reader before training.
- Track score reader confidence.
- Allow user-supplied score parser.
- Save screenshots on score parsing failure.

### 18.3 Browser Timing Instability

Risk: Browser frame rate and action timing are inconsistent.

Mitigations:

- Fixed viewport and frame pacing.
- Action repeat.
- Environment step synchronization.
- Record timing diagnostics.

### 18.4 Sparse Rewards

Risk: Agent fails to discover reward.

Mitigations:

- Random/scripted warm-start.
- Intrinsic reward from JEPA prediction error.
- Human demonstrations.
- Curriculum tasks.
- Exploration schedules.

### 18.5 Overfitting to One Game

Risk: JEPA learns game-specific visual shortcuts.

Mitigations:

- Train on multiple games.
- Domain randomization.
- Visual augmentations that preserve gameplay.
- Held-out game evaluation.

---

## 19. Open Questions

1. Should the first implementation use DQN only, or support PPO from the beginning?
2. Should score extraction be treated as privileged state or only as reward?
3. How much DOM assistance is acceptable for the target use case?
4. Should JEPA pretraining use only random data, or include early policy rollouts?
5. How should continuous mouse control be represented in the JEPA predictor?
6. What browser games should define the initial benchmark suite?
7. Should the world model support latent-space planning in version 1, or only representation learning?

---

## 20. Recommended Initial Experiment

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

The first success milestone should not be “superhuman Breakout.” It should be:

1. Stable browser control.
2. Reliable score extraction.
3. DQN beats random.
4. JEPA representation remains non-collapsed.
5. JEPA+DQN reaches the same score as pixel DQN in fewer environment steps.

---

## 21. Acceptance Criteria for Version 1

Version 1 is complete when:

- A user can configure a browser game through YAML.
- The agent can train from visual observations.
- The system supports configurable model sizes and hyperparameters.
- The system logs scores, losses, videos, and checkpoints.
- Baseline DQN and JEPA+DQN are both available.
- JEPA metrics detect collapse or unstable representation learning.
- At least one Breakout-like game shows JEPA+DQN outperforming random and matching or exceeding pixel DQN sample efficiency.

---

## 22. Future Extensions

- Latent-space model predictive control.
- Multi-task shared JEPA encoder.
- Hierarchical action abstractions.
- Natural-language game instructions.
- Human demonstration imitation learning.
- Automated game adapter generation.
- Browser game benchmark suite.
- Distributed actors for faster data collection.
- Agent57-style exploration policy family.
- Population-based hyperparameter tuning.



---

## 23. Research Update: Recent Papers and Architecture Changes

This section updates the design based on recent JEPA, world-model, and sample-efficient RL papers. The main architectural change is that JEPA should not be treated only as a passive visual encoder. The roadmap should support action-conditioned, policy-conditioned, and eventually value-aware latent prediction.

### 23.1 Papers to Attach to This Design

#### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Design implication: add a two-stage representation strategy. First, pretrain a video JEPA from passive gameplay recordings. Second, post-train an action-conditioned predictor from browser-game trajectories. This maps well to browser games because passive video is cheap to collect and browser interaction can be slower or less stable than offline training.

Recommended doc changes:

- Add a passive gameplay pretraining phase before online RL.
- Keep the JEPA encoder reusable across games.
- Add a future latent-planning path using the action-conditioned predictor.
- Track video-representation metrics separately from policy metrics.

#### ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning

Design implication: support action chunking and optional imitation-learning warm starts. The design should allow the model to predict action sequences and abstract future observation sequences, not only one-step future latents.

Recommended doc changes:

- Add `action_chunk_size` to the predictor configuration.
- Support human gameplay traces as optional offline data.
- Add an imitation pretraining mode for policy warm-starts.
- Track action-sequence prediction accuracy as a diagnostic metric.

#### TD-JEPA: Latent-Predictive Representations for Zero-Shot Reinforcement Learning

Design implication: add a version 2 path for policy-conditioned multi-step latent prediction. This is important because high-score optimization depends on long-term policy dynamics, not merely local visual prediction.

Recommended doc changes:

- Add optional `policy_embedding` conditioning to the predictor.
- Add optional task or goal embeddings.
- Add reward-free offline pretraining support across multiple games.
- Use multi-step prediction horizons as a first-class hyperparameter.
- Evaluate representation transfer to new score functions or game variants.

#### Value-Guided Action Planning with JEPA World Models

Design implication: add a version 3 path for value-shaped latent geometry. The latent space should eventually encode not only what future states look like, but which states are closer to high-value or high-score outcomes.

Recommended doc changes:

- Add optional `value_guidance` to the world model.
- Add a value-shaped loss term.
- Represent high-scoring trajectory segments as goal embeddings.
- Add latent action search for evaluation and high-score fine-tuning.

#### DreamerV3 / Mastering Diverse Control Tasks through World Models

Design implication: keep a strong world-model RL baseline that uses fixed or minimally tuned hyperparameters across many environments. This is directly aligned with the goal of arbitrary browser-game support.

Recommended doc changes:

- Add Dreamer-style latent actor-critic as an optional baseline.
- Report whether one preset works across multiple games without heavy tuning.
- Add imagined-rollout training as a future backend.

#### TD-MPC2: Scalable, Robust World Models for Continuous Control

Design implication: add a future decoder-free latent MPC backend, especially for mouse-heavy games and hybrid discrete/continuous action spaces.

Recommended doc changes:

- Add latent MPC as a future planning mode.
- Include continuous mouse control in the planning roadmap.
- Prefer robust presets over per-game hyperparameter overfitting.

#### EfficientZero: Mastering Atari Games with Limited Data

Design implication: evaluate sample efficiency, not only final high score. Browser-game agents should be compared at fixed environment-step budgets.

Recommended doc changes:

- Add score at 100k, 500k, 1M, and 5M environment steps.
- Add time-to-target-score metrics.
- Compare against pixel DQN and model-based Atari-style baselines.

#### DreamerV2: Mastering Atari with Discrete World Models

Design implication: compare against compact-latent world models that learn behavior from imagined trajectories, especially for Breakout-like games.

Recommended doc changes:

- Add DreamerV2/DreamerV3 as world-model baselines.
- Track whether JEPA+DQN is better than a pure imagined-rollout actor-critic.

---

## 24. Updated Architecture Recommendations

### 24.1 Updated JEPA Roadmap

The JEPA component should evolve across three versions:

#### Version 1: Action-Conditioned JEPA

Version 1 predicts future latent states from current visual context and candidate browser actions.

```text
p_{t+k} = Predictor(z_t, a_t, a_{t+1}, ..., a_{t+k-1}, k)
```

This is the correct MVP for Breakout-like browser games.

#### Version 2: Policy-Conditioned JEPA

Version 2 predicts long-term latent dynamics under policy families.

```text
p_{t+k} = Predictor(z_t, pi_id, a_{t:t+k-1}, goal, k)
```

This is useful for multi-game training, offline reward-free data, and fast adaptation to new games or new scoring rules.

#### Version 3: Value-Guided JEPA Planning

Version 3 shapes latent space so that movement through latent space is useful for high-score planning.

```text
L_world = L_jepa
        + lambda_var * L_variance
        + lambda_cov * L_covariance
        + lambda_action * L_action_chunk
        + lambda_value_shape * L_value_shape
```

For browser games, practical goal embeddings can come from high-scoring replay segments. The planner can then search for action sequences that move the latent state toward high-value trajectory regions.

### 24.2 Updated Predictor Configuration

```yaml
world_model:
  predictor:
    type: transformer
    hidden_dim: 512
    depth: 4
    num_heads: 8
    horizons: [1, 2, 4, 8]
    action_embed_dim: 64
    action_chunk_size: 4
    conditioning:
      action_sequence: true
      policy_embedding: false       # version 2
      task_or_goal_embedding: false # version 2
      value_guidance: false         # version 3
  planning:
    enabled: false
    method: latent_mpc
    horizon: 8
    num_candidates: 64
    objective: predicted_value_plus_score_delta
    use_during_training: false
    use_during_eval: false
```

### 24.3 Updated Training Phases

The training pipeline should now include passive pretraining before random data collection.

#### Phase 0: Passive Gameplay Collection

Collect or import gameplay videos before online training. Sources can include random play, scripted play, human demonstration, and previous agent runs.

Purpose:

- Pretrain visual temporal representations.
- Learn game dynamics before reward optimization.
- Reduce online sample cost.
- Enable multi-game representation learning.

#### Phase 1: Random or Scripted Data Collection

Collect action-labeled trajectories for action-conditioned JEPA and baseline replay.

#### Phase 2: JEPA Pretraining

Train the video/action-conditioned JEPA before policy learning begins.

#### Phase 3: Joint JEPA + RL Training

Alternate environment rollout, replay insertion, JEPA updates, and policy updates.

#### Phase 4: High-Score Fine-Tuning

Reduce exploration and select top-k checkpoints by deterministic evaluation score.

#### Phase 5: Latent Planning Fine-Tuning

Optional later phase. Use latent action search during evaluation or difficult states after the world model is sufficiently accurate.

### 24.4 Updated Training Config Additions

```yaml
training:
  passive_pretrain_steps: 100000
  total_env_steps: 5000000
  learning_starts: 10000
  world_updates_per_env_step: 1
  policy_updates_per_env_step: 1
  planning_start_step: 1000000
  planning_eval_only_until: 2000000
```

### 24.5 Updated Baselines

Minimum baselines should now include:

1. Random policy.
2. Pixel DQN without JEPA.
3. DQN with frozen JEPA pretrained on random data.
4. DQN with frozen JEPA pretrained on passive gameplay video.
5. DQN with jointly trained action-conditioned JEPA.

Optional baselines:

- PPO from pixels.
- Dreamer-style latent actor-critic.
- EfficientZero/MuZero-style planning baseline for Atari-like games.
- TD-MPC-style latent planner for continuous or hybrid-action games.
- Human demonstration warm-start.

### 24.6 Updated Evaluation Metrics

Primary metrics should include:

- Best score achieved.
- Mean, median, and p95 score over deterministic evaluation episodes.
- Time to target score.
- Score after fixed environment-step budgets: 100k, 500k, 1M, and 5M steps.
- Sample efficiency relative to pixel DQN.
- Wall-clock training time.
- Browser reset failure rate.
- Score-reader failure rate.

Representation metrics should include:

- JEPA prediction loss by horizon.
- Latent variance and covariance.
- Collapse indicators.
- Action-sequence prediction accuracy.
- Prediction error on high-score versus low-score trajectories.
- Transfer performance when freezing the encoder and training a new policy head.

Planning metrics should include:

- Planner-improved score versus policy-only score.
- Planning latency per decision.
- Candidate action sequence success rate.
- Value-prediction calibration.

---

## 25. Updated Recommendation

The strongest next version of this project is:

```yaml
initial_research_track:
  env: breakout_like_browser_game
  representation: action_conditioned_jepa
  passive_pretraining: true
  rl_backend: dqn
  baseline_1: pixel_dqn
  baseline_2: frozen_passive_jepa_plus_dqn
  baseline_3: joint_jepa_plus_dqn
  future_backend_1: dreamer_style_actor_critic
  future_backend_2: latent_mpc
```

The first milestone should remain practical: prove that JEPA improves sample efficiency on a Breakout-like browser game. Once that works, the next step should be policy-conditioned JEPA or value-guided latent planning, not a larger encoder by default.


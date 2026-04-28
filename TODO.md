# TODO

This TODO turns [`docs/design_doc.md`](docs/design_doc.md) into an implementation checklist. The priority order is intentionally practical: first make the browser environment reliable, then establish pixel-DQN baselines, then add JEPA, then compare sample efficiency, then evolve JEPA toward policy-conditioned and value-guided variants.

> **Picking up a task?** Read the matching phase in [DEVELOPMENT.md §4](DEVELOPMENT.md) first. It documents prerequisites, parallel-decomposable subtasks, the verification gate, expected artifacts, and which agent type to dispatch.

## Priority Key

- **P0**: Required for the first runnable system.
- **P1**: Required for Version 1 acceptance.
- **P2**: Useful after Version 1 is stable.

## Design References

Phases below cite sections of [`docs/design_doc.md`](docs/design_doc.md) and reference PDFs in [`docs/`](docs/). The reference papers are:

| Paper | File | Phases informed |
|---|---|---|
| V-JEPA 2 | `docs/2506.09985v1.pdf` | 6, 13 (passive video pretraining → action post-training) |
| ACT-JEPA | `docs/2501.14622v4.pdf` | 6 (action chunking), 8 (imitation warm-start) |
| TD-JEPA | `docs/2510.00739v1.pdf` | 16 (V2 policy-conditioned JEPA) |
| Value-Guided JEPA Planning | `docs/2601.00844v1.pdf` | 17 (V3 value-shaped latent geometry) |
| DreamerV3 | `docs/s41586-025-08744-2.pdf` | 14 (Dreamer-style latent actor-critic baseline) |
| TD-MPC2 | `docs/2310.16828v2.pdf` | 14 (latent MPC backend) |

---

## Phase 0: Project Foundation

References: design_doc §17 (Repository Structure).

- [x] P0 Create Python package layout under `src/jepa_rl/`.
- [x] P0 Add `pyproject.toml` with package metadata, console scripts, test config, and lint config.
- [x] P0 Choose and pin the initial dependency set.
  - PyTorch.
  - Playwright.
  - NumPy.
  - Pillow.
  - OpenCV.
  - Pydantic or OmegaConf.
  - PyYAML if not included through the config stack.
  - pytest.
  - ruff.
- [x] P0 Add `.gitignore` for Python, checkpoints, videos, replay stores, logs, browser profiles, and OS metadata.
- [x] P0 Add `configs/`, `scripts/`, `src/jepa_rl/`, and `tests/` directories.
- [x] P0 Add a minimal CI workflow once the project is in git.
- [x] P0 Add a smoke-test command that runs without GPU.
- [x] P1 Add `Makefile` or task runner commands for setup, lint, test, format, and smoke tests.
- [ ] P1 Document supported Python and CUDA versions after first verified install.

Definition of done:

- `python -m pytest` runs.
- `jepa-rl --help` works.
- A developer can install the package in editable mode.

## Phase 1: Configuration System

References: design_doc §10 (Configuration System), §12 (Model Size Presets).

- [x] P0 Define typed config models for experiment, game, observation, actions, reward, world model, agent, replay, exploration, training, and evaluation.
- [x] P0 Implement config loading from YAML.
- [x] P0 Implement base config plus override merging (`base + game + experiment override`).
- [x] P0 Add fail-fast validation for invalid combinations.
- [x] P0 Add config snapshot saving to each run directory.
- [x] P0 Add `configs/base.yaml`.
- [x] P0 Add `configs/presets/tiny.yaml`.
- [x] P0 Add `configs/presets/small.yaml`.
- [x] P1 Add `configs/presets/base.yaml` (ViT, 224x224).
- [x] P0 Add `configs/games/breakout.yaml` for the first target game.
- [ ] P1 Add `configs/games/snake.yaml` after Breakout works.
- [x] P1 Add `jepa-rl validate-config --config ...`.
- [x] P1 Include derived settings in validation, such as input channels from RGB/grayscale and frame stack.
- [x] P1 Validate that `world_model.predictor.conditioning.policy_embedding` and `value_guidance` are off until V2/V3 phases (16, 17).

Definition of done:

- Invalid configs fail before browser launch.
- Merged configs are written to `runs/<experiment>/config.yaml`.
- Tiny preset can be used for CPU smoke tests.

## Phase 2: Browser Environment MVP

References: design_doc §6 (Environment Interface), §15 (Browser Automation).

- [x] P0 Implement `BrowserGameEnv` interface.
  - `reset()`.
  - `step(action)`.
  - `observe()`.
  - `read_score()`.
  - `is_done()`.
  - `render_video_frame()`.
- [x] P0 Implement Playwright Chromium runner.
- [x] P0 Add `jepa-rl open-game` for visible browser launch.
- [x] P0 Use isolated browser contexts and dedicated browser profiles.
- [x] P0 Set fixed viewport from config.
- [x] P0 Implement screenshot observation mode.
- [ ] P1 Implement canvas capture observation mode.
- [ ] P1 Implement DOM-assisted observation mode (screenshot + DOM-extracted score/lives/timer).
- [x] P0 Implement resize, grayscale, and frame stack wrappers.
- [ ] P0 Implement crop and normalization wrappers.
- [x] P0 Implement discrete keyboard action space.
- [x] P0 Implement no-op action.
- [x] P0 Implement key combination actions such as `ArrowLeft+Space`.
- [x] P0 Implement action repeat.
- [x] P0 Implement max steps per episode.
- [x] P0 Implement reset by page reload.
- [ ] P1 Implement reset by configured key or button.
- [ ] P1 Implement reset by optional JavaScript callback (mark privileged).
- [ ] P1 Implement full browser context restart fallback.
- [x] P0 Implement DOM score reader.
- [ ] P1 Implement OCR score reader fallback.
- [ ] P1 Implement visual template score reader for digit sprites or lives icons.
- [x] P0 Implement terminal detection by max steps and configured done selector.
- [ ] P1 Implement visual terminal detection hook.
- [ ] P0 Save screenshots when score reading fails.
- [ ] P0 Log reset failures and score-reader failures.
- [x] P0 Add random policy runner.
- [ ] P0 Add replay video recording for evaluation episodes.

Definition of done:

- A random policy can run 100 Breakout-like episodes without crashing.
- Each episode logs score, return, length, reset status, and score-reader status.
- Representative replay videos are written to the run directory.

## Phase 3: Replay and Data Storage

References: design_doc §14 (Data and Replay).

- [x] P0 Define transition schema.
  - Observation.
  - Action.
  - Reward.
  - Next observation.
  - Done flag.
  - Score.
  - Timestamp.
  - Game ID.
  - Episode ID.
  - Metadata.
- [x] P0 Implement in-memory replay buffer.
- [x] P0 Implement capacity eviction.
- [x] P0 Implement uniform transition sampling.
- [x] P0 Implement contiguous sequence sampling for JEPA.
- [x] P0 Ensure sequences do not cross episode boundaries unless explicitly allowed.
- [ ] P1 Implement prioritized replay (alpha/beta schedule).
- [ ] P1 Implement n-step return support.
- [ ] P1 Implement replay serialization to disk.
- [ ] P1 Implement replay loading for offline pretraining.
- [ ] P1 Add dataset manifest files with config, game, episode counts, and collection policy.
- [ ] P2 Add compressed chunked storage for larger datasets.
- [ ] P2 Add multi-game replay sampling.

Definition of done:

- Replay unit tests cover insertion, eviction, transition sampling, and sequence sampling.
- A random collection run can be reused for JEPA pretraining without launching the browser again.

## Phase 4: Logging, Checkpointing, and Artifacts

References: design_doc §13 (Evaluation), Product Requirements §3.1.

- [x] P0 Create run directory structure.
  - `config.yaml`.
  - `metrics/`.
  - `checkpoints/`.
  - `videos/`.
  - `replay/`.
  - `diagnostics/`.
- [x] P0 Implement scalar metrics logger.
- [x] P0 Implement JSONL event log for episode summaries.
- [x] P0 Implement checkpoint save and load.
- [x] P0 Save latest checkpoint.
- [x] P0 Save best checkpoint by deterministic mean evaluation score.
- [ ] P1 Save top-k checkpoints.
- [x] P0 Save optimizer and scheduler state.
- [ ] P0 Save replay cursor or replay metadata.
- [ ] P1 Add TensorBoard or Weights & Biases adapter.
- [ ] P1 Log browser diagnostics.
- [ ] P1 Log GPU and memory diagnostics.
- [ ] P1 Add run summary report generation.

Definition of done:

- Training can be interrupted and resumed from `latest.pt`.
- Evaluation can load a checkpoint and produce deterministic score summaries.

## Phase 5: Pixel DQN Baseline

References: design_doc §8 (Reinforcement Learning Agent).

- [x] P0 Implement convolutional pixel encoder or reuse the world-model encoder without JEPA loss.
- [x] P0 Implement DQN Q-network.
- [x] P0 Implement dueling Q-network option.
- [x] P0 Implement target network with `target_update_interval`.
- [x] P0 Implement Double DQN target computation.
- [x] P0 Implement epsilon-greedy exploration.
- [x] P0 Implement configurable epsilon schedule.
- [x] P0 Implement n-step returns or leave a validated one-step fallback for the first baseline.
- [x] P0 Implement pixel DQN training loop.
- [x] P0 Implement training/evaluation alternation.
- [x] P0 Implement deterministic evaluation mode.
- [x] P0 Log episode return, score, TD error, loss, Q-value scale, action counts, and epsilon.
- [x] P0 Add a temporary NumPy linear pixel-Q smoke trainer before full DQN lands.
- [x] P0 Add `jepa-rl ml-smoke` to verify learner loss decreases on a controlled task.
- [x] P0 Add replay minibatch updates and a target network to the temporary trainer.
- [ ] P1 Implement prioritized replay integration.
- [ ] P1 Implement reward clipping option.
- [ ] P1 Add distributional DQN head as optional extension.

Definition of done:

- Pixel DQN beats random on one simple browser game.
- Pixel DQN results are reproducible enough to serve as a baseline for JEPA experiments.

## Phase 6: JEPA World Model (Version 1: Action-Conditioned)

References: design_doc §7 (JEPA World Model), §24 (Updated Architecture). Papers: ACT-JEPA, V-JEPA 2.

- [x] P0 Implement convolutional encoder.
- [ ] P1 Implement Vision Transformer encoder option.
- [x] P0 Implement target encoder as EMA copy of online encoder.
- [x] P0 Implement EMA tau schedule (e.g., 0.996 → 0.9999).
- [x] P0 Implement stop-gradient on target branch.
- [x] P0 Implement action embedding for discrete actions.
- [x] P0 Implement transformer predictor.
- [ ] P1 Implement GRU predictor option.
- [x] P0 Implement horizon embeddings.
- [x] P0 Implement multi-horizon prediction (e.g., `[1, 2, 4, 8]`).
- [ ] P0 Implement `action_chunk_size` (ACT-JEPA-style action chunking).
- [x] P0 Implement normalized latent prediction loss (cosine + MSE on normalized embeddings).
- [x] P0 Implement variance regularization (`lambda_var`).
- [x] P0 Implement covariance regularization (`lambda_cov`).
- [x] P0 Implement latent normalization option.
- [x] P0 Implement collapse metrics (variance floor, rank, dimension utilization).
- [x] P0 Implement JEPA shape tests (RGB, grayscale, frame stacks 1/4/8, multiple horizons).
- [x] P0 Implement offline JEPA pretraining loop.
- [x] P0 Log prediction loss by horizon.
- [x] P0 Log latent variance and covariance.
- [ ] P1 Log action-sequence prediction diagnostics (accuracy on next-action prediction from latents).
- [ ] P1 Add image augmentations that preserve gameplay semantics.
- [ ] P2 Add passive video pretraining without action labels (see Phase 13).

Definition of done:

- JEPA loss decreases on random replay.
- Latent variance remains above configured floor across training.
- Shape tests cover RGB, grayscale, different frame stacks, and multiple horizons.

## Phase 7: Frozen JEPA + DQN

References: design_doc §9 (Training Strategy), §13.2 (Baselines).

- [ ] P1 Implement frozen encoder feature extraction.
- [ ] P1 Implement latent replay path or on-the-fly latent encoding path.
- [ ] P1 Train DQN head on frozen JEPA latents.
- [ ] P1 Compare frozen random-data JEPA + DQN against pixel DQN.
- [ ] P1 Compare frozen passive-video JEPA + DQN against pixel DQN once Phase 13 exists.
- [ ] P1 Track sample efficiency at fixed step budgets (100k, 500k, 1M, 5M).
- [ ] P1 Add transfer test with frozen encoder and new policy head after a second game exists.

Definition of done:

- Frozen JEPA + DQN produces a clean baseline curve.
- Metrics answer whether pretraining helps, hurts, or has no effect relative to pixel DQN.

## Phase 8: Joint JEPA + DQN

References: design_doc §9.2 (Training Loop), §8.5 (Exploration).

- [ ] P1 Implement joint training loop.
- [ ] P1 Alternate rollout, replay insertion, JEPA updates, and DQN updates.
- [ ] P1 Make `world_updates_per_env_step` configurable (defaults: 1; compute-constrained: 0.25; representation-heavy: 2).
- [ ] P1 Make `policy_updates_per_env_step` configurable.
- [ ] P1 Implement world-model warmup before policy learning.
- [ ] P1 Implement optional intrinsic reward from JEPA prediction error.
- [ ] P1 Normalize intrinsic reward.
- [ ] P1 Log environment reward and intrinsic reward separately.
- [ ] P1 Support freezing/unfreezing encoder for ablation.
- [ ] P1 Avoid accidental target leakage from future observations into policy action selection.
- [ ] P1 Save joint checkpoints with world model, agent, optimizers, schedules, and replay metadata.
- [ ] P2 Add ACT-JEPA-style imitation pretraining mode for policy warm-starts.

Definition of done:

- Joint JEPA + DQN improves high score faster than random policy.
- Joint JEPA + DQN matches or beats pixel DQN sample efficiency on the initial Breakout-like benchmark.

## Phase 9: Evaluation and Benchmarking

References: design_doc §13 (Evaluation), §24.6 (Updated Evaluation Metrics).

- [x] P0 Implement `jepa-rl eval` for the temporary linear pixel-Q checkpoint format.
- [x] P0 Support deterministic action selection.
- [x] P0 Support configurable number of evaluation episodes.
- [ ] P0 Record videos for best and representative episodes.
- [x] P0 Report best, mean, median, and p95 score.
- [ ] P1 Report score at 100k, 500k, 1M, and 5M environment steps.
- [ ] P1 Report time to target score.
- [ ] P1 Report sample efficiency relative to pixel DQN.
- [ ] P1 Report wall-clock training time.
- [ ] P1 Report browser reset failure rate.
- [ ] P1 Report score-reader failure rate.
- [ ] P1 Add random-policy baseline report.
- [ ] P1 Add pixel-DQN baseline report.
- [ ] P1 Add frozen-JEPA baseline report.
- [ ] P1 Add joint-JEPA baseline report.
- [ ] P1 Report prediction error on high-score versus low-score trajectories.
- [ ] P1 Report transfer performance with frozen encoder + new policy head.
- [ ] P2 Report planner-improved score versus policy-only score (Phase 17).
- [ ] P2 Report planning latency per decision (Phase 17).
- [ ] P2 Report candidate action sequence success rate (Phase 17).
- [ ] P2 Report value-prediction calibration (Phase 17).
- [ ] P2 Add HTML or Markdown experiment report generation.
- [x] P2 Add a lightweight static HTML training dashboard for smoke runs.
- [x] P2 Add a local live training UI with embedded game and Start/Stop/Eval controls.

Definition of done:

- A completed run creates an evaluation summary that can be compared across algorithms without opening raw logs.

## Phase 10: Initial Benchmark Games

References: design_doc §20 (Recommended Initial Experiment).

- [x] P0 Pick the first Breakout-like browser game.
- [x] P0 Confirm license and local/reproducible access path.
- [x] P0 Build `configs/games/breakout.yaml`.
- [x] P0 Validate score reader for Breakout.
- [x] P0 Validate reset logic for Breakout.
- [ ] P0 Measure random-policy baseline for Breakout.
- [ ] P1 Add Snake config.
- [ ] P1 Add Flappy Bird config.
- [ ] P1 Add a simple platformer config.
- [ ] P2 Add clicker or mouse-heavy benchmark.
- [ ] P2 Add multi-game benchmark manifest.

Definition of done:

- Breakout config is stable enough for automated training.
- At least one additional game exists before making multi-game claims.

## Phase 11: Testing Strategy

- [x] P0 Test config validation and default merging.
- [x] P0 Test action parsing and key-combination expansion.
- [ ] P0 Test frame stack shape and reset behavior.
- [ ] P0 Test score readers with saved HTML snippets and screenshots.
- [x] P0 Test replay insertion, eviction, transition sampling, and sequence sampling.
- [x] P0 Test DQN target shape and loss computation.
- [x] P0 Test JEPA encoder and predictor shapes.
- [x] P0 Test EMA target update.
- [x] P0 Test checkpoint save/load round trip.
- [ ] P1 Add browser integration smoke test behind an opt-in marker.
- [x] P1 Add short CPU training smoke test with a fake environment.
- [x] P1 Add deterministic seed test for fake environment training.
- [ ] P1 Add regression tests for score-reader failure handling.
- [ ] P2 Add performance tests for replay sampling throughput.

Definition of done:

- Unit tests run quickly without launching a browser.
- Integration tests can be run explicitly before long training jobs.

## Phase 12: Safety and Isolation

References: design_doc §15.2, §18.

- [ ] P0 Use dedicated browser contexts.
- [ ] P0 Avoid personal browser profiles.
- [ ] P0 Disable extensions.
- [ ] P0 Keep headless mode as default.
- [ ] P0 Make viewport deterministic.
- [ ] P0 Do not mutate game source, score, memory, local storage, or network requests for advantage.
- [ ] P1 Add optional network allowlist.
- [ ] P1 Mark JavaScript score/reset callbacks as privileged in config and logs.
- [ ] P1 Save explicit run metadata when privileged callbacks are enabled.
- [ ] P1 Add documentation for acceptable game adapter behavior.

Definition of done:

- Runs are isolated and reproducible.
- Experiment reports make it clear whether a score reader used privileged DOM or JavaScript assistance.

## Phase 13: Passive Pretraining (Two-Stage JEPA)

References: design_doc §24.3 (Updated Training Phases). Paper: V-JEPA 2 (two-stage video JEPA → action post-training).

- [ ] P1 Define passive video dataset format.
- [ ] P1 Add importer for gameplay videos.
- [ ] P1 Add random gameplay video collection.
- [ ] P1 Add scripted gameplay video collection.
- [ ] P2 Add human demonstration import.
- [ ] P2 Add previous-agent replay import.
- [ ] P1 Train video JEPA without action labels (Stage 1 of V-JEPA-2-style two-stage).
- [ ] P1 Post-train action-conditioned predictor on action-labeled trajectories (Stage 2).
- [ ] P1 Track passive-pretraining metrics separately from policy metrics.
- [ ] P1 Compare passive JEPA + DQN against random-data JEPA + DQN.
- [ ] P2 Reuse passive-pretrained encoder across multiple games (multi-game shared encoder).

Definition of done:

- Passive pretraining produces a reusable encoder checkpoint.
- The passive-pretrained encoder can be loaded by DQN experiments.
- Stage 1 (video) and Stage 2 (action-conditioned) metrics are logged independently.

## Phase 14: Future RL and Planning Backends

References: design_doc §24.5 (Updated Baselines), §22 (Future Extensions). Papers: DreamerV3, TD-MPC2.

- [ ] P2 Implement PPO baseline (actor-critic, GAE, entropy bonus).
- [ ] P2 Add recurrent PPO policy option (GRU/LSTM).
- [ ] P2 Add Dreamer-style latent actor-critic baseline (DreamerV3-aligned).
- [ ] P2 Add latent MPC planning interface (TD-MPC2-aligned, decoder-free).
- [ ] P2 Report whether one preset works across multiple games without heavy tuning (Dreamer-style robust hyperparameters claim).
- [ ] P2 Add candidate action sequence search during evaluation.
- [ ] P2 Add continuous mouse action representation.
- [ ] P2 Add hybrid keyboard/mouse action spaces.

Definition of done:

- Future backends can be compared against JEPA+DQN using the same environment, logging, and evaluation interfaces.

## Phase 15: Version 1 Release Checklist

References: design_doc §21 (Acceptance Criteria for Version 1).

- [ ] P1 User can configure a browser game through YAML.
- [ ] P1 Agent can train from visual observations.
- [ ] P1 Model sizes and hyperparameters are configurable without code changes.
- [ ] P1 Scores, losses, diagnostics, videos, checkpoints, and configs are logged.
- [ ] P1 Random policy baseline is available.
- [ ] P1 Pixel DQN baseline is available.
- [ ] P1 Frozen JEPA + DQN baseline is available.
- [ ] P1 Joint JEPA + DQN is available.
- [ ] P1 JEPA collapse metrics are available.
- [ ] P1 Breakout-like benchmark shows JEPA+DQN outperforming random.
- [ ] P1 Breakout-like benchmark shows JEPA+DQN matching or exceeding pixel-DQN sample efficiency.
- [ ] P1 README quickstart is updated with real commands after implementation.
- [ ] P1 TODO is pruned to reflect completed work and next milestones.

## Phase 16: Version 2 — Policy-Conditioned JEPA

References: design_doc §24.1 (Updated JEPA Roadmap). Paper: TD-JEPA.

```text
p_{t+k} = Predictor(z_t, pi_id, a_{t:t+k-1}, goal, k)
```

- [ ] P2 Add `policy_embedding` conditioning input to predictor.
- [ ] P2 Add task or goal embedding conditioning input.
- [ ] P2 Implement policy ID lookup table or learnable embeddings per checkpoint family.
- [ ] P2 Add reward-free offline pretraining mode across multiple games.
- [ ] P2 Promote multi-step prediction horizons to a first-class hyperparameter.
- [ ] P2 Evaluate representation transfer to new score functions or game variants.
- [ ] P2 Add held-out game evaluation for the shared policy-conditioned encoder.
- [ ] P2 Compare policy-conditioned JEPA + DQN against action-conditioned JEPA + DQN.

Definition of done:

- A single shared encoder works across at least 2 games.
- Switching the policy embedding without retraining the encoder produces measurably different predictions.
- Encoder transfer to a new game with a fresh policy head is faster than training from scratch.

## Phase 17: Version 3 — Value-Guided JEPA Planning

References: design_doc §24.1 (V3), §24.6 (Planning Metrics). Paper: Value-Guided Action Planning with JEPA World Models.

```text
L_world = L_jepa
        + lambda_var * L_variance
        + lambda_cov * L_covariance
        + lambda_action * L_action_chunk
        + lambda_value_shape * L_value_shape
```

- [ ] P2 Add `value_guidance` flag to predictor.
- [ ] P2 Implement value-shaped loss term `L_value_shape`.
- [ ] P2 Build high-scoring trajectory goal embeddings from replay.
- [ ] P2 Implement latent action search (latent MPC) for evaluation.
- [ ] P2 Implement latent action search for high-score fine-tuning at difficult states.
- [ ] P2 Add `planning.enabled`, `planning.method`, `planning.horizon`, `planning.num_candidates` config plumbing.
- [ ] P2 Track planner-improved score versus policy-only score.
- [ ] P2 Track planning latency per decision.
- [ ] P2 Track candidate action sequence success rate.
- [ ] P2 Track value-prediction calibration.

Definition of done:

- Latent geometry is shaped by `L_value_shape` without collapse.
- Planner-improved score is measurably higher than policy-only score on the initial benchmark.
- Planning latency is documented per decision and per evaluation episode.

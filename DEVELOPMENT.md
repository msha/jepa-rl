# DEVELOPMENT.md

Concrete development guide for JEPA-RL. Pairs with [TODO.md](TODO.md) (what to build) and [docs/design_doc.md](docs/design_doc.md) (why and how). This document covers:

1. [Development Principles](#1-development-principles)
2. [Dependency Graph](#2-dependency-graph)
3. [Parallel Development Tracks](#3-parallel-development-tracks)
4. [Per-Phase Development Plan](#4-per-phase-development-plan)
5. [Standard Verification Recipes](#5-standard-verification-recipes)
6. [Agent Dispatch Playbook](#6-agent-dispatch-playbook)
7. [Integration Checkpoints](#7-integration-checkpoints)

---

## 1. Development Principles

### 1.1 Verification Before Completion

Every task has a verification gate. A task is not "done" until:

- An automated check passes (`pytest -k …`, a CLI exit code, a metric crosses a threshold).
- The check is committed to the repo (in `tests/` or `scripts/verify_*.py`).
- The output of the check is reproducible from the run config.

Vibes do not count. "I think it works" does not count. If you cannot write the verification, you cannot mark the task complete.

### 1.2 Interface Contracts Before Implementation

Parallel work only accelerates if components have stable contracts. Before any phase that touches multiple components:

1. **Write the dataclasses / TypedDicts / Pydantic models** for cross-component data first.
2. **Commit fake/stub implementations** that satisfy the type signatures (return zeros, raise NotImplementedError).
3. **Write the integration test** that exercises the contract with stubs.
4. **Then dispatch parallel agents** to fill in the real implementations against the locked contract.

This mirrors the JEPA architecture itself: the predictor, encoder, and policy all communicate through `z_t` (a tensor of shape `[B, latent_dim]`). Once that contract is fixed, three teams can work independently.

### 1.3 Smallest Slice First

Each phase has P0 (required for the runnable system), P1 (required for V1 acceptance), and P2 (post-V1 nice-to-haves). Land **all P0 of a phase** before starting P1, and **all P0/P1 of upstream phases** before starting downstream phases. This avoids the trap of three half-finished phases that cannot be exercised end-to-end.

### 1.4 No Privileged Shortcuts

Re-stated from CLAUDE.md because it is the most common temptation during debugging: do not fix a flaky score reader by injecting JS to mutate the DOM, do not skip browser-launch determinism by spinning a private viewport, do not pretend the env is solved by clipping rewards in code without making it a config. Document the failure, fix the root cause, or mark the workaround `privileged: true` in run metadata.

### 1.5 Reproducibility From Config

A run is reproducible from `runs/<experiment>/config.yaml` plus the code revision. Anything that affects the run must come through config. Anything that affects results must be logged. If you find yourself adding a hardcoded constant, ask whether it belongs in the config schema instead.

---

## 2. Dependency Graph

Phase numbers refer to [TODO.md](TODO.md). Arrows mean "blocks the start of."

```text
                        Phase 0 (Foundation)
                              │
                              ▼
                        Phase 1 (Configs)
                       /      │       \
                      ▼       ▼        ▼
              Phase 2     Phase 3    Phase 4    Phase 11 (Tests scaffolding)
            (Browser)    (Replay)  (Logging)         │
                 │           │        │              │
                 │           └────────┼──────────────┘
                 │                    │
                 └─────────┬──────────┘
                           ▼
                   Phase 5 (Pixel DQN)
                           │
                           ├──────► Phase 10 (Breakout config + baseline)
                           ▼
                   Phase 6 (JEPA World Model)
                           │
                           ├──────► Phase 13 (Passive Pretraining)
                           ▼
                   Phase 7 (Frozen JEPA + DQN)
                           │
                           ▼
                   Phase 8 (Joint JEPA + DQN)
                           │
                           ▼
                   Phase 9 (Evaluation Reports)
                           │
                           ▼
                   Phase 15 (V1 Release)
                           │
                           ├──────► Phase 16 (V2 Policy-Conditioned JEPA)
                           ├──────► Phase 17 (V3 Value-Guided Planning)
                           └──────► Phase 14 (Future RL/Planning Backends)
```

Cross-cutting:
- **Phase 11 (Testing)** runs alongside every other phase. Each phase ships its own tests; Phase 11 owns the harness, fixtures, fakes, and CI integration.
- **Phase 12 (Safety)** runs alongside Phase 2 and Phase 10. Treat it as gates on those phases, not a separate sprint.

---

## 3. Parallel Development Tracks

Once Phase 0 + Phase 1 land, the project splits into **five independent tracks** that converge at Phase 5 and Phase 6.

### Track A — Browser & Environment

Phases 2, 10, 12. Owner: one or two engineers. Outputs `BrowserGameEnv`, observation wrappers, score readers, action spaces, the random-policy runner, and the first game configs.

**Contract** with the rest of the system: produces `Observation` (uint8 image tensor of shape `[C, H, W]` after wrappers), accepts `Action` (int for discrete spaces), exposes `read_score() -> float`. Once this contract is locked in Phase 1, Track A can iterate without touching any model code.

### Track B — Replay & Data

Phases 3, 13. Outputs the replay buffer, transition/sequence samplers, dataset manifests, optional disk serialization, and the passive-video importer.

**Contract**: takes `Transition` dicts (see design_doc §14.1), returns `TransitionBatch` and `SequenceBatch` of fixed shape. The buffer never sees model code; the model never sees the buffer's storage layout.

### Track C — Models

Phase 5 Q-networks, Phase 6 JEPA encoder/predictor, Phase 13 video encoder. This track itself has internal parallelism — see §4.6 — because the conv encoder, ViT encoder, transformer predictor, GRU predictor, EMA target wrapper, and collapse-metric module are independent components.

**Contract**: pure functions of tensors. Encoder maps `[B, C, H, W]` → `[B, latent_dim]`. Predictor maps `(z_t [B, D], actions [B, K], horizon int)` → `p_{t+k} [B, D]`. Q-network maps `z_t [B, D]` → `q [B, num_actions]`. Lock these signatures in a `models/contracts.py` first, then parallelize.

### Track D — Training Infrastructure

Phase 4 (logging, checkpointing), Phase 9 scaffolding (eval CLI, report generation). Owns `runs/<experiment>/` directory layout, scalar metrics writer, JSONL event log, checkpoint save/load round-trip, deterministic eval driver.

**Contract**: every other track calls `logger.scalar(name, value, step)`, `checkpoint.save(state, path)`, `checkpoint.load(path) -> state`. No track owns its own logging format.

### Track E — Configs & Validation

Phase 1. Owns `configs/base.yaml`, presets, game configs, the typed config models, and `jepa-rl validate-config`. This track moves fastest because it has no runtime dependencies — it only needs the schema agreed on with Tracks A–D.

**Contract**: every other track receives a frozen, validated config object. No track parses YAML directly.

---

## 4. Per-Phase Development Plan

Each phase block includes: **Prerequisites**, **Parallel decomposition** (what can be split between agents), **Verification gate** (the single check that proves the phase is done), and **Expected artifacts**.

### 4.1 Phase 0 — Project Foundation

- **Prerequisites**: None. This phase initializes everything.
- **Parallel decomposition**:
  - Agent 1 (deployment-engineer): `pyproject.toml`, console-script entry point, dependency pinning, editable install.
  - Agent 2 (devops-engineer): `.gitignore`, basic CI workflow stub, smoke-test command.
  - Agent 3 (deployment-engineer): `Makefile` or task runner with setup/lint/test/format targets.
- **Verification gate**:
  ```bash
  uv sync                          # creates .venv, installs from uv.lock
  uv run jepa-rl --help            # exits 0
  uv run pytest                    # exits 0 (zero tests acceptable at the very start of Phase 0)
  uv run ruff check src tests      # exits 0
  ```
- **Expected artifacts**: `pyproject.toml` (with `[dependency-groups].dev`, `[tool.uv]`, runtime extras), `uv.lock` (committed), `.gitignore`, `Makefile` wrapping `uv` targets, empty `src/jepa_rl/__init__.py`, empty `tests/__init__.py`, `.github/workflows/ci.yml` using `astral-sh/setup-uv`.

### 4.2 Phase 1 — Configuration System

- **Prerequisites**: Phase 0.
- **Parallel decomposition**:
  - Agent 1 (typescript-pro / ai-engineer): typed config models for `experiment`, `game`, `observation`, `actions`.
  - Agent 2 (ai-engineer): typed config models for `world_model`, `agent`, `replay`, `exploration`.
  - Agent 3 (backend-developer): config loader, base+override merging, snapshot writer, `jepa-rl validate-config` CLI.
  - Agent 4 (ai-engineer): preset YAMLs (tiny, small, base) and `configs/games/breakout.yaml` skeleton.
- **Verification gate**:
  ```bash
  uv run jepa-rl validate-config --config configs/games/breakout.yaml      # exits 0
  uv run jepa-rl validate-config --config tests/fixtures/invalid_*.yaml    # exits non-zero with specific error message
  uv run pytest tests/test_config.py                                       # passes
  ```
  Plus: a snapshot test that loads `base + presets/small + games/breakout` and diffs the merged result against a golden fixture.
- **Expected artifacts**: `src/jepa_rl/utils/config.py`, `configs/base.yaml`, `configs/presets/{tiny,small,base}.yaml`, `configs/games/breakout.yaml`, `tests/test_config.py`, `tests/fixtures/configs/*.yaml`.

### 4.3 Phase 2 — Browser Environment MVP

- **Prerequisites**: Phase 0, Phase 1 (needs config schema for `game`, `observation`, `actions`, `reward`).
- **Parallel decomposition**:
  - Agent 1 (backend-developer): `BrowserGameEnv` Playwright runner, viewport, isolated context, screenshot capture.
  - Agent 2 (backend-developer): action space module — discrete keyboard, key combinations, action repeat.
  - Agent 3 (backend-developer): score readers — DOM selector primary, OCR fallback (use `pytesseract` or similar; pin in Phase 0).
  - Agent 4 (backend-developer): observation wrappers — resize, crop, normalize, grayscale, frame stack.
  - Agent 5 (backend-developer): random policy runner + replay video recorder.
- **Verification gate**: the **100-episode random run** must complete without crashing on a synthetic local game (a static HTML+JS Breakout served over `python -m http.server`):
  ```bash
  uv run jepa-rl collect-random --config configs/games/breakout_local.yaml --episodes 100
  # success: 100 episode_summary.jsonl entries, 0 unhandled exceptions, score-reader failure rate < 5%
  uv run python scripts/verify_random_run.py runs/breakout_random_v1
  ```
  `verify_random_run.py` asserts: 100 episodes, every episode has score+length+reset_status, reset failures ≤ threshold, at least one replay video written.
- **Expected artifacts**: `src/jepa_rl/browser/{playwright_env,score_readers,action_spaces}.py`, `src/jepa_rl/envs/{browser_game_env,wrappers}.py`, `scripts/collect_random.py`, `scripts/verify_random_run.py`, fixtures: a static `tests/fixtures/breakout_local/` HTML+JS Breakout for offline tests.

### 4.4 Phase 3 — Replay and Data Storage

- **Prerequisites**: Phase 0, Phase 1.
- **Parallel decomposition**:
  - Agent 1 (data-engineer): in-memory replay buffer with capacity eviction and uniform sampling.
  - Agent 2 (data-engineer): contiguous sequence sampler with episode-boundary safety.
  - Agent 3 (data-engineer): prioritized replay (alpha/beta schedule).
  - Agent 4 (data-engineer): n-step return computation.
  - Agent 5 (data-engineer): disk serialization + dataset manifest.
- **Verification gate**:
  ```bash
  uv run pytest tests/test_replay.py
  ```
  Tests must cover: insertion, eviction at capacity, uniform sampling distribution (chi-square test on small buffer), sequence sampling never crosses an episode boundary unless explicitly allowed, prioritized replay updates priorities correctly, n-step returns match a hand-computed reference for a known reward sequence.
- **Expected artifacts**: `src/jepa_rl/replay/{replay_buffer,sequence_sampler}.py`, `tests/test_replay.py`.

### 4.5 Phase 4 — Logging, Checkpointing, Artifacts

- **Prerequisites**: Phase 0.
- **Parallel decomposition**:
  - Agent 1 (devops-engineer): run directory layout, scalar metrics logger, JSONL event log.
  - Agent 2 (deployment-engineer): checkpoint save/load round-trip with optimizer + scheduler state.
  - Agent 3 (devops-engineer): TensorBoard / W&B adapter (behind a flag).
  - Agent 4 (devops-engineer): run summary report generator (Markdown).
- **Verification gate**: the **interrupt-and-resume** test:
  ```python
  # tests/test_checkpoint_resume.py
  state_a = train_for_n_steps(config, n=100)
  ckpt = save_checkpoint(state_a)
  state_b = load_checkpoint(ckpt)
  state_c = train_for_n_steps_from(state_b, n=100)
  state_d = train_for_n_steps(config, n=200)
  assert tensors_equal(state_c, state_d, atol=1e-6)  # bitwise close
  ```
  Plus: `uv run pytest tests/test_logging.py` covers metric writes, JSONL events, run summary structure.
- **Expected artifacts**: `src/jepa_rl/utils/{logging,checkpointing,video}.py`, `tests/test_checkpoint_resume.py`, `tests/test_logging.py`.

### 4.6 Phase 5 — Pixel DQN Baseline

- **Prerequisites**: Phases 0–4.
- **Parallel decomposition**:
  - Agent 1 (ai-engineer): convolutional pixel encoder (or shared encoder shell).
  - Agent 2 (ai-engineer): Q-network with optional dueling head, target network, Double DQN target computation.
  - Agent 3 (ai-engineer): epsilon-greedy controller with configurable schedule.
  - Agent 4 (ai-engineer): training loop, train/eval alternation, deterministic eval mode.
  - Agent 5 (ai-engineer): metrics — TD error, Q-value scale, action distribution, loss.
- **Verification gates** (in order):
  1. **Shape test**: `uv run pytest tests/test_dqn_shapes.py` — encoder, Q-network, and loss compute the right shapes for every preset.
  2. **CartPole-fake smoke test**: `uv run pytest tests/test_dqn_smoke.py` — replace the browser env with a deterministic fake env (sequence of generated frames + scripted reward); pixel DQN must reach `score > random_baseline + 3*random_baseline_std` within 50k steps. This is fast (< 5 min on CPU) and catches most regressions.
  3. **Real Breakout sanity**: a one-time gated run using `configs/games/breakout_local.yaml` that produces a learning curve where DQN clearly beats random by 1M env steps. The artifact is `runs/breakout_dqn_v1/eval_summary.md`.
- **Expected artifacts**: `src/jepa_rl/models/{encoders,q_networks}.py`, `src/jepa_rl/training/train_dqn.py`, `tests/test_dqn_shapes.py`, `tests/test_dqn_smoke.py`, `tests/fakes/scripted_env.py`.

### 4.7 Phase 6 — JEPA World Model (V1: Action-Conditioned)

This is the highest-parallelism phase in V1. Lock the contracts in `src/jepa_rl/models/contracts.py` first, then dispatch.

- **Prerequisites**: Phase 0, 1, 3 (sequence sampler), 4 (logging).
- **Parallel decomposition** (all independent once contracts are locked):
  - Agent 1 (ai-engineer): convolutional encoder. Verifiable in isolation via shape + parameter count test.
  - Agent 2 (ai-engineer): ViT encoder option (P1).
  - Agent 3 (ai-engineer): EMA target encoder wrapper (clones a module, applies tau-schedule update, asserts stop-gradient).
  - Agent 4 (ai-engineer): action embedding table for discrete actions.
  - Agent 5 (ai-engineer): transformer predictor with horizon embedding and `action_chunk_size`.
  - Agent 6 (ai-engineer): GRU predictor option (P1).
  - Agent 7 (ai-engineer): JEPA loss — normalized prediction loss, variance regularization, covariance regularization.
  - Agent 8 (ai-engineer): collapse metrics module — per-dim variance, off-diagonal covariance norm, effective rank.
  - Agent 9 (ai-engineer): offline JEPA pretraining loop tying everything together.
- **Verification gates** (in order):
  1. **Shape battery**: `uv run pytest tests/test_jepa_shapes.py` — for each combination of `{RGB, grayscale} × {frame_stack 1, 4, 8} × {horizons [1], [1,2,4], [1,2,4,8]}`, encoder + predictor produce the documented shapes. ~100 parameterized cases.
  2. **EMA correctness**: `uv run pytest tests/test_ema.py` — after `update(tau=0.99)`, target params equal `0.99 * old_target + 0.01 * online`. Stop-gradient is enforced (no grad flows into target).
  3. **Loss decreases on random replay**: `uv run pytest tests/test_jepa_smoke.py` — train JEPA for 5k steps on a recorded replay fixture; final prediction loss is at least 30% below initial.
  4. **No collapse**: same smoke test asserts per-dim variance > `variance_floor` at the end of training, and effective rank > 0.5 * `latent_dim`.
- **Expected artifacts**: `src/jepa_rl/models/{contracts,encoders,jepa,predictors}.py`, `src/jepa_rl/training/train_world.py`, `tests/test_jepa_shapes.py`, `tests/test_ema.py`, `tests/test_jepa_smoke.py`, `tests/fixtures/replay/random_breakout.npz`.

### 4.8 Phase 7 — Frozen JEPA + DQN

- **Prerequisites**: Phase 5 (pixel DQN), Phase 6 (JEPA pretraining produces a checkpoint).
- **Parallel decomposition**:
  - Agent 1 (ai-engineer): frozen-encoder wrapper that exposes `encode(obs) -> z` with no grad.
  - Agent 2 (ai-engineer): adapt DQN training loop to operate on latents instead of pixels (latent replay path or on-the-fly encoding path; choose one and document).
  - Agent 3 (ai-engineer): comparison harness — runs pixel DQN, frozen-random-JEPA + DQN, and (later) frozen-passive-JEPA + DQN on the same env with the same seeds and same eval cadence.
- **Verification gate**: the **sample-efficiency comparison plot**. Run all three configurations to 1M env steps with three seeds each. Produce `runs/comparison_phase7/sample_efficiency.png` and `summary.md`. Pass if frozen-JEPA + DQN curve is within ±10% of pixel DQN at every checkpoint or strictly above. Fail loudly if frozen-JEPA + DQN is below pixel DQN by more than 10% at the 1M step mark — that means the encoder is hurting more than helping.
- **Expected artifacts**: `src/jepa_rl/models/frozen_encoder.py`, `scripts/compare_baselines.py`, `runs/comparison_phase7/`.

### 4.9 Phase 8 — Joint JEPA + DQN

- **Prerequisites**: Phase 7.
- **Parallel decomposition**:
  - Agent 1 (ai-engineer): joint training loop — alternate rollout, replay insertion, JEPA updates, DQN updates with configurable ratios.
  - Agent 2 (ai-engineer): world-model warmup before policy learning starts.
  - Agent 3 (ai-engineer): intrinsic reward from JEPA prediction error, with normalization.
  - Agent 4 (ai-engineer): leakage audit — verify the policy never sees a future observation through a side channel (target encoder, replay metadata, etc.).
- **Verification gate**: V1 acceptance criterion — joint JEPA + DQN must match or exceed pixel DQN sample efficiency on the Breakout-like benchmark with three seeds. The pass criterion is **score at 1M env steps ≥ pixel-DQN score at 1M env steps** for at least 2 of 3 seeds. Artifact: `runs/comparison_phase8/sample_efficiency.png` and a Markdown report.
- **Expected artifacts**: `src/jepa_rl/training/train_joint.py`, `runs/comparison_phase8/`, leakage-audit doc in `docs/leakage_audit.md`.

### 4.10 Phase 9 — Evaluation and Benchmarking

- **Prerequisites**: Phase 4 (logging), Phase 5 (DQN running).
- **Parallel decomposition**:
  - Agent 1 (ai-engineer): `jepa-rl eval` CLI with deterministic action selection and configurable episode count.
  - Agent 2 (ai-engineer): score budget metrics (best/mean/median/p95, score at 100k/500k/1M/5M steps, time-to-target).
  - Agent 3 (ai-engineer): operational metrics (env steps/sec, browser failure rates, GPU util).
  - Agent 4 (documentation-engineer): Markdown / HTML report generator that diffs runs.
- **Verification gate**: a single command produces a comparable report:
  ```bash
  uv run jepa-rl eval --checkpoint runs/breakout_jepa_dqn_v1/best.pt --episodes 50
  cat runs/breakout_jepa_dqn_v1/eval_summary.md     # contains all primary metrics
  uv run jepa-rl report --runs runs/breakout_*_v1   # produces a comparison table
  ```
  Tested via golden-output snapshot of the report on a fixture run.
- **Expected artifacts**: `src/jepa_rl/training/evaluate.py`, `scripts/eval.py`, `scripts/report.py`, `tests/test_eval_report.py`.

### 4.11 Phase 10 — Initial Benchmark Games

- **Prerequisites**: Phase 2.
- **Parallel decomposition**:
  - Agent 1: Breakout (P0) — config + score-reader validation + reset validation.
  - Agent 2: Snake (P1).
  - Agent 3: Flappy Bird (P1).
  - Agent 4: a simple platformer (P1).
- **Verification gate per game**: 100-episode random run completes without crashing, score-reader failure rate < 5%, reset failure rate < 1%, replay video plays back. Captured in `runs/<game>_random_v1/health_report.md`.

### 4.12 Phase 11 — Testing Strategy

This phase is **continuous** — every other phase ships its own tests. Phase 11 owns the harness:

- **Parallel decomposition**:
  - Agent 1: pytest configuration, markers (`@pytest.mark.browser`, `@pytest.mark.slow`, `@pytest.mark.gpu`), fixtures.
  - Agent 2: scripted-env fake for fast CPU smoke training (used by Phases 5, 6, 8).
  - Agent 3: HTML-fixture score reader tests with saved snippets.
  - Agent 4: replay sampling throughput benchmarks (P2).
- **Verification gate**: `uv run pytest -m 'not slow and not browser and not gpu'` runs in under 30 seconds. `uv run pytest -m browser` runs the browser integration smoke test (opt-in).

### 4.13 Phase 12 — Safety and Isolation

This phase is **gates on Phase 2 and 10**, not separate work. Verification:

```bash
uv run pytest tests/test_safety.py
```

Tests assert: dedicated profile is used (path under tmpdir), extensions disabled, fixed viewport, no JS callback unless `privileged: true` in config (which propagates to `runs/<exp>/run_metadata.json` with a top-level `privileged_score_reader: true`).

### 4.14 Phase 13 — Passive Pretraining (Two-Stage JEPA)

- **Prerequisites**: Phase 6.
- **Parallel decomposition**:
  - Agent 1: passive video dataset format + importer.
  - Agent 2: random gameplay video collection script.
  - Agent 3: scripted gameplay video collection.
  - Agent 4: video JEPA Stage 1 (no action labels).
  - Agent 5: action-conditioned post-training Stage 2.
- **Verification gate**:
  ```bash
  uv run jepa-rl train-world --config configs/passive_pretrain.yaml --stage video    # Stage 1
  uv run jepa-rl train-world --config configs/passive_pretrain.yaml --stage action   # Stage 2
  uv run jepa-rl eval-encoder --checkpoint runs/passive_pretrain/encoder.pt          # latent-quality probe
  ```
  Stage 1 succeeds when: prediction loss decreases, no collapse, masked-prediction probe passes.
  Stage 2 succeeds when: action-sequence prediction accuracy > 50% on held-out trajectories (chance-baseline depends on action space size).

### 4.15 Phase 14 — Future RL and Planning Backends (P2)

Independent baselines, all parallel:

- Agent 1: PPO baseline with optional recurrent policy.
- Agent 2: DreamerV3-style latent actor-critic (paper: `docs/s41586-025-08744-2.pdf`).
- Agent 3: TD-MPC2-style latent MPC (paper: `docs/2310.16828v2.pdf`).

Each owns its own training loop; they share Phase 4 logging and Phase 9 eval interfaces.

### 4.16 Phase 15 — V1 Release Checklist

Final integration. Run all baselines on Breakout-like, generate the comparison report, prune TODO.md, update README quickstart with real (now-implemented) commands.

### 4.17 Phase 16 — V2 Policy-Conditioned JEPA (Post-V1)

Single ai-engineer track. Verification: a single shared encoder works across ≥ 2 games; switching the policy embedding measurably changes predictions; transfer to a held-out game with a fresh policy head is faster than from scratch.

### 4.18 Phase 17 — V3 Value-Guided JEPA Planning (Post-V1)

Single ai-engineer track. Verification: planner-improved score is measurably higher than policy-only score on the Breakout-like benchmark; planning latency per decision is documented; latent geometry remains non-collapsed under `L_value_shape`.

---

## 5. Standard Verification Recipes

These patterns recur across phases. Implement once in `tests/recipes/` and reuse.

### 5.1 Shape Battery

For any model component:

```python
@pytest.mark.parametrize("config", PRESET_CONFIGS)
@pytest.mark.parametrize("frame_stack", [1, 4, 8])
@pytest.mark.parametrize("grayscale", [True, False])
def test_encoder_shape(config, frame_stack, grayscale):
    model = build_encoder(config, frame_stack=frame_stack, grayscale=grayscale)
    obs = torch.zeros(2, channels(grayscale, frame_stack), config.height, config.width)
    z = model(obs)
    assert z.shape == (2, config.latent_dim)
```

### 5.2 Collapse Detection

For any JEPA training run:

```python
def assert_not_collapsed(latents: Tensor, variance_floor: float, latent_dim: int):
    per_dim_var = latents.var(dim=0)
    assert per_dim_var.min() > variance_floor, "variance collapse"
    cov = torch.cov(latents.T)
    eig = torch.linalg.eigvalsh(cov)
    effective_rank = (eig.sum() ** 2) / (eig ** 2).sum()
    assert effective_rank > 0.5 * latent_dim, "rank collapse"
```

Call this at the end of every JEPA training test and as a periodic check during real runs.

### 5.3 EMA Correctness

```python
def test_ema_update():
    online = nn.Linear(8, 8)
    target = build_ema_target(online, tau=0.99)
    online.weight.data.fill_(1.0)
    target.weight.data.fill_(0.0)
    update_ema(target, online, tau=0.99)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.01))
    # stop-gradient: target weights have requires_grad=False
    assert not target.weight.requires_grad
```

### 5.4 Sample-Efficiency Comparison

Common harness for comparing learning curves across configurations:

```python
def compare_at_budgets(
    configs: list[Config],
    seeds: list[int],
    budgets: list[int] = [100_000, 500_000, 1_000_000, 5_000_000],
) -> ComparisonReport:
    results = run_all(configs, seeds)
    return ComparisonReport.from_results(results, budgets)
```

Phase 7, 8, 13, and 16 all use this. Implement once.

### 5.5 Deterministic Eval

```python
def deterministic_eval(checkpoint: Path, episodes: int = 50) -> EvalSummary:
    agent = load_checkpoint(checkpoint)
    agent.set_deterministic(True)  # epsilon = 0, argmax actions
    scores = [run_episode(agent, seed=i) for i in range(episodes)]
    return EvalSummary(best=max(scores), mean=mean(scores), median=median(scores), p95=quantile(scores, 0.95))
```

### 5.6 Browser Integration Smoke

Opt-in via `uv run pytest -m browser`. Hits a static `tests/fixtures/breakout_local/` HTML+JS Breakout served by `python -m http.server`. Asserts: 10 episodes complete, score ≥ 0 in all of them, score-reader never throws, reset always succeeds.

### 5.7 Leakage Audit (Phase 8)

```python
def test_no_future_leakage_in_action_selection():
    """The policy must select actions using only z_t, never z_{t+k} or future targets."""
    # Trace a forward pass with hooks; assert no module called select_action() received a target_encoder output.
```

---

## 6. Agent Dispatch Playbook

When the user (or a planning step) authorizes parallel agent dispatch, use these templates. Each template is **self-contained** — the agent gets no implicit context from the calling conversation.

### 6.1 Component Implementation (e.g., conv encoder)

```
subagent_type: ai-engineer
description: Implement convolutional encoder
prompt: |
  You are implementing src/jepa_rl/models/encoders.py::ConvEncoder for the JEPA-RL project.

  Read first:
    - docs/design_doc.md §7.2 (Encoders) — convolutional default config
    - src/jepa_rl/models/contracts.py — locked interface for encoders
    - tests/test_jepa_shapes.py — the shape battery this must pass

  Specification:
    - Input: tensor [B, C, H, W] where C = (1 if grayscale else 3) * frame_stack
    - Output: tensor [B, latent_dim]
    - Hyperparameters from world_model.encoder config: hidden_channels, kernel_sizes, strides, activation, norm, latent_dim
    - Use GELU + LayerNorm by default

  Verification: tests/test_jepa_shapes.py must pass for the conv encoder cases.

  Out of scope: ViT encoder (separate task), JEPA loss (separate task), training loop.

  Report: file paths touched, test command run, test output (last 10 lines).
```

### 6.2 Architecture Review Before Multi-Component Work

```
subagent_type: feature-dev:code-architect
description: Design JEPA component layout before parallel impl
prompt: |
  Design the file and class layout for src/jepa_rl/models/jepa.py and predictors.py for JEPA-RL Phase 6.

  Read: docs/design_doc.md §7, README.md "JEPA Roadmap", TODO.md Phase 6.

  Goal: produce a contracts file (models/contracts.py) and a skeleton (NotImplementedError stubs) so that 9 parallel agents can implement {conv encoder, ViT encoder, EMA target, action embedding, transformer predictor, GRU predictor, JEPA loss, collapse metrics, training loop} without merge conflicts.

  Deliverable: a markdown plan listing exact file paths, class signatures, type annotations, and which test file proves each piece works. Do not write production code yet.
```

### 6.3 Code Review After Phase Completion

```
subagent_type: feature-dev:code-reviewer
description: Review Phase N completion
prompt: |
  Review the diff for Phase N of JEPA-RL. Authoritative spec:
    - docs/design_doc.md §<section>
    - TODO.md Phase N (specifically the Definition of done bullets)
    - DEVELOPMENT.md §4.<n> (verification gate)

  Files changed: <git diff --name-only ...>

  Check: (1) verification gate command actually runs and passes; (2) the implementation matches the design doc, not the plan-of-the-moment; (3) no privileged shortcuts (DOM mutation, JS eval'd into the page, etc.); (4) configs are declarative — no hardcoded constants that should be in YAML; (5) for any JEPA code, collapse metrics are present.

  Report only high-confidence issues. Be specific: file:line references.
```

### 6.4 Recon Before Touching Unfamiliar Code

```
subagent_type: Explore
description: Map current state of <subsystem>
prompt: |
  Survey the current state of <subsystem> in JEPA-RL. Specifically:
    - What files exist under <path>?
    - What contracts (Pydantic models, TypedDicts, dataclasses) define cross-component data?
    - What tests cover this area?
    - Which TODO.md tasks are visibly complete vs visibly incomplete?

  Thoroughness: medium. Report under 300 words.
```

### 6.5 Parallel Dispatch Pattern

When dispatching multiple component-implementation agents in parallel for one phase, send them in a single message with multiple Agent tool calls. Order does not matter. Wait for all to complete before integrating. After all return, run the phase's verification gate as the integration test.

**Pre-requisites for parallel dispatch:**

- Contracts file is committed (no agent will fight over the interface).
- Stub implementations exist for every component being parallelized (so import-time failures don't cascade).
- Each agent has a dedicated test file or a dedicated parametrized test case (no test-file merge conflicts).
- Each agent gets a worktree (`isolation: "worktree"`) so file-level merges are explicit.

**Anti-patterns:**

- Dispatching parallel agents on phases with shared state (e.g., two agents both editing `train_joint.py`).
- Dispatching parallel agents before the contracts file exists.
- Letting one parallel agent's verification depend on another's output (always test components in isolation first, then run the integration verification gate as a single sequential step).

### 6.6 When to Use Which Agent Type

| Task type | Preferred agent |
|---|---|
| JEPA models, encoders, predictors, training loops | `ai-engineer` |
| Browser adapter, env wrappers, score readers | `backend-developer` |
| Replay buffer, sequence sampler, dataset manifests | `data-engineer` |
| Pydantic schemas, typed config models | `typescript-pro` (cross-applies to Python typing) or `ai-engineer` |
| `pyproject.toml`, CI workflow, Makefile, smoke-test runners | `deployment-engineer` |
| Logging adapters, metric writers, observability | `devops-engineer` |
| Multi-component design before parallel impl | `feature-dev:code-architect` |
| Recon / survey of unfamiliar area | `Explore` |
| Per-phase code review | `feature-dev:code-reviewer` |
| Cross-phase architectural review | `architect-reviewer` |
| Run reports, docstring sweep, README updates | `documentation-engineer` |

---

## 7. Integration Checkpoints

After every phase, before declaring it complete, run the integration checkpoint for that phase:

All commands assume `uv` is installed and the lockfile is committed. Run them from the repo root.

| Phase | Integration checkpoint |
|---|---|
| 0 | `uv sync && uv run jepa-rl --help && uv run pytest` |
| 1 | `uv run jepa-rl validate-config --config configs/games/breakout.yaml && uv run pytest tests/test_config.py` |
| 2 | `uv run jepa-rl collect-random --config configs/games/breakout_local.yaml --episodes 100 && uv run python scripts/verify_random_run.py runs/...` |
| 3 | `uv run pytest tests/test_replay.py` |
| 4 | `uv run pytest tests/test_checkpoint_resume.py tests/test_logging.py` |
| 5 | `uv run pytest tests/test_dqn_*.py` then a 1M-step real Breakout run that beats random |
| 6 | `uv run pytest tests/test_jepa_*.py tests/test_ema.py` then offline JEPA pretraining produces a non-collapsed encoder |
| 7 | `uv run python scripts/compare_baselines.py --configs pixel_dqn,frozen_random_jepa_dqn --seeds 0,1,2` produces a sample-efficiency report within ±10% of pixel DQN |
| 8 | Same comparison report for joint JEPA + DQN; passes when ≥ 2 of 3 seeds match or beat pixel DQN at 1M steps |
| 9 | `uv run jepa-rl eval ... && uv run jepa-rl report ...` produces a Markdown summary with all primary metrics |
| 10 | Per-game `runs/<game>_random_v1/health_report.md` shows < 5% reader failures, < 1% reset failures |
| 11 | `uv run pytest -m 'not slow and not browser and not gpu'` < 30 sec; `uv run pytest -m browser` opt-in passes |
| 12 | `uv run pytest tests/test_safety.py` |
| 13 | Two-stage encoder checkpoint loads cleanly into a Phase 7 frozen-JEPA + DQN run |
| 14 | Each future backend produces an eval report comparable to Phase 9 output |
| 15 | All Phase 1–13 integration checkpoints pass on a clean checkout |
| 16 | Held-out game transfer test passes |
| 17 | Planner-improved score is higher than policy-only score on Breakout-like benchmark |

A phase is **only** complete when its integration checkpoint passes on a fresh checkout in CI. "Works on my machine" is not a checkpoint.

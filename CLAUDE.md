# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status: Initial Scaffold

This repository now contains documentation plus an initial installable Python scaffold:

```
jepa-rl/
  pyproject.toml
  Makefile
  README.md
  TODO.md
  DEVELOPMENT.md
  AGENTS.md
  CLAUDE.md
  configs/
    base.yaml
    presets/
    games/
  docs/
    design_doc.md
    *.pdf            (6 reference papers)
  scripts/
  src/jepa_rl/
  tests/
```

Implemented now:

- uv-managed environment: `uv sync` creates `.venv` and installs from `uv.lock`. Dev tooling (pytest, ruff) is in the `[dependency-groups].dev` table; runtime extras (`config`, `browser`, `train`, `all`) remain under `[project.optional-dependencies]`.
- `jepa-rl validate-config`.
- `jepa-rl init-run`.
- `jepa-rl open-game` to launch the configured game in visible Chromium.
- `jepa-rl ml-smoke` to verify the current linear Q learner reduces synthetic loss.
- `jepa-rl ui` to run the live training dashboard and control panel (accepts `--config`, `--run`, or auto-discovers `configs/base.yaml` with no args). The UI is a Vue 3 + Vite SPA in `ui/` served from the Python server.
- `jepa-rl collect-random` for the local Breakout smoke game.
- `jepa-rl train` for the current NumPy linear pixel-Q smoke model.
- `jepa-rl eval` for the current `.npz` smoke-model checkpoints.
- Typed config validation and starter configs.
- Local Breakout-like HTML game at `games/breakout/index.html`.
- Playwright screenshot environment and DOM score reader.
- Discrete keyboard action parsing.
- In-memory replay buffer and sequence sampling.
- Unit tests for the implemented contracts.

Run any project command via `uv run <cmd>` (e.g. `uv run pytest`, `uv run jepa-rl ...`). The Makefile wraps the common targets.

The current `train` command is not the planned DQN/JEPA stack; it is a small linear Q-learning smoke path that proves browser control, reward reading, metrics, and checkpointing. `jepa-rl train-world` is still a planned stub.

The directory may still not be a git repository. Do not run destructive git commands without first checking `git status`.

## Canonical Documents

Read these before any non-trivial change:

1. **[docs/design_doc.md](docs/design_doc.md)** — the full design contract. 25 numbered sections covering goals, non-goals, environment interface, JEPA model, RL agent, training strategy, configuration, evaluation, risks, V1/V2/V3 roadmap. When the design doc and a paper disagree, the design doc wins for V1; defer V2/V3 details to the doc's §23–25.
2. **[TODO.md](TODO.md)** — the implementation checklist. Each phase cites the relevant design doc section and reference paper. Treat phase numbers as stable references; new tasks belong inside an existing phase or at the end.
3. **[DEVELOPMENT.md](DEVELOPMENT.md)** — the development methodology. Per-phase verification gates, the dependency graph, parallel-track decomposition, agent-dispatch templates, and standard verification recipes. **Use this document when picking up any TODO task** — it tells you the verification command, the parallelizable subtasks, and which agent type fits.
4. **[README.md](README.md)** — the public-facing summary. Keep it consistent with the design doc, not ahead of it.

## JEPA Roadmap (Do Not Mix Versions)

The architecture evolves in three explicit versions. Code should keep them separable via config flags, not branches:

| Version | Predictor signature | Phase in TODO | Reference paper |
|---|---|---|---|
| V1 (MVP) | `Predictor(z_t, a_{t:t+k-1}, k)` | Phase 6 | ACT-JEPA, V-JEPA 2 |
| V2 | `Predictor(z_t, pi_id, a_{t:t+k-1}, goal, k)` | Phase 16 | TD-JEPA |
| V3 | V2 + value-shaped latent loss + latent MPC planning | Phase 17 | Value-Guided JEPA Planning |

V1 is the **only** version targeted for Version 1 acceptance. V2/V3 conditioning hooks should exist in config (`world_model.predictor.conditioning.policy_embedding`, `value_guidance`, etc.) but be off by default and rejected by config validation until their phases land.

## Reference Papers (docs/*.pdf)

| File | Paper | Informs |
|---|---|---|
| `2506.09985v1.pdf` | V-JEPA 2 | Two-stage video pretraining → action post-training (Phase 13) |
| `2501.14622v4.pdf` | ACT-JEPA | Action chunking, imitation warm-start (Phase 6, 8) |
| `2510.00739v1.pdf` | TD-JEPA | V2 policy-conditioned predictor (Phase 16) |
| `2601.00844v1.pdf` | Value-Guided Action Planning with JEPA | V3 value-shaped latent loss, latent MPC (Phase 17) |
| `s41586-025-08744-2.pdf` | DreamerV3 | Robust hyperparameters, latent actor-critic baseline (Phase 14) |
| `2310.16828v2.pdf` | TD-MPC2 | Decoder-free latent MPC, hybrid/continuous actions (Phase 14) |

## Hard Constraints (Non-Goals)

These are intentional. Do not add features that violate them without checking with the user:

- **No mutation of game state for advantage** — score, memory, local storage, network requests are off-limits as adversarial leverage. Score *reading* is fine; score *writing* is not.
- **No privileged JavaScript callbacks unless explicitly configured.** When a config opts into a JS score/reset callback, it must be marked `privileged: true` and that fact must appear in run metadata and evaluation reports.
- **No personal browser sessions.** Dedicated browser profiles only. No logged-in accounts, no extensions, no shared state with the user's real Chromium.
- **No game source code as privileged input.** The agent learns from pixels (and optionally DOM text from the score reader), not from reading the game's JS.

## Configuration-First Discipline

Every important hyperparameter is declarative. When adding a new knob:

1. Add it to the typed config model (Pydantic / OmegaConf).
2. Add a default in `configs/base.yaml`.
3. Add validation that catches invalid combinations.
4. Update the `Configuration Contract` example in [README.md](README.md) if the knob is user-facing.
5. Snapshot it into `runs/<experiment>/config.yaml` automatically (no separate code path).

A run must be reproducible from `config.yaml` + code revision. Never hardcode a value in training code that should live in config.

## JEPA Collapse Is a Real Failure Mode

Any JEPA training code must log:

- Latent variance per dimension (and a per-batch variance floor metric).
- Latent covariance off-diagonal magnitude.
- Effective rank of the latent covariance.
- EMA tau schedule progress.
- Prediction loss broken down by horizon.

If you write a JEPA loss without these diagnostics, you are building a system that can silently regress to constant embeddings. Variance/covariance regularization (`lambda_var`, `lambda_cov`), latent normalization, and the EMA stop-gradient target encoder are all required at the same time — they are not interchangeable.

## Acceptance Criteria for V1

V1 is **not** "superhuman Breakout." It is, in order:

1. Stable browser control across 100 episodes.
2. Reliable score extraction (DOM preferred, OCR fallback).
3. Pixel DQN beats random.
4. JEPA representation remains non-collapsed during training.
5. JEPA+DQN reaches the same score as pixel DQN in **fewer environment steps**.

Sample efficiency is the headline metric, not absolute high score. Evaluation must report scores at fixed budgets (100k, 500k, 1M, 5M env steps).

## Local Development Hardware

This dev machine is **Apple Silicon (macOS, MPS-capable)**. `experiment.device: auto`
resolves to MPS via `models/device.py::resolve_torch_device`. Notes:

- `configs/base.yaml` ships with `experiment.device: cpu` for safe smoke defaults;
  real training runs should override to `device: auto` (or explicitly `mps`).
- Smoke tests (`tests/test_dqn_*.py`, `tests/test_checkpoint_resume.py`) pin
  `device: cpu` for determinism and to avoid the ~20 s MPS kernel-cache warmup.
- MPS gotchas the trainer accounts for:
  - `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` so unsupported ops
    fall back to CPU instead of crashing.
  - fp16/bf16 are unsupported for many ops on MPS — fp32 only for V1.
  - `map_location="cpu"` on every checkpoint load so MPS-saved `.pt` files load on
    CPU CI.
  - `torch.use_deterministic_algorithms(True)` is not honored on MPS; smoke tests
    use statistical margins (`mean + 3*std`), not bit-exact equality.

## UI and CLI Feature Parity

Every training and evaluation feature must be accessible from both the CLI and the `jepa-rl ui` web interface. When adding a new capability:

1. **Wire it in `src/jepa_rl/cli.py`** as a subcommand or flag.
2. **Wire it in `src/jepa_rl/ui/server.py`** — expose it through the existing `/api/train/start`, `/api/eval`, or a new POST endpoint.
3. **Wire it in the Vue frontend** (`ui/src/`) — add API calls in the relevant Pinia store, update or add components in `ui/src/components/`. Run `make ui-build` to rebuild.
4. The UI dispatches training and evaluation based on `config.agent.algorithm`, exactly like the CLI. Adding a new algorithm requires updating both `_training_worker` and the handler in `server.py`, plus the corresponding Vue store/component.

Current UI capabilities (must stay in sync with CLI):

| Feature | CLI | UI |
|---|---|---|
| Start DQN or linear_q training | `jepa-rl train` | Start button, dispatches on `agent.algorithm` |
| Stop training | Ctrl-C | Stop button (signals `stop_event`) |
| Evaluate latest checkpoint | `jepa-rl eval` | Eval button, dispatches on `agent.algorithm`, reads `.pt` or `.npz` |
| View live metrics | dashboard.html | Real-time charts: score, loss, epsilon, td-error |
| View config | `validate-config` | Config panel (toggle) |
| Browse runs | `ls runs/` | Run selector dropdown |
| Switch config | `--config` flag | Config selector dropdown |
| Play game manually | `open-game` | Embedded iframe with ◀ serve ▶ reset controls |

## Recommended Stack (Phase 0 pinned, more arrives per phase)

- uv (environment manager + lockfile)
- Python 3.11+
- PyTorch (in `train` extra; not yet wired)
- Playwright (Chromium) (in `browser` extra; not yet wired)
- NumPy, Pillow, OpenCV (in `train` / `browser` extras)
- Pydantic or OmegaConf for typed config (currently using a hand-rolled validator + PyYAML)
- pytest, ruff (in the `dev` dependency group; mypy may be added later)
- Vue 3 + Vite + Pinia (in `ui/`; build with `make ui-build`, dev with `make ui-dev`)
- TensorBoard / W&B / MLflow for experiment logging (Phase 4)

Common commands:

```bash
uv sync                                          # dev env from uv.lock
uv sync --all-extras                             # plus browser/train/config
uv run jepa-rl <subcommand>                      # CLI
uv run pytest                                    # tests
uv run ruff check src tests                      # lint
uv lock                                          # refresh the lockfile
make ui-build                                    # build Vue frontend to ui/dist/
make ui-dev                                      # start both Python + Vite dev servers
```

## When You're Asked to Implement Something

Default workflow when picking up a task from TODO.md:

1. Find the matching phase. Read its citation of the design doc.
2. Open [DEVELOPMENT.md §4](DEVELOPMENT.md) at the same phase number. It lists the prerequisites, the parallelizable subtasks, the verification gate (the command/test that proves done), and the expected artifacts.
3. If the phase touches JEPA, also skim the relevant reference PDF — the design doc summarizes the paper but specific equations and ablations live in the source.
4. Check that no upstream phase is incomplete (see the dependency graph in [DEVELOPMENT.md §2](DEVELOPMENT.md)).
5. Lock the contracts file (`models/contracts.py`, Pydantic config schemas, etc.) before parallelizing implementation work. See [DEVELOPMENT.md §1.2](DEVELOPMENT.md).
6. Implement the smallest slice that passes the phase's verification gate. Don't fold in tasks from other phases.
7. Run the integration checkpoint from [DEVELOPMENT.md §7](DEVELOPMENT.md). A phase is not complete until its checkpoint passes on a fresh checkout.
8. Update TODO.md when a task is genuinely complete.

When dispatching parallel agents, use the templates in [DEVELOPMENT.md §6](DEVELOPMENT.md). Don't invent your own — they encode hard-won constraints (each agent self-contained, contracts locked, worktree isolation, no shared mutable state).

Open questions are tracked in [README.md](README.md) "Open Questions" — defer to the user before resolving them.

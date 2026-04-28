# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Status

This repository has moved past the pure design stage. It currently contains planning documents, reference papers, and an initial installable Python scaffold:

```text
jepa-rl/
  pyproject.toml
  Makefile
  README.md
  TODO.md
  DEVELOPMENT.md
  CLAUDE.md
  AGENTS.md
  configs/
    base.yaml
    presets/
    games/
  docs/
    design_doc.md
    *.pdf
  scripts/
  src/jepa_rl/
  tests/
```

Implemented now:

- uv-managed environment: `uv sync` creates `.venv` and installs from `uv.lock`. Dev tooling (pytest, ruff) is declared in `[dependency-groups].dev`; runtime extras stay under `[project.optional-dependencies]` (`config`, `browser`, `train`, `all`).
- `jepa-rl validate-config`.
- `jepa-rl init-run`.
- `jepa-rl open-game` to launch the configured game in visible Chromium.
- `jepa-rl ml-smoke` to verify the current linear Q learner reduces synthetic loss.
- `jepa-rl dashboard --run ...` to generate/open the run dashboard HTML.
- `jepa-rl ui --config ...` to run the live local control panel.
- `jepa-rl collect-random` for the local Breakout smoke game.
- `jepa-rl train` for the current NumPy linear pixel-Q smoke model.
- `jepa-rl eval` for the current `.npz` smoke-model checkpoints.
- Typed config validation and starter configs.
- Local Breakout-like HTML game at `games/breakout/index.html`.
- Playwright screenshot environment and DOM score reader.
- Discrete keyboard action parsing.
- In-memory replay buffer and sequence sampling.
- Unit tests for the implemented contracts.

Run any project command via `uv run <cmd>` (e.g. `uv run pytest`). The `Makefile` wraps the common targets and is the canonical reference for "how do I exercise this code path."

The current `train` command is not the planned DQN/JEPA stack; it is a small linear Q-learning smoke path that proves browser control, reward reading, metrics, and checkpointing. `jepa-rl train-world` is still a planned stub. This directory may not be a git repository yet. Check the actual workspace before assuming package, test, or git commands exist.

## Canonical Documents

Read these before any non-trivial change:

1. `docs/design_doc.md` is the design contract. If a reference paper disagrees with the design doc, the design doc wins for V1.
2. `TODO.md` is the implementation checklist. Treat phase numbers as stable references; add new tasks inside the relevant phase or at the end.
3. `DEVELOPMENT.md` is the development playbook. Use it when picking up any TODO item because it defines prerequisites, verification gates, expected artifacts, and parallelization boundaries.
4. `README.md` is the public-facing summary. Keep it accurate, but do not make it promise features that the implementation does not yet provide.
5. `CLAUDE.md` is the Claude-specific companion guide. Keep it broadly consistent with this file when changing repo guidance.

Precedence for implementation decisions:

```text
design_doc.md > TODO.md > DEVELOPMENT.md > README.md
```

When documents disagree, resolve the conflict explicitly in the relevant docs instead of silently choosing one.

## Version Boundaries

Keep the three JEPA roadmap versions separable through config and validation, not through ad hoc branches or mixed implementations.

| Version | Predictor signature | TODO phase | Status |
|---|---|---:|---|
| V1 | `Predictor(z_t, a_{t:t+k-1}, k)` | 6 | Target for first acceptance |
| V2 | `Predictor(z_t, pi_id, a_{t:t+k-1}, goal, k)` | 16 | Future |
| V3 | V2 + value-shaped latent loss + latent MPC planning | 17 | Future |

V1 is the only version required for the first accepted system. V2/V3 hooks may exist in config, but they must be off by default and rejected by validation until their phases land.

## Hard Constraints

Do not add features that violate these constraints without explicit user approval and matching updates to the design docs:

- No mutation of game state for advantage. Reading score is allowed; writing score, memory, storage, or network responses is not.
- No privileged JavaScript callbacks unless explicitly configured. If a config opts into JS score/reset callbacks, mark it `privileged: true` and include that fact in run metadata and evaluation reports.
- No personal browser sessions. Use isolated browser contexts and dedicated profiles only.
- No game source code as privileged input. The agent learns from pixels and, where configured, DOM text for score/state readers.
- No hardcoded experiment behavior that belongs in config.

## Configuration-First Discipline

Every important hyperparameter and runtime behavior must be declarative. When adding a user-facing or experiment-affecting knob:

1. Add it to the typed config model.
2. Add a default in `configs/base.yaml`.
3. Add validation for invalid combinations.
4. Update README's configuration example if the knob is public.
5. Ensure merged config snapshots are written to `runs/<experiment>/config.yaml`.

A run must be reproducible from `config.yaml` plus the code revision. Training code should not hide meaningful constants.

## JEPA Failure Modes

JEPA collapse is a first-class failure mode. Any JEPA training implementation must log:

- Latent variance per dimension and a per-batch variance floor metric.
- Latent covariance off-diagonal magnitude.
- Effective rank of the latent covariance.
- EMA tau schedule progress.
- Prediction loss broken down by horizon.

Variance/covariance regularization, latent normalization, and the EMA stop-gradient target encoder are required together for V1 JEPA training. Do not treat them as interchangeable alternatives.

## V1 Acceptance Criteria

V1 is not defined by an absolute high score. It is accepted when the system demonstrates:

1. Stable browser control across 100 episodes.
2. Reliable score extraction, with DOM preferred and OCR fallback where needed.
3. Pixel DQN beats random.
4. JEPA representation remains non-collapsed during training.
5. JEPA+DQN reaches the same score as pixel DQN in fewer environment steps.

Report scores at fixed budgets: 100k, 500k, 1M, and 5M environment steps.

## Implementation Workflow

When asked to implement a TODO item:

1. Identify the matching TODO phase.
2. Read the corresponding section in `DEVELOPMENT.md`.
3. Verify that upstream phases are complete before building downstream work.
4. Lock shared contracts first: config schemas, dataclasses, tensor shapes, interfaces, and integration tests.
5. Implement the smallest slice that passes the phase verification gate.
6. Run the phase verification command from `DEVELOPMENT.md`.
7. Update `TODO.md` only when the task is genuinely complete and verified.

Do not mark tasks complete based on inspection alone. If a verification gate cannot run yet, state that clearly and leave the TODO unchecked.

## Phase 0 Defaults

The Phase 0 scaffold is in place. The active toolchain is:

- uv (environment manager + locked dependency resolution; `uv.lock` is committed)
- Python 3.11+
- pytest, ruff via the `dev` dependency group
- PyYAML for config loading
- Optional extras (not installed by default): `browser` (Playwright + Pillow + OpenCV), `train` (NumPy + PyTorch), `all`
- TensorBoard / W&B / MLflow are deferred to Phase 4 and will sit behind configuration flags
- mypy may be added once the package structure stabilises

Bootstrap:

```bash
uv sync                  # dev env from uv.lock
uv sync --all-extras     # plus browser/train/config
uv run jepa-rl --help
uv run pytest
```

CI installs with `uv sync --frozen` to guarantee `uv.lock` parity. Do not regenerate the lock unintentionally — run `uv lock` (or `make lock`) only when dependencies actually change, then commit the resulting `uv.lock` diff.

## Reference Papers

Reference PDFs live in `docs/`:

| File | Paper | Main use |
|---|---|---|
| `2506.09985v1.pdf` | V-JEPA 2 | Passive video pretraining and action post-training |
| `2501.14622v4.pdf` | ACT-JEPA | Action chunking and imitation warm-start |
| `2510.00739v1.pdf` | TD-JEPA | V2 policy-conditioned predictor |
| `2601.00844v1.pdf` | Value-Guided Action Planning with JEPA | V3 value-shaped latent loss and latent MPC |
| `s41586-025-08744-2.pdf` | DreamerV3 | Robust latent actor-critic baseline |
| `2310.16828v2.pdf` | TD-MPC2 | Decoder-free latent MPC and hybrid/continuous actions |

Use the papers for detail, but keep V1 implementation aligned with `docs/design_doc.md`.

## Agent Coordination

Use `DEVELOPMENT.md` section 6 for agent-dispatch templates when work is explicitly split across agents. Keep write scopes disjoint and lock interfaces before parallel implementation.

For local Codex work:

- Prefer `rg`/`rg --files` for search.
- Inspect existing docs before editing.
- Keep edits scoped to the requested phase or guidance file.
- Do not run destructive git commands.
- Do not initialize git, install dependencies, or create generated artifacts unless the task calls for it.
- If tests or commands do not exist yet, say so instead of inventing a passing result.

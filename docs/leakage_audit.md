# Leakage Audit — Phase 8 Joint JEPA + DQN

This document explains why the joint JEPA + DQN training loop cannot leak future
observations into policy action selection.

## The three data paths

### 1. Action selection (policy path)

```
obs_t  -->  online_encoder(obs_t)  -->  z_t  -->  q_head(z_t)  -->  action
```

- `obs_t` is the current frame stack from the live environment — no future frames.
- `online_encoder` = `JepaWorldModel.encoder`, a plain convolutional network.
- This entire path runs under `torch.no_grad()`.
- The **target encoder is never called here**.

### 2. JEPA loss path (world model training)

```
obs_t  -->  online_encoder  -->  z_context
                                      |
                                  predictor(z_context, actions)  -->  z_pred_{t+k}
                                                                            |
                                                                       compare
                                                                            |
obs_{t+k}  -->  stop_grad(target_encoder)  -->  z_target_{t+k}  ----------/
```

- `obs_{t+k}` is a **stored replay transition** — not a future observation from the current
  step. It was collected in a previous env step and is a known past sample.
- The target encoder is decorated `@torch.no_grad()` and updated only via EMA
  (`update_target_encoder`). Its parameters do not receive gradients from the JEPA loss.
- The target encoder is **not called during action selection** — only during JEPA batch
  training. See `test_target_encoder_not_called_during_action_selection` in
  `tests/test_joint_dqn.py`.

### 3. DQN target computation (policy learning path)

```
replay.next_obs  -->  online_encoder  -->  z_next_online  -->  q_head  -->  best_action
                                                                                  |
replay.next_obs  -->  target_encoder  -->  z_next_target  -->  q_target  -->  Q(best_action)
                                                                                  |
                                                                        td_target  <-- reward + gamma * Q
```

- `replay.next_obs` is a **stored past transition** (`obs_{t+1}` at time `t`, inserted
  at env step `t`). It is never the live observation at the current decision step.
- The target encoder is used here, but only to evaluate the value of stored
  `next_obs` — not to observe anything the online policy hasn't already seen.
- This matches standard Double DQN: target network stabilizes value estimates, but
  cannot cause the online policy to act on information it hasn't received.

## Why the invariant is robust

1. **Temporal ordering**: the environment steps sequentially. `obs_t` is always the
   current observation; `next_obs` in replay was collected at step `t` and stored
   before step `t+1` is processed.

2. **Stop-gradient on target encoder**: `JepaWorldModel.encode_target` is
   `@torch.no_grad()` and `target_encoder.requires_grad = False`. Gradients from the
   DQN loss flow through `online_encoder`, not through `target_encoder`.

3. **Action selection under `no_grad`**: the action selection block wraps
   `jepa.encode(obs_t)` and `q_head(z_t)` in `torch.no_grad()`, making it
   impossible for the computation to affect encoder weights in-flight.

## Test reference

`tests/test_joint_dqn.py::test_target_encoder_not_called_during_action_selection`
registers forward hooks on all layers of `target_encoder`, runs the action selection
path, and asserts no hook fires.

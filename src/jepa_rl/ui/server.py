import base64
import collections
import contextlib
import dataclasses
import io
import json
import mimetypes
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from jepa_rl.browser.playwright_env import resolve_game_url
from jepa_rl.training.simple_q import train_linear_q
from jepa_rl.utils.artifacts import create_run_dir
from jepa_rl.utils.config import ProjectConfig, load_config, snapshot_config


@dataclass
class TrainingJob:
    run_name: str
    run_dir: Path
    requested_steps: int
    stop_event: threading.Event
    thread: threading.Thread
    status: str = "starting"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass
class EvalJob:
    run_name: str
    run_dir: Path
    config: ProjectConfig
    algorithm: str
    episodes_target: int
    checkpoint_name: str = ""
    episode_count: int = 0
    scores: list = field(default_factory=list)
    status: str = "running"
    result: dict[str, Any] | None = None
    error: str | None = None
    model: Any = None
    device: Any = None
    frame_buffer: Any = None


@dataclass
class CollectJob:
    run_name: str
    run_dir: Path
    episodes_target: int
    stop_event: threading.Event
    thread: threading.Thread
    status: str = "starting"
    episodes_done: int = 0
    mean_score: float = 0.0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    save_frames: bool = False
    scores: list[float] = field(default_factory=list)


@dataclass
class UiState:
    config_path: Path
    config: ProjectConfig
    experiment: str
    default_steps: int
    learning_starts: int
    batch_size: int | None
    dashboard_every: int
    run_dir: Path
    job: TrainingJob | None = None
    eval_job: EvalJob | None = None
    world_job: TrainingJob | None = None
    collect_job: CollectJob | None = None
    live_step_events: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=500)
    )
    world_step_events: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=200)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)


def run_ui_server(
    *,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    host: str,
    port: int,
    experiment: str,
    steps: int,
    learning_starts: int,
    batch_size: int | None,
    dashboard_every: int,
    open_browser: bool,
) -> None:
    if config_path is not None:
        config = load_config(config_path)
        actual_run_dir = Path(config.experiment.output_dir) / experiment
        actual_experiment = experiment
    elif run_dir is not None:
        actual_run_dir = Path(run_dir)
        actual_experiment = actual_run_dir.name
        snapshot = actual_run_dir / "config.yaml"
        if not snapshot.exists():
            raise OSError(f"no config snapshot in {actual_run_dir}; use --config instead")
        config = load_config(snapshot)
        config_path = snapshot
    else:
        for candidate in [Path("configs/games/breakout.yaml"), Path("configs/base.yaml")]:
            if candidate.exists():
                config_path = candidate
                break
        if config_path is None:
            raise OSError("no config found; run from project root or pass --config")
        config = load_config(config_path)
        actual_run_dir = Path(config.experiment.output_dir) / experiment
        actual_experiment = experiment

    state = UiState(
        config_path=config_path,
        config=config,
        experiment=actual_experiment,
        default_steps=steps,
        learning_starts=learning_starts,
        batch_size=batch_size,
        dashboard_every=dashboard_every,
        run_dir=actual_run_dir,
    )
    handler = _make_handler(state)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    print(f"training UI: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with state.lock:
            if state.job is not None and state.job.thread.is_alive():
                state.job.stop_event.set()
        print("training UI stopped")
    finally:
        server.server_close()


def _ui_dist_dir() -> Path | None:
    """Return the ui/dist/ directory if it exists (built Vue app)."""
    d = Path(__file__).resolve().parent.parent.parent.parent / "ui" / "dist"
    return d if d.is_dir() else None


def _make_handler(state: UiState) -> type[BaseHTTPRequestHandler]:
    class TrainingUiHandler(BaseHTTPRequestHandler):
        server_version = "JepaRlTrainingUi/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_index()
            elif parsed.path == "/game":
                self._send_game()
            elif parsed.path == "/api/state":
                self._send_json(build_state_payload(state))
            elif parsed.path == "/api/runs":
                qs = parse_qs(parsed.query)
                self._send_json(list_runs(state, include_smoke="true" in qs.get("smoke", [])))
            elif parsed.path == "/api/configs":
                self._send_json(list_configs())
            elif parsed.path == "/api/collected-datasets":
                self._send_json(list_collected_datasets(state))
            elif parsed.path == "/api/defaults":
                self._send_json(_get_defaults(state))
            elif parsed.path.startswith("/api/run-detail"):
                qs = parse_qs(urlparse(self.path).query)
                self._send_run_detail(qs)
            elif parsed.path == "/api/frame":
                self._send_frame()
            elif parsed.path.startswith("/assets/"):
                self._send_static_asset(parsed.path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/train/start":
                self._handle_train_start()
            elif parsed.path == "/api/train/stop":
                self._handle_train_stop()
            elif parsed.path == "/api/eval/start":
                self._handle_eval_start()
            elif parsed.path == "/api/eval/step":
                self._handle_eval_step()
            elif parsed.path == "/api/eval/stop":
                self._handle_eval_stop()
            elif parsed.path == "/api/switch-config":
                self._handle_switch_config()
            elif parsed.path == "/api/update-config":
                self._handle_update_config()
            elif parsed.path == "/api/create-config":
                self._handle_create_config()
            elif parsed.path == "/api/delete-run":
                self._handle_delete_run()
            elif parsed.path == "/api/validate-config":
                self._handle_validate_config()
            elif parsed.path == "/api/open-game":
                self._handle_open_game()
            elif parsed.path == "/api/ml-smoke":
                self._handle_ml_smoke()
            elif parsed.path == "/api/collect-random/start":
                self._handle_collect_random_start()
            elif parsed.path == "/api/collect-random/stop":
                self._handle_collect_random_stop()
            elif parsed.path == "/api/collect-random/step":
                self._handle_collect_random_step()
            elif parsed.path == "/api/run/create":
                self._handle_run_create()
            elif parsed.path == "/api/run/select":
                self._handle_run_select()
            elif parsed.path == "/api/train-world/start":
                self._handle_train_world_start()
            elif parsed.path == "/api/train-world/stop":
                self._handle_train_world_stop()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_index(self) -> None:
            dist = _ui_dist_dir()
            if dist is not None and (dist / "index.html").exists():
                self._send_file(dist / "index.html", "text/html; charset=utf-8")
            else:
                self.send_error(
                    HTTPStatus.NOT_FOUND,
                    "Vue UI not built — run `make ui-build` first",
                )

        def _send_static_asset(self, request_path: str) -> None:
            dist = _ui_dist_dir()
            if dist is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            rel = request_path.lstrip("/")
            file_path = (dist / rel).resolve()
            if not file_path.is_relative_to(dist.resolve()) or not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(file_path.name)
            self._send_file(file_path, mime or "application/octet-stream")

        def _handle_train_start(self) -> None:
            body = self._read_json_body()
            with state.lock:
                if state.job is not None and state.job.thread.is_alive():
                    self._send_json({"ok": False, "error": "training already running"}, status=409)
                    return

                run_name = str(body.get("experiment") or state.experiment)
                steps = int(body.get("steps") or state.default_steps)
                dashboard_every = int(body.get("dashboard_every") or state.dashboard_every)
                headed = bool(body.get("headed", False))
                jepa_ckpt_str = body.get("jepa_checkpoint")
                jepa_checkpoint = Path(jepa_ckpt_str) if jepa_ckpt_str else None
                try:
                    _validate_run_name(run_name)
                    run_dir = Path(state.config.experiment.output_dir) / run_name
                    snapshot = run_dir / "config.yaml"
                    if snapshot.exists():
                        if _has_locked_start_overrides(body):
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": (
                                        "run model settings are locked after creation; "
                                        "create a new run to change algorithm, observation, "
                                        "optimizer, replay, or exploration settings"
                                    ),
                                },
                                status=409,
                            )
                            return
                        run_config = load_config(snapshot)
                        run_dir = create_run_dir(run_config.experiment.output_dir, run_name)
                    else:
                        overrides = list(body.get("overrides") or [])
                        overrides.extend(_legacy_start_overrides(body))
                        run_config = _apply_config_overrides(state.config, overrides)
                        run_config = dataclasses.replace(
                            run_config,
                            experiment=dataclasses.replace(
                                run_config.experiment, name=run_name
                            ),
                        )
                        run_dir = create_run_dir(run_config.experiment.output_dir, run_name)
                        snapshot_config(run_config, run_dir / "config.yaml")
                except (OSError, TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return

                stop_event = threading.Event()
                job = TrainingJob(
                    run_name=run_name,
                    run_dir=run_dir,
                    requested_steps=steps,
                    stop_event=stop_event,
                    thread=threading.Thread(
                        target=_training_worker,
                        args=(
                            state,
                            run_config,
                            run_name,
                            steps,
                            run_config.agent.learning_starts,
                            run_config.agent.batch_size,
                            dashboard_every,
                            stop_event,
                            None,
                            headed,
                            jepa_checkpoint,
                        ),
                        daemon=True,
                    ),
                )
                state.experiment = run_name
                state.run_dir = run_dir
                state.job = job
                state.live_step_events.clear()
                job.thread.start()
            self._send_json({"ok": True, "run_dir": str(run_dir)})

        def _handle_train_stop(self) -> None:
            with state.lock:
                if state.job is None or not state.job.thread.is_alive():
                    self._send_json({"ok": True, "status": "not_running"})
                    return
                state.job.status = "stopping"
                state.job.stop_event.set()
            self._send_json({"ok": True, "status": "stopping"})

        def _handle_eval_start(self) -> None:
            body = self._read_json_body()
            episodes = int(body.get("episodes") or 3)
            eval_run_dir_str = body.get("run_dir")
            eval_run_dir = Path(eval_run_dir_str) if eval_run_dir_str else state.run_dir
            eval_config = state.config
            snapshot = eval_run_dir / "config.yaml"
            if snapshot.exists():
                try:
                    eval_config = load_config(snapshot)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        {"ok": False, "error": f"run config invalid: {exc}"}, status=400
                    )
                    return
            algorithm = eval_config.agent.algorithm
            pt_algs = {"dqn", "frozen_jepa_dqn", "joint_jepa_dqn"}
            expected_suffix = ".pt" if algorithm in pt_algs else ".npz"
            ckpt_name = body.get("checkpoint")
            if ckpt_name:
                if not ckpt_name.endswith(expected_suffix):
                    self._send_json(
                        {
                            "ok": False,
                            "error": (
                                f"checkpoint {ckpt_name} is not a {algorithm} "
                                f"checkpoint (expected {expected_suffix})"
                            ),
                        },
                        status=400,
                    )
                    return
                checkpoint = eval_run_dir / "checkpoints" / ckpt_name
            else:
                checkpoint = eval_run_dir / "checkpoints" / f"latest{expected_suffix}"
            if not checkpoint.exists():
                self._send_json(
                    {"ok": False, "error": f"checkpoint not found: {checkpoint.name}"}, status=404
                )
                return
            try:
                device = None
                if algorithm == "dqn":
                    from jepa_rl.models.device import resolve_torch_device
                    from jepa_rl.models.dqn import build_q_network
                    from jepa_rl.utils.checkpoint import load_torch_checkpoint

                    device = resolve_torch_device(eval_config.experiment.device)
                    net = build_q_network(
                        eval_config, num_actions=eval_config.actions.num_actions
                    ).to(device)
                    load_torch_checkpoint(checkpoint, model=net, target_model=None, optimizer=None)
                    net.eval()
                    model = net
                elif algorithm == "frozen_jepa_dqn":
                    from jepa_rl.models.device import resolve_torch_device
                    from jepa_rl.models.frozen_jepa_dqn import build_frozen_jepa_q_network
                    from jepa_rl.utils.checkpoint import load_torch_checkpoint

                    device = resolve_torch_device(eval_config.experiment.device)
                    jepa_ckpt_override = body.get("jepa_checkpoint")
                    if jepa_ckpt_override:
                        jepa_ckpt = Path(jepa_ckpt_override)
                    else:
                        jepa_ckpt = (
                            eval_config.training.extra.get("jepa_checkpoint")
                            if hasattr(eval_config.training, "extra")
                            else None
                        )
                    if not jepa_ckpt:
                        candidates = sorted(
                            Path(eval_config.experiment.output_dir).glob(
                                "*_world/checkpoints/latest.pt"
                            ),
                            reverse=True,
                        )
                        if not candidates:
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": "No JEPA checkpoint found for frozen_jepa_dqn eval",
                                },
                                status=400,
                            )
                            return
                        jepa_ckpt = candidates[0]
                    net = build_frozen_jepa_q_network(
                        eval_config,
                        num_actions=eval_config.actions.num_actions,
                        jepa_checkpoint_path=Path(jepa_ckpt),
                    ).to(device)
                    load_torch_checkpoint(checkpoint, model=net, target_model=None, optimizer=None)
                    net.eval()
                    model = net
                elif algorithm == "joint_jepa_dqn":
                    import torch as _torch

                    from jepa_rl.models.device import resolve_torch_device
                    from jepa_rl.models.jepa import JepaWorldModel, JepaWorldModelConfig
                    from jepa_rl.models.joint_jepa_dqn import LatentQHead

                    device = resolve_torch_device(eval_config.experiment.device)
                    _jstate = _torch.load(checkpoint, map_location="cpu", weights_only=False)
                    _jepa_cfg = JepaWorldModelConfig.from_project_config(eval_config)
                    _jepa_enc = JepaWorldModel(_jepa_cfg).to(device)
                    _jepa_enc.load_state_dict(_jstate["jepa_model"])
                    _jepa_enc.eval()
                    _q_head = LatentQHead(
                        latent_dim=eval_config.world_model.latent_dim,
                        num_actions=eval_config.actions.num_actions,
                        hidden_dims=eval_config.agent.q_network.hidden_dims,
                        dueling=eval_config.agent.q_network.dueling,
                    ).to(device)
                    _q_head.load_state_dict(_jstate["q_net"])
                    _q_head.eval()
                    model = (_jepa_enc, _q_head)
                elif algorithm == "linear_q":
                    from jepa_rl.training.simple_q import LinearQModel

                    model = LinearQModel.load(checkpoint)
                else:
                    self._send_json(
                        {"ok": False, "error": f"in-iframe eval not supported for {algorithm}"},
                        status=400,
                    )
                    return
                frame_stack = eval_config.observation.frame_stack
                frame_buffer: collections.deque = collections.deque(maxlen=frame_stack)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            with state.lock:
                state.eval_job = EvalJob(
                    run_name=eval_run_dir.name,
                    run_dir=eval_run_dir,
                    config=eval_config,
                    algorithm=algorithm,
                    episodes_target=episodes,
                    checkpoint_name=checkpoint.name,
                    model=model,
                    device=device,
                    frame_buffer=frame_buffer,
                )
            self._send_json({"ok": True})

        def _handle_eval_step(self) -> None:
            with state.lock:
                eval_job = state.eval_job
            if eval_job is None or eval_job.model is None:
                self._send_json(
                    {"ok": False, "error": "no eval session; call /api/eval/start first"},
                    status=400,
                )
                return
            eval_config = eval_job.config
            body = self._read_json_body()
            frame_b64 = body.get("frame", "")
            is_done = bool(body.get("done", False))
            score = float(body.get("score") or 0.0)

            if is_done:
                with state.lock:
                    eval_job.scores.append(score)
                    eval_job.episode_count += 1
                    eval_job.frame_buffer.clear()
                    ep_count = eval_job.episode_count
                    ep_target = eval_job.episodes_target
                    complete = ep_count >= ep_target
                    if complete:
                        import numpy as np

                        sc = eval_job.scores
                        eval_job.status = "completed"
                        eval_job.result = {
                            "episodes": ep_count,
                            "scores": sc,
                            "best_score": max(sc) if sc else 0.0,
                            "mean_score": float(np.mean(sc)) if sc else 0.0,
                        }
                self._send_json(
                    {"ok": True, "done": True, "complete": complete, "episode": ep_count}
                )
                return

            try:
                png_bytes = base64.b64decode(frame_b64)
                frame = _preprocess_canvas_frame(png_bytes, eval_config)
            except Exception as exc:  # noqa: BLE001
                with state.lock:
                    eval_job.status = "error"
                    eval_job.error = f"frame decode failed: {exc}"
                self._send_json({"ok": False, "error": f"frame decode failed: {exc}"}, status=400)
                return

            with state.lock:
                eval_job.frame_buffer.append(frame)
                frames = list(eval_job.frame_buffer)
                model = eval_job.model
                device = eval_job.device
                algorithm = eval_job.algorithm

            while len(frames) < eval_config.observation.frame_stack:
                frames.insert(0, frames[0])

            try:
                import numpy as np

                q_values_list: list[float] = []
                obs = np.concatenate(frames, axis=0)
                if algorithm in {"dqn", "frozen_jepa_dqn"}:
                    import torch

                    obs_t = torch.from_numpy(obs[None]).to(device)
                    with torch.no_grad():
                        q_out = model(obs_t)
                        action_idx = int(q_out.argmax(dim=1).item())
                        q_values_list = q_out.squeeze().tolist()
                elif algorithm == "joint_jepa_dqn":
                    import torch

                    _jepa_m, _q_m = model
                    obs_t = torch.from_numpy(obs[None]).float().to(device)
                    with torch.no_grad():
                        _z = _jepa_m.encode(obs_t)
                        q_out = _q_m(_z)
                        action_idx = int(q_out.argmax(dim=1).item())
                        q_values_list = q_out.squeeze().tolist()
                else:
                    from jepa_rl.envs.browser_game_env import Observation
                    from jepa_rl.training.simple_q import featurize_observation

                    observation = Observation(
                        data=obs,
                        width=eval_config.observation.width,
                        height=eval_config.observation.height,
                        channels=obs.shape[0],
                    )
                    features = featurize_observation(observation)
                    raw_q = model.q_values(features)
                    action_idx = int(np.argmax(raw_q))
                    q_values_list = raw_q.tolist()
                action_key = eval_config.actions.keys[action_idx]
            except Exception as exc:  # noqa: BLE001
                with state.lock:
                    eval_job.status = "error"
                    eval_job.error = f"inference failed: {exc}"
                self._send_json({"ok": False, "error": f"inference failed: {exc}"}, status=500)
                return

            self._send_json(
                {
                    "ok": True,
                    "done": False,
                    "complete": False,
                    "action": action_key,
                    "q_values": q_values_list,
                }
            )

        def _handle_eval_stop(self) -> None:
            with state.lock:
                state.eval_job = None
            self._send_json({"ok": True})

        def _handle_validate_config(self) -> None:
            body = self._read_json_body()
            config_path_str = body.get("config") or str(state.config_path)
            try:
                cfg = load_config(Path(config_path_str))
                self._send_json(
                    {
                        "ok": True,
                        "game": cfg.game.name,
                        "algorithm": cfg.agent.algorithm,
                        "actions": cfg.actions.num_actions,
                        "device": cfg.experiment.device,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _handle_open_game(self) -> None:
            body = self._read_json_body()
            seconds = float(body.get("seconds") or 10.0)
            random_steps = int(body.get("random_steps") or 0)
            hold = bool(body.get("hold", False))
            try:
                import threading as _threading

                from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

                def _open_worker() -> None:
                    with PlaywrightBrowserGameEnv(
                        state.config, headless=False, run_dir=state.run_dir
                    ) as env:
                        env.reset()
                        if random_steps > 0:
                            import random as _random

                            rng = _random.Random(state.config.experiment.seed)
                            for _ in range(random_steps):
                                env.step(rng.randrange(state.config.actions.num_actions))
                        if not hold:
                            import time as _time

                            _time.sleep(seconds)

                t = _threading.Thread(target=_open_worker, daemon=True)
                t.start()
                self._send_json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def _handle_ml_smoke(self) -> None:
            body = self._read_json_body()
            steps = int(body.get("steps") or 2000)
            lr = float(body.get("lr") or 0.03)
            seed = int(body.get("seed") or 0)
            try:
                from jepa_rl.training.simple_q import run_linear_q_ml_smoke

                result = run_linear_q_ml_smoke(steps=steps, lr=lr, seed=seed)
                self._send_json(
                    {
                        "ok": True,
                        "passed": result.improvement > 0,
                        "steps": result.steps,
                        "initial_loss": result.initial_loss,
                        "final_loss": result.final_loss,
                        "improvement": result.improvement,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def _handle_collect_random_start(self) -> None:
            body = self._read_json_body()
            with state.lock:
                if state.collect_job is not None and (
                    state.collect_job.thread.is_alive()
                    or state.collect_job.status == "running"
                ):
                    self._send_json(
                        {"ok": False, "error": "collection already running"}, status=409
                    )
                    return
                run_name = str(body.get("experiment") or f"{state.config.game.name}_random")
                episodes = int(body.get("episodes") or 5)
                max_steps = int(body.get("max_steps") or 200)
                headed = bool(body.get("headed", False))
                save_frames = bool(body.get("save_frames", False))
                run_dir = create_run_dir(state.config.experiment.output_dir, run_name)
                snapshot_config(state.config, run_dir / "config.yaml")
                stop_event = threading.Event()
                if headed:
                    # Server-driven mode: background Playwright worker
                    cj = CollectJob(
                        run_name=run_name,
                        run_dir=run_dir,
                        episodes_target=episodes,
                        stop_event=stop_event,
                        thread=threading.Thread(
                            target=_collect_random_worker,
                            args=(
                                state, run_name, run_dir,
                                episodes, max_steps, headed, stop_event,
                            ),
                            daemon=True,
                        ),
                        save_frames=save_frames,
                    )
                    state.collect_job = cj
                    cj.thread.start()
                else:
                    # Client-driven mode: frontend sends steps via /api/collect-random/step
                    dummy_thread = threading.Thread(daemon=True)
                    cj = CollectJob(
                        run_name=run_name,
                        run_dir=run_dir,
                        episodes_target=episodes,
                        stop_event=stop_event,
                        thread=dummy_thread,
                        save_frames=save_frames,
                    )
                    cj.status = "running"
                    state.collect_job = cj
            self._send_json({"ok": True, "run_dir": str(run_dir)})

        def _handle_collect_random_stop(self) -> None:
            with state.lock:
                if state.collect_job is None:
                    self._send_json({"ok": True, "status": "not_running"})
                    return
                if state.collect_job.thread.is_alive():
                    state.collect_job.stop_event.set()
                state.collect_job.status = "stopped"
                state.collect_job.completed_at = time.time()
            self._send_json({"ok": True, "status": "stopped"})

        def _handle_collect_random_step(self) -> None:
            import random as _random

            with state.lock:
                cj = state.collect_job
            if cj is None or cj.status not in ("running", "starting"):
                self._send_json(
                    {
                        "ok": False,
                        "error": (
                            "no collection running; "
                            "call /api/collect-random/start first"
                        ),
                    },
                    status=400,
                )
                return
            body = self._read_json_body()
            is_done = bool(body.get("done", False))
            score = float(body.get("score") or 0.0)
            frame_b64 = body.get("frame", "")

            # Save frame if requested
            if cj.save_frames and frame_b64:
                try:
                    png_bytes = base64.b64decode(frame_b64)
                    frame_path = cj.run_dir / "frame.png"
                    frame_path.write_bytes(png_bytes)
                except Exception:  # noqa: BLE001
                    pass

            if is_done:
                with state.lock:
                    cj.scores.append(score)
                    cj.episodes_done += 1
                    cj.mean_score = float(sum(cj.scores) / len(cj.scores))
                    ep_count = cj.episodes_done
                    ep_target = cj.episodes_target
                    complete = ep_count >= ep_target
                    if complete:
                        cj.status = "completed"
                        cj.completed_at = time.time()
                self._send_json(
                    {"ok": True, "done": True, "complete": complete, "episode": ep_count}
                )
                return

            # Pick random action
            num_actions = state.config.actions.num_actions
            action_idx = _random.randrange(num_actions)
            action_key = state.config.actions.keys[action_idx]
            self._send_json({"ok": True, "done": False, "complete": False, "action": action_key})

        def _handle_run_create(self) -> None:
            body = self._read_json_body()
            run_name = str(body.get("name") or body.get("experiment") or "").strip()
            if not run_name:
                self._send_json({"ok": False, "error": "run name required"}, status=400)
                return
            with state.lock:
                if state.job is not None and state.job.thread.is_alive():
                    self._send_json(
                        {"ok": False, "error": "cannot create a run while training"},
                        status=409,
                    )
                    return
                try:
                    _validate_run_name(run_name)
                    run_dir = Path(state.config.experiment.output_dir) / run_name
                    if (run_dir / "config.yaml").exists():
                        self._send_json(
                            {"ok": False, "error": "run already exists"},
                            status=409,
                        )
                        return
                    run_config = _apply_config_overrides(
                        state.config,
                        body.get("overrides", []),
                    )
                    run_config = dataclasses.replace(
                        run_config,
                        experiment=dataclasses.replace(
                            run_config.experiment, name=run_name
                        ),
                    )
                    run_dir = create_run_dir(run_config.experiment.output_dir, run_name)
                    snapshot_config(run_config, run_dir / "config.yaml")
                    state.experiment = run_name
                    state.run_dir = run_dir
                    state.live_step_events.clear()
                except (OSError, TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
            self._send_json(
                {
                    "ok": True,
                    "name": run_name,
                    "run_dir": str(run_dir),
                    "detail": _build_config_detail(run_config),
                    "model_info": _build_model_info(run_config),
                    "checkpoints": _list_checkpoints(
                        run_dir, algorithm=run_config.agent.algorithm
                    ),
                }
            )

        def _handle_run_select(self) -> None:
            body = self._read_json_body()
            run_name = str(body.get("name") or "").strip()
            if not run_name:
                self._send_json({"ok": False, "error": "run name required"}, status=400)
                return
            with state.lock:
                if (
                    state.job is not None
                    and state.job.thread.is_alive()
                    and state.job.run_name != run_name
                ):
                    self._send_json(
                        {"ok": False, "error": "cannot switch runs while training"},
                        status=409,
                    )
                    return
                try:
                    _validate_run_name(run_name)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                run_dir = Path(state.config.experiment.output_dir) / run_name
                snapshot = run_dir / "config.yaml"
                if not snapshot.exists():
                    self._send_json({"ok": False, "error": "run not found"}, status=404)
                    return
                try:
                    run_config = load_config(snapshot)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        {"ok": False, "error": f"run config invalid: {exc}"},
                        status=400,
                    )
                    return
                state.experiment = run_name
                state.run_dir = run_dir
                state.live_step_events.clear()
            summary = _read_json(run_dir / "metrics" / "train_summary.json")
            self._send_json(
                {
                    "ok": True,
                    "name": run_name,
                    "run_dir": str(run_dir),
                    "detail": _build_config_detail(run_config),
                    "summary": summary,
                    "model_info": _build_model_info(run_config),
                    "checkpoints": _list_checkpoints(
                        run_dir, algorithm=run_config.agent.algorithm
                    ),
                }
            )

        def _handle_train_world_start(self) -> None:
            body = self._read_json_body()
            with state.lock:
                if state.world_job is not None and state.world_job.thread.is_alive():
                    self._send_json(
                        {"ok": False, "error": "world model training already running"}, status=409
                    )
                    return
                run_name = str(body.get("experiment") or f"{state.experiment}_world")
                steps = int(body.get("steps") or 1000)
                collect_steps_val = body.get("collect_steps")
                collect_steps = (
                    int(collect_steps_val) if collect_steps_val not in (None, "") else None
                )
                batch_size_val = body.get("batch_size")
                batch_size = int(batch_size_val) if batch_size_val not in (None, "") else None
                lr_val = body.get("lr")
                lr = float(lr_val) if lr_val not in (None, "") else None
                dashboard_every = int(body.get("dashboard_every") or 25)
                headed = bool(body.get("headed", False))
                world_config = state.config
                active_snapshot = state.run_dir / "config.yaml"
                if active_snapshot.exists():
                    world_config = load_config(active_snapshot)
                run_dir = create_run_dir(world_config.experiment.output_dir, run_name)
                snapshot_config(world_config, run_dir / "config.yaml")
                stop_event = threading.Event()
                wj = TrainingJob(
                    run_name=run_name,
                    run_dir=run_dir,
                    requested_steps=steps,
                    stop_event=stop_event,
                    thread=threading.Thread(
                        target=_train_world_worker,
                        args=(
                            state,
                            world_config,
                            run_name,
                            steps,
                            collect_steps,
                            batch_size,
                            lr,
                            dashboard_every,
                            headed,
                            stop_event,
                        ),
                        daemon=True,
                    ),
                )
                state.world_job = wj
                state.world_step_events.clear()
                wj.thread.start()
            self._send_json({"ok": True, "run_dir": str(run_dir)})

        def _handle_train_world_stop(self) -> None:
            with state.lock:
                if state.world_job is None or not state.world_job.thread.is_alive():
                    self._send_json({"ok": True, "status": "not_running"})
                    return
                state.world_job.status = "stopping"
                state.world_job.stop_event.set()
            self._send_json({"ok": True, "status": "stopping"})

        def _handle_switch_config(self) -> None:
            body = self._read_json_body()
            config_path_str = body.get("config")
            if not config_path_str:
                self._send_json({"ok": False, "error": "config path required"}, status=400)
                return
            try:
                new_path = Path(config_path_str)
                new_config = load_config(new_path)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            with state.lock:
                if state.job is not None and state.job.thread.is_alive():
                    self._send_json(
                        {"ok": False, "error": "cannot switch config while training"}, status=409
                    )
                    return
                state.config = new_config
                state.config_path = new_path
            self._send_json({"ok": True, "game": new_config.game.name})

        def _handle_update_config(self) -> None:
            body = self._read_json_body()
            overrides = body.get("overrides")
            if not overrides or not isinstance(overrides, list):
                self._send_json({"ok": False, "error": "overrides list required"}, status=400)
                return
            with state.lock:
                if state.job is not None and state.job.thread.is_alive():
                    self._send_json(
                        {"ok": False, "error": "cannot update config while training"}, status=409
                    )
                    return
                try:
                    new_config = state.config
                    for item in overrides:
                        group = str(item.get("group", ""))
                        key = str(item.get("key", ""))
                        value = str(item.get("value", ""))
                        new_config = _apply_config_override(new_config, group, key, value)
                    state.config = new_config
                    snapshot_config(new_config, state.config_path)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
            self._send_json({"ok": True})

        def _handle_create_config(self) -> None:
            body = self._read_json_body()
            name = (body.get("name") or "").strip()
            if not name:
                self._send_json({"ok": False, "error": "name required"}, status=400)
                return
            if not name.replace("_", "").replace("-", "").isalnum():
                self._send_json(
                    {"ok": False, "error": "name must be alphanumeric (dashes/underscores ok)"},
                    status=400,
                )
                return
            config_dir = Path("configs/games")
            config_dir.mkdir(parents=True, exist_ok=True)
            target = config_dir / f"{name}.yaml"
            if target.exists():
                self._send_json({"ok": False, "error": "config already exists"}, status=409)
                return
            import yaml

            with state.lock, target.open("w", encoding="utf-8") as handle:
                yaml.dump(
                    {"game": state.config.game.name},
                    handle,
                    default_flow_style=False,
                )
            self._send_json({"ok": True, "path": str(target)})

        def _handle_delete_run(self) -> None:
            body = self._read_json_body()
            name = (body.get("name") or "").strip()
            if not name:
                self._send_json({"ok": False, "error": "name required"}, status=400)
                return
            with state.lock:
                if state.job is not None and state.job.thread.is_alive():
                    run_dir_name = state.job.run_dir.name if state.job.run_dir else ""
                    if run_dir_name == name:
                        self._send_json(
                            {"ok": False, "error": "cannot delete active training run"}, status=409
                        )
                        return
            import shutil

            output_dir = Path(state.config.experiment.output_dir)
            run_dir = output_dir / name
            if not run_dir.is_dir():
                self._send_json({"ok": False, "error": "run not found"}, status=404)
                return
            shutil.rmtree(run_dir)
            self._send_json({"ok": True})

        def _send_game(self) -> None:
            game_url = resolve_game_url(state.config.game.url)
            if not game_url.startswith("file://"):
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", game_url)
                self.end_headers()
                return
            path = Path(urlparse(game_url).path)
            self._send_file(path, "text/html; charset=utf-8")

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if not raw:
                return {}
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.loads(raw)
            parsed = parse_qs(raw)
            return {key: values[-1] for key, values in parsed.items()}

        def _send_json(self, data: dict[str, Any], *, status: int = 200) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_frame(self) -> None:
            with state.lock:
                job = state.job
            if job is not None:
                frame_path = job.run_dir / "frame.png"
                if frame_path.exists():
                    self._send_file(frame_path, "image/png")
                    return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def _send_run_detail(self, qs: dict[str, list[str]]) -> None:
            name = qs.get("name", [""])[0]
            if not name:
                self._send_json({"detail": [], "summary": {}})
                return
            output_dir = Path(state.config.experiment.output_dir)
            run_dir = output_dir / name
            if not (run_dir / "config.yaml").exists():
                self._send_json({"detail": [], "summary": {}})
                return
            try:
                run_config = load_config(run_dir / "config.yaml")
                detail = _build_config_detail(run_config)
            except Exception:  # noqa: BLE001
                detail = []
                run_config = None
            summary = _read_json(run_dir / "metrics" / "train_summary.json")
            algo = str(
                summary.get(
                    "algorithm",
                    run_config.agent.algorithm if run_config is not None else "",
                )
            )
            self._send_json(
                {
                    "detail": detail,
                    "summary": summary,
                    "checkpoints": _list_checkpoints(run_dir, algorithm=algo),
                    "run_dir": str(run_dir),
                    "model_info": (
                        _build_model_info(run_config)
                        if run_config is not None
                        else {}
                    ),
                }
            )

    return TrainingUiHandler


def _training_worker(
    state: UiState,
    config: ProjectConfig,
    run_name: str,
    steps: int,
    learning_starts: int,
    batch_size: int | None,
    dashboard_every: int,
    stop_event: threading.Event,
    lr: float | None = None,
    headed: bool = False,
    jepa_checkpoint: Path | None = None,
) -> None:
    with state.lock:
        if state.job is not None:
            state.job.status = "running"
            job_run_dir = state.job.run_dir
    try:
        algorithm = config.agent.algorithm
        screenshot = job_run_dir / "frame.png"

        def live_step_callback(event: dict[str, Any]) -> None:
            with state.lock:
                state.live_step_events.append(event)

        if algorithm == "dqn":
            from jepa_rl.training.pixel_dqn import train_dqn

            train_dqn(
                config,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=not headed,
                stop_event=stop_event,
                screenshot_path=screenshot,
                live_step_callback=live_step_callback,
            )
        elif algorithm == "frozen_jepa_dqn":
            from jepa_rl.training.frozen_jepa_dqn import train_frozen_jepa_dqn

            if jepa_checkpoint is None:
                jepa_ckpt_cfg = (
                    config.training.extra.get("jepa_checkpoint")
                    if hasattr(config.training, "extra")
                    else None
                )
                if jepa_ckpt_cfg:
                    jepa_checkpoint = Path(jepa_ckpt_cfg)
                else:
                    candidates = sorted(
                        Path(config.experiment.output_dir).glob(
                            "*_world/checkpoints/latest.pt"
                        ),
                        reverse=True,
                    )
                    if not candidates:
                        raise ValueError(
                            "No JEPA checkpoint found. Run train-world first or "
                            "provide a JEPA checkpoint path."
                        )
                    jepa_checkpoint = candidates[0]
            train_frozen_jepa_dqn(
                config,
                jepa_checkpoint=jepa_checkpoint,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=not headed,
                stop_event=stop_event,
                screenshot_path=screenshot,
                live_step_callback=live_step_callback,
            )
        elif algorithm == "joint_jepa_dqn":
            from jepa_rl.training.joint_jepa_dqn import train_joint_jepa_dqn

            train_joint_jepa_dqn(
                config,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=not headed,
                stop_event=stop_event,
                screenshot_path=screenshot,
                live_step_callback=live_step_callback,
            )
        else:
            train_linear_q(
                config,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                lr=lr,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=not headed,
                stop_event=stop_event,
                screenshot_path=screenshot,
                live_step_callback=live_step_callback,
            )
    except Exception as exc:  # noqa: BLE001 - local UI needs to expose worker errors.
        with state.lock:
            if state.job is not None:
                state.job.status = "error"
                state.job.error = str(exc)
                state.job.completed_at = time.time()
        return

    with state.lock:
        if state.job is not None:
            state.job.status = "stopped" if stop_event.is_set() else "completed"
            state.job.completed_at = time.time()


def _collect_random_worker(
    state: UiState,
    run_name: str,
    run_dir: Path,
    episodes: int,
    max_steps: int,
    headed: bool,
    stop_event: threading.Event,
) -> None:
    import json
    import random
    import statistics

    with state.lock:
        if state.collect_job is not None:
            state.collect_job.status = "running"

    try:
        from jepa_rl.browser.playwright_env import PlaywrightBrowserGameEnv

        metrics_path = run_dir / "metrics" / "random_events.jsonl"
        rng = random.Random(state.config.experiment.seed)
        episodes_data: list[dict[str, Any]] = []

        with (
            PlaywrightBrowserGameEnv(state.config, headless=not headed, run_dir=run_dir) as env,
            metrics_path.open("w", encoding="utf-8") as metrics,
        ):
            for episode in range(episodes):
                if stop_event.is_set():
                    break
                env.reset()
                episode_return = 0.0
                score = 0.0
                steps_taken = 0
                for step_index in range(1, max_steps + 1):
                    if stop_event.is_set():
                        break
                    steps_taken = step_index
                    action = rng.randrange(state.config.actions.num_actions)
                    result = env.step(action)
                    episode_return += result.reward
                    score = result.score
                    if result.done:
                        break
                record: dict[str, Any] = {
                    "type": "episode",
                    "episode": episode,
                    "return": episode_return,
                    "score": score,
                    "steps": steps_taken,
                }
                metrics.write(json.dumps(record) + "\n")
                episodes_data.append(record)
                with state.lock:
                    if state.collect_job is not None:
                        state.collect_job.episodes_done = episode + 1
                        scores_so_far = [e["score"] for e in episodes_data]
                        state.collect_job.mean_score = float(
                            sum(scores_so_far) / len(scores_so_far)
                        )
    except Exception as exc:  # noqa: BLE001
        with state.lock:
            if state.collect_job is not None:
                state.collect_job.status = "error"
                state.collect_job.error = str(exc)
                state.collect_job.completed_at = time.time()
        return

    if episodes_data:
        scores = [ep["score"] for ep in episodes_data]
        lengths = [ep["steps"] for ep in episodes_data]
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        summary: dict[str, Any] = {
            "total_episodes": n,
            "score_mean": statistics.mean(scores),
            "score_median": statistics.median(scores),
            "score_p95": sorted_scores[min(int(n * 0.95), n - 1)],
            "score_p5": sorted_scores[min(int(n * 0.05), n - 1)],
            "score_max": max(scores),
            "score_min": min(scores),
            "mean_episode_length": statistics.mean(lengths),
        }
        summary_path = run_dir / "random_baseline_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    with state.lock:
        if state.collect_job is not None:
            state.collect_job.status = "stopped" if stop_event.is_set() else "completed"
            state.collect_job.completed_at = time.time()


def _train_world_worker(
    state: UiState,
    config: ProjectConfig,
    run_name: str,
    steps: int,
    collect_steps: int | None,
    batch_size: int | None,
    lr: float | None,
    dashboard_every: int,
    headed: bool,
    stop_event: threading.Event,
) -> None:
    with state.lock:
        if state.world_job is not None:
            state.world_job.status = "running"

    try:
        from jepa_rl.training.jepa_world import train_jepa_world

        train_jepa_world(
            config,
            experiment=run_name,
            steps=steps,
            collect_steps=collect_steps,
            batch_size=batch_size,
            lr=lr,
            dashboard_every=dashboard_every,
            headless=not headed,
            stop_event=stop_event,
        )
    except Exception as exc:  # noqa: BLE001
        with state.lock:
            if state.world_job is not None:
                state.world_job.status = "error"
                state.world_job.error = str(exc)
                state.world_job.completed_at = time.time()
        return

    with state.lock:
        if state.world_job is not None:
            state.world_job.status = "stopped" if stop_event.is_set() else "completed"
            state.world_job.completed_at = time.time()


_CONFIG_GROUPS: dict[str, type] = {
    "experiment": type(None),  # filled at runtime
    "game": type(None),
    "observation": type(None),
    "actions": type(None),
    "reward": type(None),
    "agent": type(None),
    "exploration": type(None),
    "replay": type(None),
    "world_model": type(None),
    "training": type(None),
}


def _validate_run_name(name: str) -> None:
    if not name.strip():
        raise ValueError("run name cannot be empty")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("run name cannot contain path separators")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run name must be alphanumeric (dashes/underscores ok)")


def _has_locked_start_overrides(body: dict[str, Any]) -> bool:
    overrides = body.get("overrides")
    if isinstance(overrides, list) and len(overrides) > 0:
        return True
    if overrides not in (None, "", []):
        return True
    return any(body.get(key) not in (None, "") for key in ("learning_starts", "batch_size", "lr"))


def _legacy_start_overrides(body: dict[str, Any]) -> list[dict[str, str]]:
    overrides: list[dict[str, str]] = []
    if body.get("learning_starts") not in (None, ""):
        overrides.append(
            {
                "group": "agent",
                "key": "learning_starts",
                "value": str(body["learning_starts"]),
            }
        )
    if body.get("batch_size") not in (None, ""):
        overrides.append(
            {
                "group": "agent",
                "key": "batch_size",
                "value": str(body["batch_size"]),
            }
        )
    if body.get("lr") not in (None, ""):
        overrides.append(
            {
                "group": "agent",
                "key": "optimizer.lr",
                "value": str(body["lr"]),
            }
        )
    return overrides


def _apply_config_override(
    config: ProjectConfig, group: str, key: str, value: str
) -> ProjectConfig:
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
    sub = group_map.get(group)
    key_path = key.split(".")
    if sub is None or not key_path or not hasattr(sub, key_path[0]):
        raise ValueError(f"unknown config field: {group}.{key}")
    new_sub = _replace_config_value(sub, key_path, value, f"{group}.{key}")
    return dataclasses.replace(config, **{group: new_sub})


def _replace_config_value(obj: Any, path: list[str], value: str, label: str) -> Any:
    if not path:
        raise ValueError(f"unknown config field: {label}")
    key = path[0]
    if not hasattr(obj, key):
        raise ValueError(f"unknown config field: {label}")
    if len(path) > 1:
        child = getattr(obj, key)
        if not dataclasses.is_dataclass(child):
            raise ValueError(f"unknown config field: {label}")
        return dataclasses.replace(
            obj,
            **{key: _replace_config_value(child, path[1:], value, label)},
        )

    field_type = None
    for f in dataclasses.fields(obj):
        if f.name == key:
            field_type = f.type
            break
    if field_type is None:
        raise ValueError(f"unknown config field: {label}")
    # Map semantic color-mode values to bool for observation.grayscale
    if label == "observation.grayscale":
        parsed = value.lower() in ("grayscale", "true", "1", "yes")
    elif field_type in ("bool", bool):
        parsed = value.lower() in ("true", "1", "yes")
    elif field_type in ("int", int):
        parsed = int(value)
    elif field_type in ("float", float):
        parsed = float(value)
    else:
        parsed = value
    return dataclasses.replace(obj, **{key: parsed})


def _apply_config_overrides(
    config: ProjectConfig,
    overrides: Any,
) -> ProjectConfig:
    """Apply UI run-scoped config overrides and re-run typed validation."""

    if overrides in (None, ""):
        overrides = []
    if not isinstance(overrides, list):
        raise TypeError("overrides must be a list")

    updated = config
    for item in overrides:
        if not isinstance(item, dict):
            raise TypeError("each override must be an object")
        group = item.get("group")
        key = item.get("key")
        value = item.get("value")
        if not isinstance(group, str) or not isinstance(key, str):
            raise ValueError("each override requires group and key")
        updated = _apply_config_override(updated, group, key, str(value))
    return ProjectConfig.from_dict(updated.to_dict())


def list_configs() -> dict[str, Any]:
    configs: list[dict[str, str]] = []
    games_dir = Path("configs/games")
    _skip_suffixes = ("_smoke", "_test", "_fixture")
    if games_dir.is_dir():
        for p in sorted(games_dir.glob("*.yaml")):
            if any(p.stem.endswith(s) for s in _skip_suffixes):
                continue
            configs.append({"path": str(p), "name": p.stem})
    return {"configs": configs}


def list_runs(state: UiState, include_smoke: bool = False) -> dict[str, Any]:
    import yaml

    output_dir = Path(state.config.experiment.output_dir)
    runs: list[dict[str, Any]] = []
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir() and (child / "config.yaml").exists():
                summary = _read_json(child / "metrics" / "train_summary.json")
                algo = summary.get("algorithm", "")
                is_smoke = (
                    (bool(summary) and algo == "linear_q")
                    or "smoke" in child.name.lower()
                )
                if is_smoke and not include_smoke:
                    continue
                experiment_name = ""
                try:
                    with (child / "config.yaml").open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    experiment_name = (cfg or {}).get("experiment", {}).get("name", "")
                    if not algo:
                        algo = (cfg or {}).get("agent", {}).get("algorithm", "")
                except Exception:
                    pass
                runs.append(
                    {
                        "name": child.name,
                        "experiment_name": experiment_name or child.name,
                        "steps": summary.get("steps"),
                        "episodes": summary.get("episodes"),
                        "best_score": summary.get("best_score"),
                        "algorithm": algo,
                        "checkpoints": _list_checkpoints(child, algorithm=algo),
                        "is_smoke": is_smoke,
                    }
                )
    return {"runs": runs}


def _list_checkpoints(run_dir: Path, algorithm: str | None = None) -> list[dict[str, str]]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    if algorithm in {"dqn", "pixel_dqn", "frozen_jepa_dqn", "joint_jepa_dqn"}:
        suffix = ".pt"
    elif algorithm == "linear_q":
        suffix = ".npz"
    else:
        suffix = None
    result: list[dict[str, str]] = []
    for p in sorted(ckpt_dir.iterdir()):
        if p.suffix not in (".pt", ".npz"):
            continue
        if suffix and p.suffix != suffix:
            continue
        label = p.stem.replace("_", " ")
        result.append({"file": p.name, "label": label})
    return result


_GAME_DESCRIPTIONS: dict[str, str] = {
    "breakout": "Destroy bricks with a bouncing ball. Don't let it fall!",
    "snake": "Eat food to grow longer. Avoid walls and yourself!",
    "asteroids": (
        "Shoot asteroids. Tough ones shrink when hit. Slimy ones spread moss "
        "that shields and regenerates. Collect power-ups!"
    ),
}


def _game_description(name: str) -> str:
    return _GAME_DESCRIPTIONS.get(name, "")


def _build_config_detail(config: ProjectConfig) -> list[dict[str, Any]]:
    # Field format: (key, value, tooltip, meta_dict)
    # meta_dict keys: type ("text","number","select","bool","readonly"),
    #   options, min, max, step, label (human-friendly name)
    gs_val = "grayscale" if config.observation.grayscale else "rgb"
    groups = [
        {
            "title": "experiment",
            "group_label": "Experiment",
            "fields": [
                ("name", config.experiment.name, "", {"type": "text", "label": "Name"}),
                (
                    "device",
                    config.experiment.device,
                    "Compute device. auto picks MPS on Apple Silicon, CUDA on Nvidia.",
                    {"type": "select", "options": ["cpu", "auto", "mps", "cuda"],
                     "label": "Device"},
                ),
                (
                    "precision",
                    config.experiment.precision,
                    "Floating-point precision. fp32 is required on MPS; fp16 only works on CUDA.",
                    {"type": "select", "options": ["fp32", "fp16", "bf16"], "label": "Precision"},
                ),
                (
                    "seed", config.experiment.seed,
                    "Random seed for reproducibility.",
                    {"type": "number", "min": 0, "label": "Seed"},
                ),
            ],
        },
        {
            "title": "observation",
            "group_label": "Observation",
            "fields": [
                (
                    "mode", config.observation.mode,
                    "How frames are captured from the browser.",
                    {"type": "select",
                     "options": ["screenshot", "canvas", "dom_assisted", "hybrid"],
                     "label": "Capture mode"},
                ),
                (
                    "grayscale", gs_val,
                    "Color mode. Grayscale reduces channels and speeds up training.",
                    {"type": "select", "options": ["grayscale", "rgb"],
                     "label": "Color mode"},
                ),
                (
                    "width", config.observation.width,
                    "Frame width in pixels. 84 is standard for DQN.",
                    {"type": "number", "min": 16, "label": "Width"},
                ),
                (
                    "height", config.observation.height,
                    "Frame height in pixels. 84 is standard for DQN.",
                    {"type": "number", "min": 16, "label": "Height"},
                ),
                (
                    "frame_stack", config.observation.frame_stack,
                    "Consecutive frames stacked for temporal context.",
                    {"type": "number", "min": 1, "label": "Frame stack"},
                ),
                (
                    "normalize", config.observation.normalize,
                    "Scale pixel values to [0, 1]. Recommended for neural net training.",
                    {"type": "bool", "label": "Normalize"},
                ),
            ],
        },
        {
            "title": "game",
            "group_label": "Game",
            "fields": [
                ("name", config.game.name, "",
                 {"type": "readonly", "label": "Name"}),
                ("url", config.game.url, "",
                 {"type": "readonly", "label": "URL"}),
                (
                    "fps", config.game.fps,
                    "Browser capture frame rate.",
                    {"type": "number", "min": 1, "max": 120, "label": "FPS"},
                ),
                (
                    "action_repeat", config.game.action_repeat,
                    "How many times each action is repeated.",
                    {"type": "number", "min": 1, "label": "Action repeat"},
                ),
                (
                    "max_steps_per_episode", config.game.max_steps_per_episode,
                    "Max steps before forcing episode end.",
                    {"type": "number", "min": 100, "label": "Max steps"},
                ),
                (
                    "headless", config.game.headless,
                    "Run browser without a visible window.",
                    {"type": "bool", "label": "Headless"},
                ),
            ],
        },
        {
            "title": "agent",
            "group_label": "Agent",
            "fields": [
                (
                    "algorithm",
                    config.agent.algorithm,
                    "Learning algorithm. dqn: pixel Q-learning. "
                    "frozen_jepa_dqn: pretrained encoder + Q-head. linear_q: smoke test.",
                    {"type": "select",
                     "options": ["dqn", "frozen_jepa_dqn", "joint_jepa_dqn",
                                  "linear_q"],
                     "label": "Algorithm"},
                ),
                (
                    "gamma",
                    config.agent.gamma,
                    "Discount factor (0-1). Higher = more patient.",
                    {"type": "number", "min": 0, "max": 1, "step": 0.001,
                     "label": "Gamma"},
                ),
                (
                    "n_step", config.agent.n_step,
                    "Multi-step return length. Higher speeds up credit "
                    "propagation but increases variance.",
                    {"type": "number", "min": 1, "label": "N-step"},
                ),
                (
                    "batch_size", config.agent.batch_size,
                    "Minibatch size. 32-64 for small, 128-256 for large.",
                    {"type": "number", "min": 1, "label": "Batch size"},
                ),
                (
                    "learning_starts",
                    config.agent.learning_starts,
                    "Random steps before training. Fills replay buffer.",
                    {"type": "number", "min": 0, "label": "Learning starts"},
                ),
                (
                    "train_every",
                    config.agent.train_every,
                    "Gradient update every N env steps.",
                    {"type": "number", "min": 1, "label": "Train every"},
                ),
                (
                    "target_update_interval",
                    config.agent.target_update_interval,
                    "Steps between target network updates.",
                    {"type": "number", "min": 1, "label": "Target update"},
                ),
                (
                    "optimizer.lr",
                    config.agent.optimizer.lr,
                    "Policy optimizer learning rate.",
                    {"type": "number", "min": 0, "step": 0.00001, "label": "LR"},
                ),
            ],
        },
        {
            "title": "exploration",
            "group_label": "Exploration",
            "fields": [
                (
                    "epsilon_start",
                    config.exploration.epsilon_start,
                    "Starting exploration rate. 1.0 = fully random.",
                    {"type": "number", "min": 0, "max": 1, "step": 0.01,
                     "label": "Epsilon start"},
                ),
                (
                    "epsilon_end",
                    config.exploration.epsilon_end,
                    "Minimum exploration rate after decay.",
                    {"type": "number", "min": 0, "max": 1, "step": 0.01,
                     "label": "Epsilon end"},
                ),
                (
                    "epsilon_decay_steps",
                    config.exploration.epsilon_decay_steps,
                    "Steps for epsilon linear decay.",
                    {"type": "number", "min": 1, "label": "Decay steps"},
                ),
            ],
        },
        {
            "title": "replay",
            "group_label": "Replay",
            "fields": [
                (
                    "capacity",
                    config.replay.capacity,
                    "Max transitions stored. 100k-1M typical.",
                    {"type": "number", "min": 100, "label": "Capacity"},
                ),
                (
                    "prioritized", config.replay.prioritized,
                    "Sample proportional to TD error.",
                    {"type": "bool", "label": "Prioritized"},
                ),
                (
                    "sequence_length",
                    config.replay.sequence_length,
                    "Consecutive frames per batch (JEPA). 8-32 typical.",
                    {"type": "number", "min": 1, "label": "Sequence length"},
                ),
            ],
        },
        {
            "title": "world_model",
            "group_label": "World Model",
            "fields": [
                (
                    "enabled", config.world_model.enabled,
                    "Enable JEPA pretraining. Required for frozen_jepa_dqn.",
                    {"type": "bool", "label": "Enabled"},
                ),
                (
                    "latent_dim",
                    config.world_model.latent_dim,
                    "Latent embedding dimension. 128-512 typical.",
                    {"type": "number", "min": 1, "label": "Latent dim"},
                ),
                (
                    "encoder", config.world_model.encoder.type,
                    "Vision encoder architecture.",
                    {"type": "readonly", "label": "Encoder"},
                ),
                (
                    "predictor", config.world_model.predictor.type,
                    "Predictor architecture.",
                    {"type": "readonly", "label": "Predictor"},
                ),
                (
                    "horizons",
                    list(config.world_model.predictor.horizons),
                    "Future steps the predictor is trained to forecast.",
                    {"type": "readonly", "label": "Horizons"},
                ),
                (
                    "loss", config.world_model.loss.prediction,
                    "Objective used to train the world model.",
                    {"type": "readonly", "label": "Loss"},
                ),
                (
                    "ema_tau",
                    f"{config.world_model.target_encoder.ema_tau_start}"
                    f"–{config.world_model.target_encoder.ema_tau_end}",
                    "EMA decay for the target encoder.",
                    {"type": "readonly", "label": "EMA tau"},
                ),
            ],
        },
        {
            "title": "training",
            "group_label": "Training",
            "fields": [
                (
                    "total_env_steps", config.training.total_env_steps,
                    "Total environment steps to train for.",
                    {"type": "number", "min": 1, "label": "Total steps"},
                ),
                (
                    "learning_starts", config.training.learning_starts,
                    "Random exploration before learning begins.",
                    {"type": "number", "min": 0,
                     "label": "Learning starts"},
                ),
                (
                    "train_every", config.training.train_every,
                    "Gradient update every N env steps.",
                    {"type": "number", "min": 1, "label": "Train every"},
                ),
                (
                    "eval_interval_steps",
                    config.training.eval_interval_steps,
                    "Run evaluation every N steps.",
                    {"type": "number", "min": 100,
                     "label": "Eval interval"},
                ),
                (
                    "checkpoint_interval_steps",
                    config.training.checkpoint_interval_steps,
                    "Save a checkpoint every N steps.",
                    {"type": "number", "min": 100,
                     "label": "Checkpoint interval"},
                ),
            ],
        },
    ]
    return groups


def _get_defaults(state: UiState) -> dict[str, Any]:
    return {
        "run_name": state.run_dir.name,
        "default_steps": state.default_steps,
        "learning_starts": state.learning_starts,
        "batch_size": state.batch_size,
        "dashboard_every": state.dashboard_every,
    }


def _build_model_info(config: ProjectConfig) -> dict[str, Any]:
    return {
        "algorithm": config.agent.algorithm,
        "observation_width": config.observation.width,
        "observation_height": config.observation.height,
        "grayscale": config.observation.grayscale,
        "frame_stack": config.observation.frame_stack,
        "latent_dim": config.world_model.latent_dim,
        "encoder_type": config.world_model.encoder.type,
        "encoder_channels": list(config.world_model.encoder.hidden_channels),
        "predictor_type": config.world_model.predictor.type,
        "predictor_depth": config.world_model.predictor.depth,
        "predictor_heads": config.world_model.predictor.num_heads,
        "predictor_hidden": config.world_model.predictor.hidden_dim,
        "q_hidden_dims": list(config.agent.q_network.hidden_dims),
        "q_dueling": config.agent.q_network.dueling,
        "gamma": config.agent.gamma,
        "batch_size": config.agent.batch_size,
        "learning_starts": config.agent.learning_starts,
        "train_every": config.agent.train_every,
        "target_update_interval": config.agent.target_update_interval,
        "agent_lr": config.agent.optimizer.lr,
        "epsilon_start": config.exploration.epsilon_start,
        "epsilon_end": config.exploration.epsilon_end,
        "epsilon_decay_steps": config.exploration.epsilon_decay_steps,
        "num_actions": config.actions.num_actions,
        "action_keys": list(config.actions.keys),
    }


def list_collected_datasets(state: UiState) -> dict[str, Any]:
    output_dir = Path(state.config.experiment.output_dir)
    datasets: list[dict[str, Any]] = []
    if output_dir.is_dir():
        for child in sorted(
            output_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            summary_path = child / "random_baseline_summary.json"
            if child.is_dir() and summary_path.exists():
                summary = _read_json(summary_path)
                events_path = child / "metrics" / "random_events.jsonl"
                total_steps = 0
                if events_path.exists():
                    with contextlib.suppress(Exception):
                        total_steps = sum(
                            1
                            for _ in events_path.open(encoding="utf-8")
                        )
                dir_size = 0
                try:
                    for f in child.rglob("*"):
                        if f.is_file():
                            dir_size += f.stat().st_size
                except Exception:  # noqa: BLE001
                    pass
                datasets.append(
                    {
                        "name": child.name,
                        "episodes": summary.get("total_episodes", 0),
                        "mean_score": summary.get("score_mean"),
                        "max_score": summary.get("score_max"),
                        "min_score": summary.get("score_min"),
                        "median_score": summary.get("score_median"),
                        "mean_length": summary.get("mean_episode_length"),
                        "total_steps": total_steps,
                        "size_bytes": dir_size,
                    }
                )
    return {"datasets": datasets}


def build_state_payload(state: UiState) -> dict[str, Any]:
    with state.lock:
        job = state.job
        eval_job = state.eval_job
        world_job = state.world_job
        collect_job = state.collect_job
        run_dir = job.run_dir if job is not None else state.run_dir
        base_config = state.config
    summary = _read_json(run_dir / "metrics" / "train_summary.json")
    events = _read_jsonl(run_dir / "metrics" / "train_events.jsonl")
    step_events = [event for event in events if event.get("type") == "step"]
    episode_events = [event for event in events if event.get("type") == "episode"]
    active_config = base_config
    snapshot = run_dir / "config.yaml"
    if snapshot.exists():
        with contextlib.suppress(Exception):
            active_config = load_config(snapshot)
    with state.lock:
        live_step_events = list(state.live_step_events)
        job_payload = None
        if job is not None:
            job_payload = {
                "status": job.status,
                "run_name": job.run_name,
                "run_dir": str(job.run_dir),
                "requested_steps": job.requested_steps,
                "error": job.error,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "running": job.thread.is_alive(),
            }
        eval_payload = None
        if eval_job is not None:
            eval_payload = {
                "status": eval_job.status,
                "run_name": eval_job.run_name,
                "checkpoint_name": eval_job.checkpoint_name,
                "running": eval_job.status == "running",
                "episode_count": eval_job.episode_count,
                "episodes_target": eval_job.episodes_target,
                "result": eval_job.result,
                "error": eval_job.error,
            }
        world_job_payload = None
        if world_job is not None:
            world_job_payload = {
                "status": world_job.status,
                "run_name": world_job.run_name,
                "run_dir": str(world_job.run_dir),
                "requested_steps": world_job.requested_steps,
                "error": world_job.error,
                "started_at": world_job.started_at,
                "completed_at": world_job.completed_at,
                "running": world_job.thread.is_alive(),
            }
        collect_job_payload = None
        if collect_job is not None:
            collect_job_payload = {
                "status": collect_job.status,
                "run_name": collect_job.run_name,
                "run_dir": str(collect_job.run_dir),
                "episodes_done": collect_job.episodes_done,
                "episodes_target": collect_job.episodes_target,
                "mean_score": collect_job.mean_score,
                "error": collect_job.error,
                "started_at": collect_job.started_at,
                "completed_at": collect_job.completed_at,
                "running": collect_job.status == "running",
            }

    if live_step_events:
        merged_by_step = {
            int(event["step"]): event for event in step_events if isinstance(event.get("step"), int)
        }
        for event in live_step_events:
            if isinstance(event.get("step"), int):
                merged_by_step[int(event["step"])] = event
        step_events = [merged_by_step[key] for key in sorted(merged_by_step)]

    latest_step = step_events[-1] if step_events else {}
    payload: dict[str, Any] = {
        "run": {
            "name": state.experiment,
            "dir": str(run_dir),
            "dashboard": str(run_dir / "dashboard.html"),
            "has_checkpoint": (
                (run_dir / "checkpoints" / "latest.pt").exists()
                or (run_dir / "checkpoints" / "latest.npz").exists()
            ),
            "checkpoints": _list_checkpoints(run_dir, algorithm=active_config.agent.algorithm),
        },
        "config": {
            "path": str(state.config_path),
            "game": active_config.game.name,
            "reset_key": active_config.game.reset_key,
            "action_keys": list(active_config.actions.keys),
            "actions": active_config.actions.num_actions,
            "observation": {
                "width": active_config.observation.width,
                "height": active_config.observation.height,
                "channels": active_config.observation.input_channels,
            },
        },
        "game_settings": [
            ("controls", ", ".join(active_config.actions.keys)),
            ("reset", active_config.game.reset_key or "reload"),
            ("description", _game_description(active_config.game.name)),
        ],
        "base_config_detail": _build_config_detail(base_config),
        "config_detail": _build_config_detail(active_config),
        "job": job_payload,
        "eval": eval_payload,
        "world_job": world_job_payload,
        "collect_job": collect_job_payload,
        "summary": summary,
        "latest_step": latest_step,
        "steps": step_events[-500:],
        "episodes": episode_events[-100:],
        "base_model_info": _build_model_info(base_config),
        "model_info": _build_model_info(active_config),
    }
    return payload




def _preprocess_canvas_frame(png_bytes: bytes, config: ProjectConfig) -> Any:
    """Decode a raw PNG (from canvas.toDataURL) and preprocess to (C, H, W) uint8 array."""
    import numpy as np
    from PIL import Image

    from jepa_rl.envs.wrappers import apply_crop, apply_normalize

    obs_cfg = config.observation
    image = Image.open(io.BytesIO(png_bytes))
    mode = "L" if obs_cfg.grayscale else "RGB"
    image = image.convert(mode)

    array = np.asarray(image, dtype=np.uint8)
    array = array[None, :, :] if obs_cfg.grayscale else array.transpose(2, 0, 1)

    array = apply_crop(
        array,
        top=obs_cfg.crop_top,
        bottom=obs_cfg.crop_bottom,
        left=obs_cfg.crop_left,
        right=obs_cfg.crop_right,
    )

    if array.shape[0] == 1:
        image = Image.fromarray(array[0], mode="L")
    else:
        image = Image.fromarray(array.transpose(1, 2, 0), mode="RGB")
    image = image.resize((obs_cfg.width, obs_cfg.height), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.uint8)
    array = array[None, :, :] if obs_cfg.grayscale else array.transpose(2, 0, 1)

    if obs_cfg.normalize:
        array = apply_normalize(array)

    return array  # (C, H, W)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

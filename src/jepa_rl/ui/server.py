# ruff: noqa: E501

import base64
import collections
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
    episode_count: int = 0
    scores: list = field(default_factory=list)
    status: str = "running"
    result: dict[str, Any] | None = None
    error: str | None = None
    model: Any = None
    device: Any = None
    frame_buffer: Any = None


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
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_index(self) -> None:
            dist = _ui_dist_dir()
            if dist is not None and (dist / "index.html").exists():
                self._send_file(dist / "index.html", "text/html; charset=utf-8")
            else:
                self._send_html(render_ui_html(state))

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
                learning_starts = int(body.get("learning_starts") or state.learning_starts)
                batch_size_value = body.get("batch_size", state.batch_size)
                batch_size = int(batch_size_value) if batch_size_value not in (None, "") else None
                dashboard_every = int(body.get("dashboard_every") or state.dashboard_every)
                run_dir = create_run_dir(state.config.experiment.output_dir, run_name)
                snapshot_config(state.config, run_dir / "config.yaml")
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
                            run_name,
                            steps,
                            learning_starts,
                            batch_size,
                            dashboard_every,
                            stop_event,
                        ),
                        daemon=True,
                    ),
                )
                state.experiment = run_name
                state.run_dir = run_dir
                state.job = job
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
                    self._send_json({"ok": False, "error": f"run config invalid: {exc}"}, status=400)
                    return
            algorithm = eval_config.agent.algorithm
            ckpt_name = body.get("checkpoint")
            if ckpt_name:
                checkpoint = eval_run_dir / "checkpoints" / ckpt_name
            else:
                suffix = ".pt" if algorithm == "dqn" else ".npz"
                checkpoint = eval_run_dir / "checkpoints" / f"latest{suffix}"
            if not checkpoint.exists():
                self._send_json({"ok": False, "error": f"checkpoint not found: {checkpoint.name}"}, status=404)
                return
            try:
                device = None
                if algorithm == "dqn":
                    from jepa_rl.models.device import resolve_torch_device
                    from jepa_rl.models.dqn import build_q_network
                    from jepa_rl.utils.checkpoint import load_torch_checkpoint

                    device = resolve_torch_device(eval_config.experiment.device)
                    net = build_q_network(eval_config, num_actions=eval_config.actions.num_actions).to(device)
                    load_torch_checkpoint(checkpoint, model=net, target_model=None, optimizer=None)
                    net.eval()
                    model = net
                elif algorithm == "linear_q":
                    from jepa_rl.training.simple_q import LinearQModel

                    model = LinearQModel.load(checkpoint)
                else:
                    self._send_json({"ok": False, "error": f"in-iframe eval not supported for {algorithm}"}, status=400)
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
                    model=model,
                    device=device,
                    frame_buffer=frame_buffer,
                )
            self._send_json({"ok": True})

        def _handle_eval_step(self) -> None:
            with state.lock:
                eval_job = state.eval_job
            if eval_job is None or eval_job.model is None:
                self._send_json({"ok": False, "error": "no eval session; call /api/eval/start first"}, status=400)
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
                self._send_json({"ok": True, "done": True, "complete": complete, "episode": ep_count})
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

                obs = np.concatenate(frames, axis=0)
                if algorithm == "dqn":
                    import torch

                    obs_t = torch.from_numpy(obs[None]).to(device)
                    with torch.no_grad():
                        action_idx = int(model(obs_t).argmax(dim=1).item())
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
                    action_idx = int(np.argmax(model.q_values(features)))
                action_key = eval_config.actions.keys[action_idx]
            except Exception as exc:  # noqa: BLE001
                with state.lock:
                    eval_job.status = "error"
                    eval_job.error = f"inference failed: {exc}"
                self._send_json({"ok": False, "error": f"inference failed: {exc}"}, status=500)
                return

            self._send_json({"ok": True, "done": False, "complete": False, "action": action_key})

        def _handle_eval_stop(self) -> None:
            with state.lock:
                state.eval_job = None
            self._send_json({"ok": True})

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
                    self._send_json({"ok": False, "error": "cannot switch config while training"}, status=409)
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
                    self._send_json({"ok": False, "error": "cannot update config while training"}, status=409)
                    return
                try:
                    new_config = state.config
                    for item in overrides:
                        group = str(item.get("group", ""))
                        key = str(item.get("key", ""))
                        value = str(item.get("value", ""))
                        new_config = _apply_config_override(new_config, group, key, value)
                    state.config = new_config
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
                self._send_json({"ok": False, "error": "name must be alphanumeric (dashes/underscores ok)"}, status=400)
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
                        self._send_json({"ok": False, "error": "cannot delete active training run"}, status=409)
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

        def _send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
            summary = _read_json(run_dir / "metrics" / "train_summary.json")
            self._send_json({"detail": detail, "summary": summary, "checkpoints": _list_checkpoints(run_dir), "run_dir": str(run_dir)})

    return TrainingUiHandler


def _training_worker(
    state: UiState,
    run_name: str,
    steps: int,
    learning_starts: int,
    batch_size: int | None,
    dashboard_every: int,
    stop_event: threading.Event,
) -> None:
    with state.lock:
        if state.job is not None:
            state.job.status = "running"
            job_run_dir = state.job.run_dir
    try:
        algorithm = state.config.agent.algorithm
        screenshot = job_run_dir / "frame.png"
        if algorithm == "dqn":
            from jepa_rl.training.pixel_dqn import train_dqn

            train_dqn(
                state.config,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=True,
                stop_event=stop_event,
                screenshot_path=screenshot,
            )
        else:
            train_linear_q(
                state.config,
                experiment=run_name,
                steps=steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                dashboard_every=dashboard_every,
                headless=True,
                stop_event=stop_event,
                screenshot_path=screenshot,
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


def _apply_config_override(config: ProjectConfig, group: str, key: str, value: str) -> ProjectConfig:
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
    if sub is None or not hasattr(sub, key):
        raise ValueError(f"unknown config field: {group}.{key}")
    field_type = None
    for f in dataclasses.fields(sub):
        if f.name == key:
            field_type = f.type
            break
    if field_type is None:
        raise ValueError(f"unknown config field: {group}.{key}")
    if field_type in ("bool", bool):
        parsed = value.lower() in ("true", "1", "yes")
    elif field_type in ("int", int):
        parsed = int(value)
    elif field_type in ("float", float):
        parsed = float(value)
    else:
        parsed = value
    new_sub = dataclasses.replace(sub, **{key: parsed})
    return dataclasses.replace(config, **{group: new_sub})


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
    output_dir = Path(state.config.experiment.output_dir)
    runs: list[dict[str, Any]] = []
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir() and (child / "config.yaml").exists():
                summary = _read_json(child / "metrics" / "train_summary.json")
                algo = summary.get("algorithm", "")
                is_smoke = algo == "linear_q" or "smoke" in child.name.lower()
                if is_smoke and not include_smoke:
                    continue
                runs.append({
                    "name": child.name,
                    "steps": summary.get("steps"),
                    "episodes": summary.get("episodes"),
                    "best_score": summary.get("best_score"),
                    "algorithm": algo,
                    "checkpoints": _list_checkpoints(child),
                    "is_smoke": is_smoke,
                })
    return {"runs": runs}


def _list_checkpoints(run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(p.name for p in ckpt_dir.iterdir() if p.suffix in (".pt", ".npz"))


def _build_config_detail(config: ProjectConfig) -> list[dict[str, Any]]:
    # Field format: (key, value, tooltip, meta_dict)
    # meta_dict keys: type ("text","number","select","bool","readonly"), options, min, max, step
    groups = [
        {
            "title": "experiment",
            "fields": [
                ("name", config.experiment.name, "Run identifier", {"type": "text"}),
                ("device", config.experiment.device, "Compute device", {"type": "select", "options": ["cpu", "auto", "mps", "cuda"]}),
                ("precision", config.experiment.precision, "Numeric precision", {"type": "select", "options": ["fp32", "fp16", "bf16"]}),
                ("seed", config.experiment.seed, "Random seed", {"type": "number", "min": 0}),
            ],
        },
        {
            "title": "agent",
            "fields": [
                ("algorithm", config.agent.algorithm, "RL algorithm", {"type": "select", "options": ["dqn", "linear_q"]}),
                ("gamma", config.agent.gamma, "Discount factor", {"type": "number", "min": 0, "max": 1, "step": 0.001}),
                ("n_step", config.agent.n_step, "N-step return", {"type": "number", "min": 1}),
                ("batch_size", config.agent.batch_size, "Batch size", {"type": "number", "min": 1}),
                ("learning_starts", config.agent.learning_starts, "Steps before learning", {"type": "number", "min": 0}),
                ("train_every", config.agent.train_every, "Train every N steps", {"type": "number", "min": 1}),
                ("target_update_interval", config.agent.target_update_interval, "Target sync interval", {"type": "number", "min": 1}),
            ],
        },
        {
            "title": "exploration",
            "fields": [
                ("epsilon_start", config.exploration.epsilon_start, "Initial epsilon", {"type": "number", "min": 0, "max": 1, "step": 0.01}),
                ("epsilon_end", config.exploration.epsilon_end, "Final epsilon", {"type": "number", "min": 0, "max": 1, "step": 0.01}),
                ("epsilon_decay_steps", config.exploration.epsilon_decay_steps, "Decay steps", {"type": "number", "min": 1}),
            ],
        },
        {
            "title": "replay",
            "fields": [
                ("capacity", config.replay.capacity, "Buffer capacity", {"type": "number", "min": 100}),
                ("prioritized", config.replay.prioritized, "Prioritized replay", {"type": "bool"}),
                ("sequence_length", config.replay.sequence_length, "Sequence length", {"type": "number", "min": 1}),
            ],
        },
        {
            "title": "world_model",
            "fields": [
                ("enabled", config.world_model.enabled, "Use JEPA world model", {"type": "bool"}),
                ("latent_dim", config.world_model.latent_dim, "Latent dim", {"type": "number", "min": 1}),
                ("encoder", config.world_model.encoder.type, "Encoder", {"type": "readonly"}),
                ("predictor", config.world_model.predictor.type, "Predictor", {"type": "readonly"}),
                ("horizons", list(config.world_model.predictor.horizons), "Horizons", {"type": "readonly"}),
                ("loss", config.world_model.loss.prediction, "Loss function", {"type": "readonly"}),
                ("ema_tau", f"{config.world_model.target_encoder.ema_tau_start}–{config.world_model.target_encoder.ema_tau_end}", "EMA tau schedule", {"type": "readonly"}),
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


def build_state_payload(state: UiState) -> dict[str, Any]:
    with state.lock:
        job = state.job
        eval_job = state.eval_job
        run_dir = job.run_dir if job is not None else state.run_dir
    summary = _read_json(run_dir / "metrics" / "train_summary.json")
    events = _read_jsonl(run_dir / "metrics" / "train_events.jsonl")
    step_events = [event for event in events if event.get("type") == "step"]
    episode_events = [event for event in events if event.get("type") == "episode"]
    with state.lock:
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
                "running": eval_job.status == "running",
                "episode_count": eval_job.episode_count,
                "episodes_target": eval_job.episodes_target,
                "result": eval_job.result,
                "error": eval_job.error,
            }

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
            "checkpoints": _list_checkpoints(run_dir),
        },
        "config": {
            "path": str(state.config_path),
            "game": state.config.game.name,
            "reset_key": state.config.game.reset_key,
            "action_keys": list(state.config.actions.keys),
            "actions": state.config.actions.num_actions,
            "observation": {
                "width": state.config.observation.width,
                "height": state.config.observation.height,
                "channels": state.config.observation.input_channels,
            },
        },
        "game_settings": [
            ("resolution", f"{state.config.observation.width}x{state.config.observation.height}"),
            ("fps", state.config.game.fps),
            ("action repeat", state.config.game.action_repeat),
            ("obs mode", state.config.observation.mode),
            ("frame stack", state.config.observation.frame_stack),
            ("actions", ", ".join(state.config.actions.keys)),
            ("max steps", state.config.game.max_steps_per_episode),
        ],
        "config_detail": _build_config_detail(state.config),
        "job": job_payload,
        "eval": eval_payload,
        "summary": summary,
        "latest_step": latest_step,
        "steps": step_events[-500:],
        "episodes": episode_events[-100:],
    }
    return payload


def render_ui_html(state: UiState) -> str:
    run_name = state.run_dir.name
    default_steps = state.default_steps
    learning_starts = state.learning_starts
    batch_size = "" if state.batch_size is None else str(state.batch_size)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>jepa-rl — {run_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {{
        color-scheme: dark;
        --bg: #0f0e0c;
        --surface: #171613;
        --surface-2: #1e1d19;
        --border: #2a2822;
        --text: #ddd8cf;
        --muted: #7d7870;
        --accent: #c49152;
        --green: #5d9e5d;
        --red: #b9524c;
        --blue: #4e89ba;
        --yellow: #bfa03e;
      }}
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: var(--bg);
        color: var(--text);
        font-family: 'Outfit', system-ui, sans-serif;
        font-size: 13px;
        line-height: 1.45;
        overflow: hidden;
        height: 100vh;
      }}
      header {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 14px;
        border-bottom: 1px solid var(--border);
        flex-shrink: 0;
      }}
      .brand {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
      }}
      .sep {{
        width: 1px;
        height: 14px;
        background: var(--border);
      }}
      .run-name {{
        font-size: 13px;
        font-weight: 500;
      }}
      .header-status {{
        margin-left: auto;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: var(--muted);
      }}
      .controls-bar {{
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        padding: 5px 14px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
        flex-shrink: 0;
      }}
      .controls-bar .field {{
        display: flex;
        align-items: center;
        gap: 3px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}
      .controls-bar input, .controls-bar select {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 3px 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        width: 60px;
      }}
      .controls-bar select {{
        width: auto;
        min-width: 90px;
        max-width: 150px;
      }}
      .controls-bar button {{
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 3px 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
      }}
      .btn-accent {{
        color: var(--accent) !important;
        border-color: rgba(196,145,82,0.4) !important;
      }}
      .btn-danger {{
        color: var(--red) !important;
        border-color: rgba(185,82,76,0.4) !important;
      }}
      .hidden {{ display: none !important; }}
      .run-info-bar {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 10px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        flex-shrink: 0;
        font-size: 10px;
      }}
      .rib-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
      }}
      .rib-dot.idle {{ background: var(--muted); }}
      .rib-dot.running {{ background: #5d9e5d; animation: pulse 1.5s infinite; }}
      .rib-dot.evaluating {{ background: var(--accent); animation: pulse 1.5s infinite; }}
      .rib-dot.stopped {{ background: var(--muted); }}
      .rib-dot.error {{ background: var(--red); }}
      .rib-status {{
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 9px;
      }}
      .rib-detail {{
        color: var(--muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
      }}
      .cg-ckpt {{
        display: flex;
        align-items: center;
        gap: 5px;
      }}
      .cg-ckpt-label {{
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        flex-shrink: 0;
      }}
      .cg-ckpt select {{
        flex: 1;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 2px 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
      }}
      @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
      }}
      main {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        padding: 10px 14px;
        height: calc(100vh - 76px);
        overflow: hidden;
      }}
      .col {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-width: 0;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
      }}
      .col::-webkit-scrollbar {{ width: 4px; }}
      .col::-webkit-scrollbar-track {{ background: transparent; }}
      .col::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
      .game-section {{
        border: 1px solid var(--border);
        border-radius: 3px;
        overflow: hidden;
        background: #08080a;
        flex-shrink: 0;
      }}
      .game-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 10px;
        background: var(--surface);
        border-bottom: 1px solid var(--border);
      }}
      .gh-title {{
        font-size: 9px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }}
      .gh-status {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: var(--text);
      }}
      .gh-stat {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: var(--muted);
        margin-left: auto;
      }}
      .gh-stat strong {{ color: var(--text); }}
      .game-body {{
        width: 100%;
      }}
      .game-body iframe {{
        width: 100%;
        aspect-ratio: 4 / 3;
        border: none;
        display: block;
        overflow: hidden;
      }}
      .game-body iframe::-webkit-scrollbar {{ display: none; }}
      .train-frame {{
        width: 100%;
        aspect-ratio: 4 / 3;
        display: none;
        object-fit: contain;
        background: #08080a;
        image-rendering: auto;
      }}
      .game-section.training .train-frame {{ display: none; }}
      .game-section.training .game-body {{ display: block; }}
      .game-section.training .game-controls {{ display: none; }}
      .game-section.evaluating .game-controls {{ display: none; }}
      .game-controls {{
        display: flex;
        gap: 4px;
        padding: 5px 10px;
        background: var(--surface);
        border-top: 1px solid var(--border);
        justify-content: center;
      }}
      .game-controls button {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 3px 12px;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
      }}
      .game-controls button:active {{
        background: var(--surface-2);
      }}
      .game-controls .ctrl-label {{
        font-size: 9px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        align-self: center;
        margin-right: 4px;
      }}
      .settings-table {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 1px 10px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 5px 8px;
        font-size: 10px;
      }}
      .settings-table .sk {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 8px;
        font-weight: 500;
      }}
      .settings-table .sv {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: var(--text);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
      }}
      .metric {{
        background: var(--surface);
        padding: 5px 6px;
      }}
      .m-label {{
        font-size: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--muted);
        white-space: nowrap;
      }}
      .m-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        margin-top: 1px;
      }}
      .charts {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .chart-panel {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        background: var(--surface);
      }}
      .section-label {{
        font-size: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        padding: 4px 8px 0;
      }}
      .section-header {{
        font-size: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent);
        padding: 5px 8px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
        flex-shrink: 0;
      }}
      canvas {{
        display: block;
        width: 100%;
        height: 80px;
      }}
      .run-select {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 5px 8px;
        flex-shrink: 0;
      }}
      .run-select select {{
        width: 100%;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 4px 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
      }}
      .train-controls {{
        border: 1px solid var(--border);
        border-radius: 2px;
        background: var(--surface);
        flex-shrink: 0;
      }}
      .tc-row {{
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 5px 8px;
        flex-wrap: wrap;
      }}
      .tc-row .field {{
        display: flex;
        align-items: center;
        gap: 3px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}
      .tc-row input {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 3px 5px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        width: 52px;
      }}
      .tc-actions {{
        border-top: 1px solid var(--border);
        gap: 4px;
      }}
      .tc-actions button {{
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 3px 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
      }}
      .config-panel {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        background: var(--surface);
        flex-shrink: 0;
      }}
      .config-panel .section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .cfg-apply-btn {{
        background: transparent;
        border: 1px solid rgba(196,145,82,0.4);
        border-radius: 2px;
        color: var(--accent);
        padding: 1px 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 9px;
        font-weight: 500;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .cfg-input {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 1px 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        width: 100%;
        max-width: 160px;
        text-align: right;
      }}
      .cfg-input:focus {{
        outline: none;
        border-color: var(--accent);
      }}
      .cfg-select {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 1px 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        width: 100%;
        max-width: 160px;
        text-align: right;
        cursor: pointer;
        appearance: none;
        -webkit-appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%237a7a7a'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 4px center;
        padding-right: 14px;
      }}
      .cfg-select:focus {{
        outline: none;
        border-color: var(--accent);
      }}
      .cfg-checkbox {{
        accent-color: var(--accent);
        cursor: pointer;
        margin-left: auto;
      }}
      .cfg-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2px 0;
        gap: 6px;
      }}
      .cfg-key {{
        font-size: 9px;
        color: var(--muted);
        cursor: help;
        flex-shrink: 0;
        white-space: nowrap;
      }}
      .cfg-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        color: var(--text);
      }}
      .run-detail {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        background: var(--surface);
      }}
      .run-summary {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1px;
        background: var(--border);
        padding: 0;
      }}
      .run-summary .rs-item {{
        padding: 4px 8px;
        background: var(--surface);
      }}
      .run-summary .rs-label {{
        font-size: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--muted);
        font-weight: 500;
      }}
      .run-summary .rs-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
      }}
      .run-config {{
        max-height: none;
        overflow-y: auto;
        border-top: 1px solid var(--border);
        padding: 4px 8px;
      }}
      .run-config::-webkit-scrollbar {{ width: 3px; }}
      .run-config::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
      .cfg-group-title {{
        font-size: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent);
        margin-top: 6px;
        margin-bottom: 2px;
        cursor: pointer;
        user-select: none;
      }}
      .cfg-group-title::before {{
        content: '\\25BE ';
      }}
      .cfg-collapsed .cfg-group-title::before {{
        content: '\\25B8 ';
      }}
      .cfg-collapsed .cfg-fields {{
        display: none;
      }}
      .cfg-group:first-child .cfg-group-title {{ margin-top: 0; }}
      .checkpoint-bar {{
        display: flex;
        gap: 4px;
        padding: 4px 8px;
        border-top: 1px solid var(--border);
        align-items: center;
        font-size: 9px;
        color: var(--muted);
      }}
      .checkpoint-bar select {{
        flex: 1;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 2px 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
      }}
      .checkpoint-bar button {{
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 2px 8px;
        font-family: 'Outfit', sans-serif;
        font-size: 10px;
        cursor: pointer;
      }}
      .episodes-section {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
      }}
      th, td {{
        padding: 3px 6px;
        text-align: right;
        border-bottom: 1px solid var(--border);
      }}
      th:first-child, td:first-child {{ text-align: left; }}
      th {{
        background: var(--surface);
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        font-size: 8px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        position: sticky;
        top: 0;
        z-index: 1;
      }}
      .episodes-scroll {{
        overflow-y: auto;
        flex: 1;
      }}
      .episodes-scroll::-webkit-scrollbar {{ width: 4px; }}
      .episodes-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
      @media (max-width: 900px) {{
        main {{ grid-template-columns: 1fr 1fr; }}
        .col-right {{ display: none; }}
      }}
      @media (max-width: 600px) {{
        main {{ grid-template-columns: 1fr; height: auto; overflow-y: auto; }}
        body {{ overflow: auto; }}
        .col-left {{ display: none; }}
      }}
    </style>
  </head>
  <body>
    <header>
      <span class="brand" title="JEPA-RL: Joint-Embedding Predictive Architecture for Reinforcement Learning">jepa-rl</span>
      <span class="sep"></span>
      <span class="run-name" title="Current active run directory">{run_name}</span>
      <span class="header-status" id="headerStatus"></span>
    </header>
    <div class="controls-bar">
      <div class="field" title="Select a game configuration to load">game <select id="f-config" title="Available game configs in configs/games/"><option value="">loading...</option></select></div>
      <button onclick="createConfig()" title="Create a new game config from current settings" style="background:transparent;border:1px solid var(--border);border-radius:2px;color:var(--muted);padding:2px 8px;font-size:10px;cursor:pointer;font-family:'Outfit',sans-serif;">+ new config</button>
    </div>
    <main>
      <div class="col col-left">
        <div class="metrics" id="metrics" title="Live training metrics updated every refresh"></div>
        <div class="charts">
          <div class="chart-panel" title="Game score over time"><div class="section-label">score</div><canvas id="scoreChart"></canvas></div>
          <div class="chart-panel" title="Training loss (smooth L1) over time"><div class="section-label">loss</div><canvas id="lossChart"></canvas></div>
          <div class="chart-panel" title="Exploration rate (epsilon) over time"><div class="section-label">epsilon</div><canvas id="epsilonChart"></canvas></div>
          <div class="chart-panel" title="Temporal difference error over time"><div class="section-label">td error</div><canvas id="tdChart"></canvas></div>
        </div>
      </div>
      <div class="col col-center">
        <div class="game-section" id="gameSection">
          <div class="game-header">
            <span class="gh-title" id="gameTitle" title="Current view mode">game view</span>
            <span class="gh-status" id="gameStatus" title="Current game or training status">manual play</span>
            <span class="gh-stat" id="gameStats" title="Live score and lives from the game"></span>
            <button onclick="toggleGame()" title="Show/hide the game view" style="background:transparent;border:1px solid var(--border);border-radius:2px;color:var(--muted);padding:2px 8px;font-size:10px;cursor:pointer;font-family:'Outfit',sans-serif;">hide</button>
          </div>
          <div class="game-body">
            <iframe id="game" src="/game?embed" scrolling="no" style="overflow:hidden;" title="Game canvas"></iframe>
          </div>
          <img class="train-frame" id="trainFrame" alt="training view" title="Live training screenshot" />
          <div class="game-controls">
            <span class="ctrl-label">play</span>
            <button onclick="gameAction('left')" title="Send ArrowLeft key to the game">&#9664; left</button>
            <button onclick="gameAction('space')" title="Send Space key (serve ball)">serve</button>
            <button onclick="gameAction('right')" title="Send ArrowRight key to the game">right &#9654;</button>
            <button onclick="gameAction('reset')" title="Send R key to reset the game">reset</button>
          </div>
        </div>
        <div class="settings-table" id="gameSettings" title="Game configuration settings for the active config"></div>
      </div>
      <div class="col col-right">
        <div class="run-select">
          <select id="f-run" onchange="loadRunDetail()" title="Select a past training run to view its details. Leave unselected to configure a new run."><option value="">select run...</option></select>
          <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
            <label style="font-size:9px;color:var(--muted);cursor:pointer;display:flex;align-items:center;gap:3px;"><input type="checkbox" id="chkSmoke" onchange="loadRuns()" style="accent-color:var(--accent)"> show smoke</label>
            <button onclick="deleteSelectedRun()" id="btnDeleteRun" class="hidden" title="Delete the selected run and all its data" style="background:transparent;border:1px solid rgba(185,82,76,0.4);border-radius:2px;color:var(--red);padding:2px 8px;font-size:9px;cursor:pointer;font-family:'Outfit',sans-serif;margin-left:auto;">delete</button>
          </div>
        </div>
        <div class="run-info-bar" id="runInfoBar" title="Current run status and key metrics">
          <span class="rib-dot idle" id="ribDot"></span>
          <span class="rib-status" id="ribStatus">idle</span>
          <span class="rib-detail" id="ribDetail"></span>
        </div>
        <div class="train-controls" id="controlGroup">
          <div class="tc-row">
            <div class="field" title="Name for the new training run (creates runs/&lt;name&gt;/)">name <input id="f-run-name" type="text" value="ui_train" style="width:80px" title="Run name for new training"></div>
            <div class="field" title="Total environment steps for this training run">steps <input id="f-steps" type="number" value="{default_steps}" min="1" title="Number of game steps to train"></div>
            <div class="field" title="Steps before model updates begin (fills replay buffer first)">warmup <input id="f-ls" type="number" value="{learning_starts}" min="0" title="Warmup steps before learning starts"></div>
          </div>
          <div class="tc-row">
            <div class="field" title="Minibatch size for gradient updates (leave empty for config default)">batch <input id="f-bs" type="number" value="{batch_size}" min="1" placeholder="auto" title="Batch size override"></div>
            <div class="field" title="Rewrite dashboard every N steps">log&nbsp;every <input id="f-de" type="number" value="5" min="1" title="Dashboard write interval"></div>
          </div>
          <div class="tc-row cg-ckpt" id="ckptRow" style="display:none">
            <span class="cg-ckpt-label">checkpoint</span>
            <select id="f-checkpoint" title="Available model checkpoints for this run"></select>
          </div>
          <div class="tc-row tc-actions">
            <button onclick="startTraining()" id="btnTrain" class="btn-accent" title="Start training with the current config and parameters above">train</button>
            <button onclick="watchAiPlay()" id="btnWatch" class="btn-accent hidden" title="Watch the AI play using the selected checkpoint">watch AI play</button>
            <button onclick="stopTraining()" id="btnStop" class="btn-danger hidden" title="Stop the running training or evaluation job">stop</button>
          </div>
        </div>
        <div class="config-panel" id="configPanel">
          <div class="section-header" title="All config parameters for the active game. Edit values and they apply on next training run.">config &nbsp;<button onclick="applyConfigEdits()" class="cfg-apply-btn" title="Apply edited config values now">apply</button></div>
          <div class="run-config" id="runConfig"></div>
        </div>
        <section class="episodes-section">
          <div class="section-header" title="Recent training episodes with their scores">episodes</div>
          <div class="episodes-scroll">
            <table>
              <thead><tr><th>ep</th><th>step</th><th>return</th><th>score</th></tr></thead>
              <tbody id="episodes"></tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
    <script>
      async function api(path, payload) {{
        const res = await fetch(path, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload || {{}}),
        }});
        const data = await res.json();
        if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
        return data;
      }}

      async function startTraining() {{
        try {{
          const expName = (document.getElementById("f-run-name").value || "ui_train").trim();
          await api("/api/train/start", {{
            experiment: expName,
            steps: Number(document.getElementById("f-steps").value),
            learning_starts: Number(document.getElementById("f-ls").value),
            batch_size: document.getElementById("f-bs").value,
            dashboard_every: Number(document.getElementById("f-de").value) || 5,
          }});
        }} catch (e) {{ setStatus(e.message); }}
        refresh();
        setTimeout(loadRuns, 500);
      }}

      async function stopTraining() {{
        _stopAiPlay();
        try {{ await api("/api/train/stop"); }} catch (e) {{ setStatus(e.message); }}
        refresh();
      }}

      let _aiInterval = null;
      let _aiTargetEpisodes = 3;
      let _aiEpisodeCount = 0;

      async function watchAiPlay() {{
        const ckptSel = document.getElementById("f-checkpoint");
        if (!ckptSel || !ckptSel.value) {{ setStatus("select a checkpoint first"); return; }}
        const cg = document.getElementById("controlGroup");
        const episodes = 3;
        const payload = {{ episodes, checkpoint: ckptSel.value }};
        if (cg && cg.dataset.runDir) payload.run_dir = cg.dataset.runDir;
        try {{
          await api("/api/eval/start", payload);
        }} catch (e) {{ setStatus(e.message); return; }}
        _aiTargetEpisodes = episodes;
        _aiEpisodeCount = 0;
        // Reset the game so we start fresh
        const iframe = document.getElementById("game");
        if (iframe) iframe.src = "/game?embed&ts=" + Date.now();
        setTimeout(() => {{
          _sendEpisodeStartKey();
          _aiInterval = setInterval(_aiLoop, 150);
          _updateEvalUi();
          setStatus("AI playing live…");
        }}, 1500);
      }}

      async function _aiLoop() {{
        try {{
          const iframe = document.getElementById("game");
          if (!iframe) return;
          const doc = iframe.contentDocument || iframe.contentWindow.document;
          if (!doc || !doc.body) return;

          const doneEl = doc.querySelector("#status[data-state='done']");
          const isDone = !!doneEl;
          const scoreEl = doc.getElementById("score");
          const score = scoreEl ? (parseFloat(scoreEl.textContent) || 0) : 0;

          const gameCanvas = doc.getElementById("game");
          if (!gameCanvas || !gameCanvas.getContext) return;
          const b64 = gameCanvas.toDataURL("image/png").split(",")[1];

          const res = await fetch("/api/eval/step", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{frame: b64, done: isDone, score}})
          }});
          if (!res.ok) return;
          const data = await res.json();

          if (data.error) {{
            _stopAiPlay();
            setStatus("eval error: " + data.error);
            return;
          }}

          if (data.complete) {{
            _stopAiPlay();
            refresh();
            return;
          }}

          if (isDone) {{
            _aiEpisodeCount++;
            _updateEvalUi();
            // Reset game for next episode.
            setTimeout(() => {{
              iframe.src = "/game?embed&ts=" + Date.now();
              setTimeout(_sendEpisodeStartKey, 500);
            }}, 300);
            return;
          }}

          if (data.action && data.action !== "noop") {{
            _applyKeyAction(doc, data.action);
          }}
        }} catch(_) {{
          // Network hiccup — skip frame
        }}
      }}

      function _applyKeyAction(doc, actionStr) {{
        if (!actionStr || actionStr === "noop") return;
        const keyMap = {{"Space": " "}};
        const keys = actionStr.split("+").map(k => ({{key: keyMap[k] || k, code: k === "Space" ? "Space" : k}}));
        keys.forEach(({{key, code}}) => doc.dispatchEvent(new KeyboardEvent("keydown", {{key, code, bubbles: true}})));
        setTimeout(() => keys.slice().reverse().forEach(({{key, code}}) => doc.dispatchEvent(new KeyboardEvent("keyup", {{key, code, bubbles: true}}))), 80);
      }}

      function _sendEpisodeStartKey() {{
        const iframe = document.getElementById("game");
        if (!iframe) return false;
        const win = iframe.contentWindow;
        const doc = iframe.contentDocument || (win && win.document);
        if (!win || !doc || !doc.getElementById("game")) return false;
        const key = "{state.config.game.reset_key}";
        const normalized = key === "Space" ? {{key: " ", code: "Space"}} : {{key, code: key.length === 1 ? "Key" + key.toUpperCase() : key}};
        win.dispatchEvent(new KeyboardEvent("keydown", {{key: normalized.key, code: normalized.code, bubbles: true, cancelable: true}}));
        doc.dispatchEvent(new KeyboardEvent("keydown", {{key: normalized.key, code: normalized.code, bubbles: true, cancelable: true}}));
        setTimeout(() => {{
          win.dispatchEvent(new KeyboardEvent("keyup", {{key: normalized.key, code: normalized.code, bubbles: true, cancelable: true}}));
          doc.dispatchEvent(new KeyboardEvent("keyup", {{key: normalized.key, code: normalized.code, bubbles: true, cancelable: true}}));
        }}, 80);
        return true;
      }}

      function _stopAiPlay() {{
        if (_aiInterval !== null) {{ clearInterval(_aiInterval); _aiInterval = null; }}
        fetch("/api/eval/stop", {{method: "POST"}}).catch(() => {{}});
        _updateEvalUi();
      }}

      function _updateEvalUi() {{
        const gameSection = document.getElementById("gameSection");
        const gameTitleEl = document.getElementById("gameTitle");
        const gameStatusEl = document.getElementById("gameStatus");
        const isEval = _aiInterval !== null;
        if (gameSection) {{
          if (isEval) gameSection.classList.add("evaluating");
          else gameSection.classList.remove("evaluating");
        }}
        if (gameTitleEl && isEval) gameTitleEl.textContent = "AI playing";
        if (gameStatusEl && isEval) {{
          gameStatusEl.textContent = "AI playing live · ep " + (_aiEpisodeCount + 1) + "/" + _aiTargetEpisodes;
        }}
        toggleButtons(isEval);
      }}

      function resetGame() {{
        document.getElementById("game").src = "/game?embed&ts=" + Date.now();
      }}

      function toggleGame() {{
        const section = document.getElementById("gameSection");
        const body = section.querySelector(".game-body");
        const controls = section.querySelector(".game-controls");
        const btn = event.target;
        if (body.style.display === "none") {{
          body.style.display = "";
          controls.style.display = "";
          btn.textContent = "hide";
        }} else {{
          body.style.display = "none";
          controls.style.display = "none";
          btn.textContent = "show";
        }}
      }}

      function gameAction(action) {{
        const iframe = document.getElementById("game");
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        if (!doc) return;
        if (action === "left") {{
          const kd = new KeyboardEvent("keydown", {{ key: "ArrowLeft", code: "ArrowLeft", bubbles: true }});
          doc.dispatchEvent(kd); window.dispatchEvent(kd);
          setTimeout(() => {{
            const ku = new KeyboardEvent("keyup", {{ key: "ArrowLeft", code: "ArrowLeft", bubbles: true }});
            doc.dispatchEvent(ku); window.dispatchEvent(ku);
          }}, 120);
        }} else if (action === "right") {{
          const kd = new KeyboardEvent("keydown", {{ key: "ArrowRight", code: "ArrowRight", bubbles: true }});
          doc.dispatchEvent(kd); window.dispatchEvent(kd);
          setTimeout(() => {{
            const ku = new KeyboardEvent("keyup", {{ key: "ArrowRight", code: "ArrowRight", bubbles: true }});
            doc.dispatchEvent(ku); window.dispatchEvent(ku);
          }}, 120);
        }} else if (action === "space") {{
          const kd = new KeyboardEvent("keydown", {{ key: " ", code: "Space", bubbles: true }});
          doc.dispatchEvent(kd); window.dispatchEvent(kd);
          setTimeout(() => {{
            const ku = new KeyboardEvent("keyup", {{ key: " ", code: "Space", bubbles: true }});
            doc.dispatchEvent(ku); window.dispatchEvent(ku);
          }}, 120);
        }} else if (action === "reset") {{
          const kd = new KeyboardEvent("keydown", {{ key: "r", code: "KeyR", bubbles: true }});
          doc.dispatchEvent(kd); window.dispatchEvent(kd);
        }}
      }}

      function pollGameStats() {{
        try {{
          const iframe = document.getElementById("game");
          const doc = iframe.contentDocument;
          if (!doc) return;
          const score = doc.getElementById("score");
          const lives = doc.getElementById("lives");
          const statsEl = document.getElementById("gameStats");
          if (score && lives && statsEl) {{
            statsEl.innerHTML = "score <strong>" + (score.textContent || "0") + "</strong> &nbsp; lives <strong>" + (lives.textContent || "3") + "</strong>";
          }}
        }} catch {{}}
      }}

      async function loadRuns() {{
        try {{
          const showSmoke = document.getElementById("chkSmoke") && document.getElementById("chkSmoke").checked;
          const res = await fetch("/api/runs" + (showSmoke ? "?smoke=true" : ""));
          const data = await res.json();
          const select = document.getElementById("f-run");
          const current = select.value;
          select.innerHTML = '<option value="">select run...</option>';
          (data.runs || []).forEach(r => {{
            const opt = document.createElement("option");
            opt.value = r.name;
            const info = [r.name];
            if (r.algorithm) info.push(r.algorithm);
            if (r.best_score != null) info.push("best:" + fmt(r.best_score));
            if (r.steps != null) info.push(r.steps + " steps");
            opt.textContent = info.join(" \\u00b7 ");
            select.appendChild(opt);
          }});
          if (current) select.value = current;
          if (!select.value && (data.runs || []).length) select.selectedIndex = 0;
          if (select.value) loadRunDetail();
        }} catch {{}}
      }}

      async function loadRunDetail() {{
        const name = document.getElementById("f-run").value;
        const ckptRow = document.getElementById("ckptRow");
        if (!name) {{
          if (ckptRow) ckptRow.style.display = "none";
          // Restore current game config
          const stRes = await fetch("/api/state");
          const stData = await stRes.json();
          renderEditableConfig(stData.config_detail);
          toggleButtons();
          return;
        }}
        try {{
          const res = await fetch("/api/run-detail?name=" + encodeURIComponent(name));
          const data = await res.json();
          const cg = document.getElementById("controlGroup");
          if (cg && data.run_dir) cg.dataset.runDir = data.run_dir;

          // Render run's config into config panel
          if (data.detail) renderEditableConfig(data.detail);

          // Checkpoints
          const ckptSel = document.getElementById("f-checkpoint");
          const prevCkpt = ckptSel ? ckptSel.value : "";
          ckptSel.innerHTML = "";
          (data.checkpoints || []).forEach(c => {{
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = c;
            ckptSel.appendChild(opt);
          }});
          if (prevCkpt) ckptSel.value = prevCkpt;
          if (ckptRow) ckptRow.style.display = (data.checkpoints || []).length ? "" : "none";
          toggleButtons();
        }} catch {{}}
      }}

      async function switchConfig() {{
        const select = document.getElementById("f-config");
        const path = select.value;
        if (!path) return;
        try {{
          const res = await api("/api/switch-config", {{ config: path }});
          _currentConfigPath = path;
          document.getElementById("game").src = "/game?embed&ts=" + Date.now();
          refresh();
          loadRuns();
        }} catch (e) {{ setStatus(e.message); }}
      }}

      let _currentConfigPath = "";
      async function loadConfigs() {{
        try {{
          const res = await fetch("/api/configs");
          const data = await res.json();
          const select = document.getElementById("f-config");
          select.innerHTML = "";
          (data.configs || []).forEach(c => {{
            const opt = document.createElement("option");
            opt.value = c.path;
            opt.textContent = c.name;
            select.appendChild(opt);
          }});
          // Auto-select the current config from the state API
          if (!_currentConfigPath) {{
            try {{
              const stRes = await fetch("/api/state");
              const stData = await stRes.json();
              if (stData.config && stData.config.path) _currentConfigPath = stData.config.path;
            }} catch {{}}
          }}
          if (_currentConfigPath) select.value = _currentConfigPath;
        }} catch {{}}
      }}

      async function refresh() {{
        try {{
          const res = await fetch("/api/state");
          render(await res.json());
        }} catch {{}}
      }}

      function render(state) {{
        const s = state.summary || {{}};
        const j = state.job || {{}};
        const l = state.latest_step || {{}};
        const status = j.status || "idle";

        const statusEl = document.getElementById("headerStatus");
        if (statusEl) {{
          statusEl.textContent = j.error ? "error: " + j.error : "";
        }}

        // Game status bar
        const gameStatusEl = document.getElementById("gameStatus");
        const gameTitleEl = document.getElementById("gameTitle");
        const gameSection = document.getElementById("gameSection");
        const e = state.eval || {{}};
        const isTraining = j.running || j.status === "running" || j.status === "starting";
        const isEvaluating = !isTraining && _aiInterval !== null;
        if (gameSection) {{
          if (isTraining) gameSection.classList.add("training");
          else gameSection.classList.remove("training");
          if (!isEvaluating) gameSection.classList.remove("evaluating");
        }}
        if (gameTitleEl) {{
          if (isTraining) gameTitleEl.textContent = "training";
          else if (isEvaluating) gameTitleEl.textContent = "AI playing";
          else if (e.result) gameTitleEl.textContent = "eval done";
          else gameTitleEl.textContent = "game view";
        }}
        if (gameStatusEl) {{
          if (isTraining) {{
            const ep = s.episodes || l.episode || 0;
            const step = s.steps || l.step || 0;
            gameStatusEl.textContent = "training \\u00b7 ep " + ep + " \\u00b7 step " + step;
          }} else if (isEvaluating) {{
            gameStatusEl.textContent = "AI playing live \\u00b7 ep " + (_aiEpisodeCount + 1) + "/" + _aiTargetEpisodes;
          }} else if (j.status === "completed" || j.status === "stopped") {{
            gameStatusEl.textContent = j.status + " \\u00b7 " + (s.steps || 0) + " steps";
          }} else if (j.status === "error") {{
            gameStatusEl.textContent = "error: " + (j.error || "");
          }} else if (e.status === "error") {{
            gameStatusEl.textContent = "eval error: " + (e.error || "");
          }} else if (e.result) {{
            gameStatusEl.textContent = "eval \\u00b7 mean " + fmt(e.result.mean_score) + " \\u00b7 best " + fmt(e.result.best_score);
          }} else {{
            gameStatusEl.textContent = "manual play";
          }}
        }}

        // Run info bar — single unified status
        const ribDot = document.getElementById("ribDot");
        const ribStatus = document.getElementById("ribStatus");
        const ribDetail = document.getElementById("ribDetail");
        if (ribDot) {{
          ribDot.className = "rib-dot";
          if (isTraining) ribDot.classList.add("running");
          else if (isEvaluating) ribDot.classList.add("evaluating");
          else if (j.status === "error" || e.status === "error") ribDot.classList.add("error");
          else if (j.status === "completed") ribDot.classList.add("stopped");
          else ribDot.classList.add("idle");
        }}
        if (ribStatus) {{
          if (isTraining) ribStatus.textContent = "training";
          else if (isEvaluating) ribStatus.textContent = "evaluating";
          else if (j.status === "completed") ribStatus.textContent = "completed";
          else if (j.status === "stopped") ribStatus.textContent = "stopped";
          else if (j.status === "error") ribStatus.textContent = "error";
          else ribStatus.textContent = "idle";
        }}
        if (ribDetail) {{
          const parts = [];
          if (s.algorithm) parts.push(s.algorithm);
          if (s.episodes) parts.push(s.episodes + " eps");
          if (s.best_score != null) parts.push("best:" + fmt(s.best_score));
          if (s.steps) parts.push(fmt(s.steps) + " steps");
          ribDetail.textContent = parts.join(" \\u00b7 ");
        }}

        // Toggle train/stop/watch buttons (eval is tracked client-side via _aiInterval)
        if (!isEvaluating) toggleButtons(isTraining);

        // Game settings
        const gs = state.game_settings || [];
        document.getElementById("gameSettings").innerHTML = gs.map(([k, v]) =>
          '<span class="sk">' + k + '</span><span class="sv">' + String(v) + '</span>'
        ).join("");

        // Editable config panel (always visible)
        renderEditableConfig(state.config_detail);

        // Metrics
        const items = [
          ["steps", s.steps || l.step],
          ["episodes", s.episodes],
          ["best score", s.best_score],
          ["loss", l.loss ?? s.mean_loss],
          ["td error", l.td_error ?? s.mean_td_error],
          ["epsilon", l.epsilon],
          ["updates", l.updates ?? s.update_count],
          ["replay", l.replay_size ?? s.replay_size],
          ["weight delta", l.weight_delta_norm ?? s.weight_delta_norm],
          ["target syncs", l.target_updates ?? s.target_update_count],
        ];
        if (state.eval && state.eval.result && state.eval.result.mean_score != null) {{
          items.push(["eval mean", state.eval.result.mean_score]);
        }}
        document.getElementById("metrics").innerHTML = items.map(([label, value]) =>
          '<div class="metric"><div class="m-label">' + label + '</div><div class="m-val">' + fmt(value) + '</div></div>'
        ).join("");

        const steps = state.steps || [];
        drawChart("scoreChart", steps.map(e => [e.step, e.score]), "#5d9e5d");
        drawChart("lossChart", steps.filter(e => e.loss != null).map(e => [e.step, e.loss]), "#4e89ba");
        drawChart("epsilonChart", steps.map(e => [e.step, e.epsilon]), "#bfa03e");
        drawChart("tdChart", steps.filter(e => e.td_error != null).map(e => [e.step, e.td_error]), "#b9524c");

        const eps = (state.episodes || []).slice(-20).reverse();
        document.getElementById("episodes").innerHTML = eps.map(e =>
          "<tr><td>" + (e.episode ?? "") + "</td><td>" + (e.step ?? "") + "</td><td>" + fmt(e.return) + "</td><td>" + fmt(e.score) + "</td></tr>"
        ).join("");
      }}

      function renderEditableConfig(groups) {{
        if (!groups || !groups.length) return;
        const el = document.getElementById("runConfig");
        el.innerHTML = groups.map(g => {{
          const collapsed = g.collapsed || false;
          let html = '<div class="cfg-group' + (collapsed ? ' cfg-collapsed' : '') + '">';
          html += '<div class="cfg-group-title">' + g.title + '</div>';
          html += '<div class="cfg-fields">';
          g.fields.forEach(f => {{
            const key = f[0];
            const val = f[1];
            const tip = f[2] || '';
            const meta = f[3] || {{}};
            const ftype = meta.type || 'text';

            if (ftype === 'readonly') {{
              html += '<div class="cfg-row"><span class="cfg-key" title="' + tip + '">' + key + '</span><span class="cfg-val">' + String(val) + '</span></div>';
              return;
            }}
            if (ftype === 'bool') {{
              const checked = val ? ' checked' : '';
              html += '<div class="cfg-row"><span class="cfg-key" title="' + tip + '">' + key + '</span>' +
                '<input type="checkbox" class="cfg-checkbox" data-group="' + g.title + '" data-key="' + key + '" data-type="bool"' + checked + ' title="' + tip + '"></div>';
              return;
            }}
            if (ftype === 'select') {{
              const opts = (meta.options || []).map(o => {{
                const sel = (String(o) === String(val)) ? ' selected' : '';
                return '<option value="' + o + '"' + sel + '>' + o + '</option>';
              }}).join('');
              html += '<div class="cfg-row"><span class="cfg-key" title="' + tip + '">' + key + '</span>' +
                '<select class="cfg-select" data-group="' + g.title + '" data-key="' + key + '" data-type="select" title="' + tip + '">' + opts + '</select></div>';
              return;
            }}
            // number or text
            const numAttrs = ftype === 'number' ? [
              meta.min !== undefined ? ' min="' + meta.min + '"' : '',
              meta.max !== undefined ? ' max="' + meta.max + '"' : '',
              meta.step !== undefined ? ' step="' + meta.step + '"' : ' step="1"',
              ' data-type="number"'
            ].join('') : ' data-type="text"';
            html += '<div class="cfg-row"><span class="cfg-key" title="' + tip + '">' + key + '</span>' +
              '<input class="cfg-input" type="' + (ftype === 'number' ? 'number' : 'text') + '"' + numAttrs +
              ' data-group="' + g.title + '" data-key="' + key + '" value="' + String(val).replace(/"/g, '&quot;') + '" title="' + tip + '"></div>';
          }});
          html += '</div></div>';
          return html;
        }}).join("");
      }}

      async function applyConfigEdits() {{
        const overrides = [];
        document.querySelectorAll('#runConfig .cfg-input, #runConfig .cfg-select, #runConfig .cfg-checkbox').forEach(el => {{
          let value;
          if (el.dataset.type === 'bool') {{
            value = el.checked ? 'true' : 'false';
          }} else {{
            value = el.value;
          }}
          overrides.push({{
            group: el.dataset.group,
            key: el.dataset.key,
            value: value,
          }});
        }});
        if (!overrides.length) return;
        try {{
          await api("/api/update-config", {{ overrides }});
          setStatus("config applied");
          refresh();
        }} catch (e) {{ setStatus(e.message); }}
      }}

      function setStatus(msg) {{
        const el = document.getElementById("headerStatus");
        if (el) el.textContent = msg;
      }}

      function toggleButtons(busy) {{
        const btnTrain = document.getElementById("btnTrain");
        const btnStop = document.getElementById("btnStop");
        const btnWatch = document.getElementById("btnWatch");
        const btnDeleteRun = document.getElementById("btnDeleteRun");
        const ckptSel = document.getElementById("f-checkpoint");
        const hasCkpt = ckptSel && ckptSel.options.length > 0;
        const runSelected = document.getElementById("f-run").value !== "";
        if (btnTrain) btnTrain.classList.toggle("hidden", !!busy);
        if (btnStop) btnStop.classList.toggle("hidden", !busy);
        if (btnWatch) btnWatch.classList.toggle("hidden", !!busy || !hasCkpt);
        if (btnDeleteRun) btnDeleteRun.classList.toggle("hidden", !runSelected || !!busy);
      }}

      async function createConfig() {{
        const name = prompt("New config name (alphanumeric, dashes ok):");
        if (!name) return;
        try {{
          await api("/api/create-config", {{ name: name.trim() }});
          loadConfigs();
          setStatus("config created: " + name);
        }} catch (e) {{ setStatus(e.message); }}
      }}

      async function deleteSelectedRun() {{
        const sel = document.getElementById("f-run");
        const name = sel.value;
        if (!name) return;
        if (!confirm("Delete run '" + name + "' and all its data?")) return;
        try {{
          await api("/api/delete-run", {{ name }});
          sel.value = "";
          loadRunDetail();
          loadRuns();
          setStatus("run deleted: " + name);
        }} catch (e) {{ setStatus(e.message); }}
      }}

      function fmt(v) {{
        if (v === null || v === undefined || Number.isNaN(v)) return "\\u2014";
        if (typeof v === "number") {{
          if (Math.abs(v) >= 1000) return v.toFixed(0);
          if (Math.abs(v) >= 10) return v.toFixed(2);
          return v.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
        }}
        return String(v);
      }}

      function hexRgba(hex, a) {{
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return "rgba(" + r + "," + g + "," + b + "," + a + ")";
      }}

      function drawChart(id, points, color) {{
        const canvas = document.getElementById(id);
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        const w = rect.width;
        const h = rect.height;
        ctx.clearRect(0, 0, w, h);

        if (!points.length) {{
          ctx.fillStyle = "#7d7870";
          ctx.font = "11px 'IBM Plex Mono', monospace";
          ctx.textAlign = "center";
          ctx.fillText("no data", w / 2, h / 2 + 4);
          return;
        }}

        const xs = points.map(p => p[0]);
        const ys = points.map(p => p[1]).filter(Number.isFinite);
        let minY = Math.min(...ys);
        let maxY = Math.max(...ys);
        if (!Number.isFinite(minY)) return;
        if (maxY - minY < 1e-9) {{ minY -= 1; maxY += 1; }}
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);

        const px = (x) => ((x - minX) / Math.max(1, maxX - minX)) * w;
        const py = (y) => h - ((y - minY) / (maxY - minY)) * h;

        ctx.beginPath();
        points.forEach(([x, y], i) => {{
          if (i === 0) ctx.moveTo(px(x), py(y));
          else ctx.lineTo(px(x), py(y));
        }});
        ctx.lineTo(px(xs[xs.length - 1]), h);
        ctx.lineTo(px(xs[0]), h);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, hexRgba(color, 0.15));
        grad.addColorStop(1, hexRgba(color, 0.01));
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        points.forEach(([x, y], i) => {{
          if (i === 0) ctx.moveTo(px(x), py(y));
          else ctx.lineTo(px(x), py(y));
        }});
        ctx.stroke();

        const last = points[points.length - 1];
        ctx.beginPath();
        ctx.arc(px(last[0]), py(last[1]), 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = "#7d7870";
        ctx.font = "9px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(fmt(maxY), 3, 9);
        ctx.textAlign = "right";
        ctx.fillText(fmt(minY), w - 3, h - 3);
      }}

      function pollFrame() {{
        const section = document.getElementById("gameSection");
        if (!section || !section.classList.contains("training")) return;
        const img = document.getElementById("trainFrame");
        if (!img) return;
        img.src = "/api/frame?ts=" + Date.now();
      }}

      // Wire config dropdown to switchConfig
      document.getElementById("f-config").addEventListener("change", switchConfig);

      // Init
      refresh();
      loadRuns();
      loadConfigs();
      // Toggle config group collapse via event delegation (bind once)
      document.getElementById("runConfig").addEventListener("click", function(e) {{
        if (e.target.classList.contains("cfg-group-title")) {{
          e.target.parentElement.classList.toggle("cfg-collapsed");
        }}
      }});
      setInterval(refresh, 1000);
      setInterval(pollGameStats, 500);
      setInterval(pollFrame, 500);
    </script>
  </body>
</html>
"""


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

    array = apply_crop(array, top=obs_cfg.crop_top, bottom=obs_cfg.crop_bottom, left=obs_cfg.crop_left, right=obs_cfg.crop_right)

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

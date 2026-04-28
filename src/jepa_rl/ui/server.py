# ruff: noqa: E501

import json
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
from jepa_rl.training.simple_q import evaluate_linear_q, train_linear_q
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
    last_eval: dict[str, Any] | None = None
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


def _make_handler(state: UiState) -> type[BaseHTTPRequestHandler]:
    class TrainingUiHandler(BaseHTTPRequestHandler):
        server_version = "JepaRlTrainingUi/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_ui_html(state))
            elif parsed.path == "/game":
                self._send_game()
            elif parsed.path == "/api/state":
                self._send_json(build_state_payload(state))
            elif parsed.path == "/api/runs":
                self._send_json(list_runs(state))
            elif parsed.path == "/api/configs":
                self._send_json(list_configs())
            elif parsed.path.startswith("/api/run-detail"):
                qs = parse_qs(urlparse(self.path).query)
                self._send_run_detail(qs)
            elif parsed.path == "/api/frame":
                self._send_frame()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/train/start":
                self._handle_train_start()
            elif parsed.path == "/api/train/stop":
                self._handle_train_stop()
            elif parsed.path == "/api/eval":
                self._handle_eval()
            elif parsed.path == "/api/switch-config":
                self._handle_switch_config()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

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

        def _handle_eval(self) -> None:
            body = self._read_json_body()
            episodes = int(body.get("episodes") or 1)
            algorithm = state.config.agent.algorithm
            ckpt_name = body.get("checkpoint")
            if ckpt_name:
                checkpoint = state.run_dir / "checkpoints" / ckpt_name
            elif algorithm == "dqn":
                checkpoint = state.run_dir / "checkpoints" / "latest.pt"
            else:
                checkpoint = state.run_dir / "checkpoints" / "latest.npz"
            if not checkpoint.exists():
                self._send_json({"ok": False, "error": "checkpoint does not exist"}, status=404)
                return
            try:
                if algorithm == "dqn":
                    from jepa_rl.training.pixel_dqn import evaluate_dqn
                    result = evaluate_dqn(
                        state.config,
                        checkpoint=checkpoint,
                        episodes=episodes,
                        headless=True,
                        run_dir=state.run_dir,
                    )
                else:
                    result = evaluate_linear_q(
                        state.config,
                        checkpoint=checkpoint,
                        episodes=episodes,
                        headless=True,
                        run_dir=state.run_dir,
                    )
            except Exception as exc:  # noqa: BLE001 - convert local UI failures to JSON.
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            with state.lock:
                state.last_eval = result
            self._send_json({"ok": True, "result": result})

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
            self._send_json({"detail": detail, "summary": summary, "checkpoints": _list_checkpoints(run_dir)})

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


def list_configs() -> dict[str, Any]:
    configs: list[dict[str, str]] = []
    games_dir = Path("configs/games")
    if games_dir.is_dir():
        for p in sorted(games_dir.glob("*.yaml")):
            configs.append({"path": str(p), "name": p.stem})
    return {"configs": configs}


def list_runs(state: UiState) -> dict[str, Any]:
    output_dir = Path(state.config.experiment.output_dir)
    runs: list[dict[str, Any]] = []
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir() and (child / "config.yaml").exists():
                summary = _read_json(child / "metrics" / "train_summary.json")
                runs.append({
                    "name": child.name,
                    "steps": summary.get("steps"),
                    "episodes": summary.get("episodes"),
                    "best_score": summary.get("best_score"),
                    "algorithm": summary.get("algorithm"),
                    "checkpoints": _list_checkpoints(child),
                })
    return {"runs": runs}


def _list_checkpoints(run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(p.name for p in ckpt_dir.iterdir() if p.suffix in (".pt", ".npz"))


def _build_config_detail(config: ProjectConfig) -> list[dict[str, Any]]:
    groups = [
        {
            "title": "experiment",
            "fields": [
                ("name", config.experiment.name, "Run identifier"),
                ("seed", config.experiment.seed, "Random seed for reproducibility"),
                ("device", config.experiment.device, "Compute device: cpu, auto, mps, cuda"),
                ("precision", config.experiment.precision, "Numeric precision: fp32, fp16, bf16"),
                ("output_dir", config.experiment.output_dir, "Directory for run artifacts"),
            ],
        },
        {
            "title": "game",
            "fields": [
                ("name", config.game.name, "Game identifier"),
                ("url", config.game.url, "Game URL or file path"),
                ("browser", config.game.browser, "Browser engine: chromium, chrome"),
                ("fps", config.game.fps, "Frames per second for screenshots"),
                ("action_repeat", config.game.action_repeat, "Repeat each action N times"),
                ("max_steps", config.game.max_steps_per_episode, "Max steps before forced reset"),
            ],
        },
        {
            "title": "observation",
            "fields": [
                ("mode", config.observation.mode, "Capture method: screenshot, canvas, dom_assisted, hybrid"),
                ("size", f"{config.observation.width}x{config.observation.height}", "Obs resolution in pixels"),
                ("grayscale", config.observation.grayscale, "Convert to single channel"),
                ("frame_stack", config.observation.frame_stack, "Stack N consecutive frames together"),
                ("channels", config.observation.input_channels, "Total input channels (computed)"),
                ("normalize", config.observation.normalize, "Scale pixels to [0,1]"),
            ],
        },
        {
            "title": "actions",
            "fields": [
                ("type", config.actions.type, "Action space type"),
                ("keys", ", ".join(config.actions.keys), "Mapped keyboard keys"),
                ("num_actions", config.actions.num_actions, "Discrete action count"),
            ],
        },
        {
            "title": "reward",
            "fields": [
                ("type", config.reward.type, "Reward function: score_delta"),
                ("score_reader", config.reward.score_reader, "How to read score: dom, ocr, javascript"),
                ("score_selector", config.reward.score_selector or "auto", "CSS selector for DOM score element"),
                ("survival_bonus", config.reward.survival_bonus, "Reward per step survived"),
                ("idle_penalty", config.reward.idle_penalty, "Penalty for no score change"),
                ("clip_rewards", config.reward.clip_rewards, "Clip rewards to [-1, 1]"),
            ],
        },
        {
            "title": "agent",
            "fields": [
                ("algorithm", config.agent.algorithm, "RL algorithm: dqn, linear_q"),
                ("gamma", config.agent.gamma, "Discount factor for future rewards"),
                ("n_step", config.agent.n_step, "N-step return for TD learning"),
                ("batch_size", config.agent.batch_size, "Minibatch size for gradient updates"),
                ("target_sync", config.agent.target_update_interval, "Steps between target network syncs"),
                ("learning_starts", config.agent.learning_starts, "Env steps before first update"),
                ("train_every", config.agent.train_every, "Update every N env steps"),
            ],
        },
        {
            "title": "exploration",
            "fields": [
                ("type", config.exploration.type, "Strategy: epsilon_greedy"),
                ("epsilon_start", config.exploration.epsilon_start, "Initial exploration rate"),
                ("epsilon_end", config.exploration.epsilon_end, "Final exploration rate"),
                ("epsilon_decay", config.exploration.epsilon_decay_steps, "Steps for epsilon annealing"),
            ],
        },
        {
            "title": "replay",
            "fields": [
                ("capacity", config.replay.capacity, "Max experiences stored"),
                ("prioritized", config.replay.prioritized, "Use prioritized experience replay"),
                ("seq_length", config.replay.sequence_length, "Sequence length for JEPA sampling"),
            ],
        },
        {
            "title": "world model",
            "fields": [
                ("enabled", config.world_model.enabled, "Use JEPA world model"),
                ("latent_dim", config.world_model.latent_dim, "Latent representation dimension"),
                ("encoder", config.world_model.encoder.type, "Encoder architecture: conv, vit"),
                ("predictor", config.world_model.predictor.type, "Predictor: transformer, gru"),
                ("depth", config.world_model.predictor.depth, "Predictor layer depth"),
                ("horizons", list(config.world_model.predictor.horizons), "Prediction horizons in steps"),
                ("loss", config.world_model.loss.prediction, "Loss function: cosine_mse, mse"),
                ("ema_tau", f"{config.world_model.target_encoder.ema_tau_start}-{config.world_model.target_encoder.ema_tau_end}", "EMA momentum schedule"),
            ],
        },
    ]
    return groups


def build_state_payload(state: UiState) -> dict[str, Any]:
    with state.lock:
        job = state.job
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
        last_eval = state.last_eval

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
        "summary": summary,
        "latest_step": latest_step,
        "steps": step_events[-500:],
        "episodes": episode_events[-100:],
        "last_eval": last_eval,
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
        image-rendering: pixelated;
        image-rendering: crisp-edges;
      }}
      .game-section.training .train-frame {{ display: block; }}
      .game-section.training .game-body {{ display: none; }}
      .game-section.training .game-controls {{ display: none; }}
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
        max-height: 200px;
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
      }}
      .cfg-group:first-child .cfg-group-title {{ margin-top: 0; }}
      .cfg-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 1px 0;
      }}
      .cfg-key {{
        font-size: 9px;
        color: var(--muted);
        cursor: help;
      }}
      .cfg-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        color: var(--text);
      }}
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
      <span class="brand">jepa-rl</span>
      <span class="sep"></span>
      <span class="run-name">{run_name}</span>
      <span class="header-status" id="headerStatus"></span>
    </header>
    <div class="controls-bar">
      <div class="field">game <select id="f-config"><option value="">loading...</option></select></div>
    </div>
    <main>
      <div class="col col-left">
        <div class="metrics" id="metrics"></div>
        <div class="charts">
          <div class="chart-panel"><div class="section-label">score</div><canvas id="scoreChart"></canvas></div>
          <div class="chart-panel"><div class="section-label">loss</div><canvas id="lossChart"></canvas></div>
          <div class="chart-panel"><div class="section-label">epsilon</div><canvas id="epsilonChart"></canvas></div>
          <div class="chart-panel"><div class="section-label">td error</div><canvas id="tdChart"></canvas></div>
        </div>
      </div>
      <div class="col col-center">
        <div class="game-section" id="gameSection">
          <div class="game-header">
            <span class="gh-title" id="gameTitle">game view</span>
            <span class="gh-status" id="gameStatus">manual play</span>
            <span class="gh-stat" id="gameStats"></span>
            <button onclick="toggleGame()" style="background:transparent;border:1px solid var(--border);border-radius:2px;color:var(--muted);padding:2px 8px;font-size:10px;cursor:pointer;font-family:'Outfit',sans-serif;">hide</button>
          </div>
          <div class="game-body">
            <iframe id="game" src="/game?embed" scrolling="no" style="overflow:hidden;"></iframe>
          </div>
          <img class="train-frame" id="trainFrame" alt="training view" />
          <div class="game-controls">
            <span class="ctrl-label">play</span>
            <button onclick="gameAction('left')">&#9664; left</button>
            <button onclick="gameAction('space')">serve</button>
            <button onclick="gameAction('right')">right &#9654;</button>
            <button onclick="gameAction('reset')">reset</button>
          </div>
        </div>
        <div class="settings-table" id="gameSettings"></div>
      </div>
      <div class="col col-right">
        <div class="run-select">
          <select id="f-run" onchange="loadRunDetail()"><option value="">select run...</option></select>
        </div>
        <div class="run-detail" id="runDetail" style="display:none;">
          <div class="section-header">run info</div>
          <div class="run-summary" id="runSummary"></div>
          <div class="checkpoint-bar" id="checkpointBar">
            <span>checkpoint</span>
            <select id="f-checkpoint"></select>
            <button onclick="evalCheckpoint()">eval</button>
          </div>
          <div class="run-config" id="runConfig"></div>
        </div>
        <section class="episodes-section">
          <div class="section-header">episodes</div>
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
          const expName = document.getElementById("f-run").value || "ui_train";
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
        try {{ await api("/api/train/stop"); }} catch (e) {{ setStatus(e.message); }}
        refresh();
      }}

      async function runEval() {{
        const ckptSel = document.getElementById("f-checkpoint");
        const payload = {{ episodes: 1 }};
        if (ckptSel && ckptSel.value) payload.checkpoint = ckptSel.value;
        try {{
          const res = await api("/api/eval", payload);
          if (res.result) setStatus("eval done: mean " + fmt(res.result.mean_score) + " best " + fmt(res.result.best_score));
        }} catch (e) {{ setStatus(e.message); }}
        refresh();
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
          const res = await fetch("/api/runs");
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
        const panel = document.getElementById("runDetail");
        if (!name) {{ panel.style.display = "none"; return; }}
        try {{
          const res = await fetch("/api/run-detail?name=" + encodeURIComponent(name));
          const data = await res.json();
          panel.style.display = "";

          // Summary
          const s = data.summary || {{}};
          const rows = [
            ["algorithm", s.algorithm || "\\u2014"],
            ["steps", s.steps || "\\u2014"],
            ["episodes", s.episodes || "\\u2014"],
            ["best score", s.best_score != null ? fmt(s.best_score) : "\\u2014"],
            ["status", s.status || "\\u2014"],
            ["updates", s.update_count != null ? s.update_count : "\\u2014"],
          ];
          document.getElementById("runSummary").innerHTML = rows.map(([k, v]) =>
            '<div class="rs-item"><div class="rs-label">' + k + '</div><div class="rs-val">' + v + '</div></div>'
          ).join("");

          // Checkpoints
          const ckptSel = document.getElementById("f-checkpoint");
          const prevCkpt = ckptSel.value;
          ckptSel.innerHTML = "";
          (data.checkpoints || []).forEach(c => {{
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = c;
            ckptSel.appendChild(opt);
          }});
          if (prevCkpt) ckptSel.value = prevCkpt;

          // Config
          const configEl = document.getElementById("runConfig");
          if (data.detail && data.detail.length) {{
            configEl.innerHTML = data.detail.map(g =>
              '<div class="cfg-group">' +
              '<div class="cfg-group-title">' + g.title + '</div>' +
              g.fields.map(f =>
                '<div class="cfg-row"><span class="cfg-key" title="' + (f[2] || '') + '">' + f[0] + '</span><span class="cfg-val">' + String(f[1]) + '</span></div>'
              ).join("") +
              '</div>'
            ).join("");
          }} else {{
            configEl.innerHTML = '<div style="padding:6px;color:var(--muted);font-size:10px;">no config data</div>';
          }}
        }} catch {{}}
      }}

      async function evalCheckpoint() {{
        const ckptSel = document.getElementById("f-checkpoint");
        if (!ckptSel || !ckptSel.value) {{ setStatus("select a checkpoint first"); return; }}
        try {{
          const res = await api("/api/eval", {{ episodes: 1, checkpoint: ckptSel.value }});
          if (res.result) setStatus("eval " + ckptSel.value + ": mean " + fmt(res.result.mean_score) + " best " + fmt(res.result.best_score));
        }} catch (e) {{ setStatus(e.message); }}
        refresh();
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
          let txt = status === "idle" ? "" : status;
          if (j.error) txt += ": " + j.error;
          statusEl.textContent = txt;
        }}

        // Game status bar
        const gameStatusEl = document.getElementById("gameStatus");
        const gameTitleEl = document.getElementById("gameTitle");
        const gameSection = document.getElementById("gameSection");
        const isTraining = j.running || j.status === "running" || j.status === "starting";
        if (gameSection) {{
          if (isTraining) gameSection.classList.add("training");
          else gameSection.classList.remove("training");
        }}
        if (gameTitleEl) {{
          if (isTraining) gameTitleEl.textContent = "training";
          else if (state.last_eval) gameTitleEl.textContent = "evaluation";
          else gameTitleEl.textContent = "game view";
        }}
        if (gameStatusEl) {{
          if (isTraining) {{
            const ep = s.episodes || l.episode || 0;
            const step = s.steps || l.step || 0;
            gameStatusEl.textContent = "training \\u00b7 ep " + ep + " \\u00b7 step " + step;
          }} else if (j.status === "completed" || j.status === "stopped") {{
            gameStatusEl.textContent = j.status + " \\u00b7 " + (s.steps || 0) + " steps";
          }} else if (j.status === "error") {{
            gameStatusEl.textContent = "error";
          }} else if (state.last_eval) {{
            gameStatusEl.textContent = "eval \\u00b7 mean " + fmt(state.last_eval.mean_score);
          }} else {{
            gameStatusEl.textContent = "manual play";
          }}
        }}

        // Game settings
        const gs = state.game_settings || [];
        document.getElementById("gameSettings").innerHTML = gs.map(([k, v]) =>
          '<span class="sk">' + k + '</span><span class="sv">' + String(v) + '</span>'
        ).join("");

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
        if (state.last_eval && state.last_eval.mean_score != null) {{
          items.push(["eval mean", state.last_eval.mean_score]);
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

      function setStatus(msg) {{
        const el = document.getElementById("headerStatus");
        if (el) el.textContent = msg;
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
      setInterval(refresh, 1000);
      setInterval(pollGameStats, 500);
      setInterval(pollFrame, 500);
    </script>
  </body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

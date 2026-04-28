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
from jepa_rl.utils.dashboard import write_training_dashboard


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
        raise OSError("ui requires --config or --run")

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
            elif parsed.path == "/dashboard":
                dashboard = write_training_dashboard(state.run_dir)
                self._send_file(dashboard, "text/html; charset=utf-8")
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
            checkpoint = state.run_dir / "checkpoints" / "latest.npz"
            if not checkpoint.exists():
                self._send_json({"ok": False, "error": "checkpoint does not exist"}, status=404)
                return
            try:
                result = evaluate_linear_q(
                    state.config,
                    checkpoint=checkpoint,
                    episodes=episodes,
                    headless=True,
                )
            except Exception as exc:  # noqa: BLE001 - convert local UI failures to JSON.
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            with state.lock:
                state.last_eval = result
            self._send_json({"ok": True, "result": result})

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
    try:
        train_linear_q(
            state.config,
            experiment=run_name,
            steps=steps,
            learning_starts=learning_starts,
            batch_size=batch_size,
            dashboard_every=dashboard_every,
            headless=True,
            stop_event=stop_event,
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


def build_state_payload(state: UiState) -> dict[str, Any]:
    run_dir = state.run_dir
    summary = _read_json(run_dir / "metrics" / "train_summary.json")
    events = _read_jsonl(run_dir / "metrics" / "train_events.jsonl")
    step_events = [event for event in events if event.get("type") == "step"]
    episode_events = [event for event in events if event.get("type") == "episode"]
    with state.lock:
        job = state.job
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
            "has_checkpoint": (run_dir / "checkpoints" / "latest.npz").exists(),
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
    experiment = state.experiment
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
        font-size: 14px;
        line-height: 1.5;
      }}
      header {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 14px 24px;
        border-bottom: 1px solid var(--border);
      }}
      .brand {{
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
      }}
      .sep {{
        width: 1px;
        height: 16px;
        background: var(--border);
      }}
      .run-name {{
        font-size: 14px;
        font-weight: 500;
      }}
      .header-status {{
        margin-left: auto;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: var(--muted);
      }}
      .controls-bar {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        padding: 8px 24px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
      }}
      .controls-bar .field {{
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}
      .controls-bar input {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 4px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        width: 72px;
      }}
      .controls-bar .spacer {{ flex: 1; }}
      .controls-bar button {{
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 2px;
        color: var(--text);
        padding: 4px 12px;
        font-family: 'Outfit', sans-serif;
        font-size: 12px;
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
        max-width: 1120px;
        margin: 0 auto;
        padding: 20px 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }}
      .top-section {{
        display: flex;
        gap: 16px;
      }}
      .game-wrap {{
        flex: 0 0 280px;
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
        background: #08080a;
        aspect-ratio: 4 / 3;
      }}
      .game-wrap iframe {{
        width: 100%;
        height: 100%;
        border: none;
        display: block;
      }}
      .metrics-wrap {{
        flex: 1;
        min-width: 0;
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
      }}
      .metric {{
        background: var(--surface);
        padding: 10px 12px;
      }}
      .m-label {{
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}
      .m-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 16px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        margin-top: 2px;
      }}
      .charts {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
      }}
      .chart-panel {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
      }}
      .section-label {{
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        padding: 10px 12px 0;
      }}
      canvas {{
        display: block;
        width: 100%;
        height: 180px;
      }}
      .episodes-section {{
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
      }}
      .episodes-section .section-label {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
      }}
      th, td {{
        padding: 7px 12px;
        text-align: right;
        border-bottom: 1px solid var(--border);
      }}
      th:first-child, td:first-child {{ text-align: left; }}
      th {{
        background: var(--surface);
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--muted);
      }}
      @media (max-width: 860px) {{
        .top-section {{ flex-direction: column; }}
        .game-wrap {{ flex: none; width: 100%; }}
        .charts {{ grid-template-columns: 1fr; }}
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
      <div class="field">experiment <input id="f-exp" value="{experiment}"></div>
      <div class="field">steps <input id="f-steps" type="number" value="{default_steps}" min="1"></div>
      <div class="field">learn&nbsp;start <input id="f-ls" type="number" value="{learning_starts}" min="0"></div>
      <div class="field">batch <input id="f-bs" type="number" value="{batch_size}" min="1" placeholder="auto"></div>
      <div class="field">dash&nbsp;every <input id="f-de" type="number" value="5" min="1"></div>
      <div class="spacer"></div>
      <button onclick="startTraining()" class="btn-accent">start</button>
      <button onclick="stopTraining()" class="btn-danger">stop</button>
      <button onclick="runEval()">eval</button>
      <button onclick="resetGame()">reset</button>
    </div>
    <main>
      <div class="top-section">
        <div class="game-wrap">
          <iframe id="game" src="/game"></iframe>
        </div>
        <div class="metrics-wrap">
          <div class="metrics" id="metrics"></div>
        </div>
      </div>
      <section class="charts">
        <div class="chart-panel">
          <div class="section-label">score</div>
          <canvas id="scoreChart"></canvas>
        </div>
        <div class="chart-panel">
          <div class="section-label">loss</div>
          <canvas id="lossChart"></canvas>
        </div>
        <div class="chart-panel">
          <div class="section-label">epsilon</div>
          <canvas id="epsilonChart"></canvas>
        </div>
        <div class="chart-panel">
          <div class="section-label">td error</div>
          <canvas id="tdChart"></canvas>
        </div>
      </section>
      <section class="episodes-section">
        <div class="section-label">episodes</div>
        <table>
          <thead><tr><th>ep</th><th>step</th><th>return</th><th>score</th></tr></thead>
          <tbody id="episodes"></tbody>
        </table>
      </section>
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
        await api("/api/train/start", {{
          experiment: document.getElementById("f-exp").value,
          steps: Number(document.getElementById("f-steps").value),
          learning_starts: Number(document.getElementById("f-ls").value),
          batch_size: document.getElementById("f-bs").value,
          dashboard_every: Number(document.getElementById("f-de").value) || 5,
        }});
      }} catch (e) {{ setStatus(e.message); }}
      refresh();
    }}

    async function stopTraining() {{
      try {{ await api("/api/train/stop"); }} catch (e) {{ setStatus(e.message); }}
      refresh();
    }}

    async function runEval() {{
      try {{ await api("/api/eval", {{ episodes: 1 }}); }} catch (e) {{ setStatus(e.message); }}
      refresh();
    }}

    function resetGame() {{
      document.getElementById("game").src = "/game?ts=" + Date.now();
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
        if (statusEl) statusEl.textContent = status === "idle" ? "" : "status: " + status;

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

        ctx.strokeStyle = "#2a2822";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 4]);
        for (let i = 1; i <= 3; i++) {{
          const y = (i / 4) * h;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(w, y);
          ctx.stroke();
        }}
        ctx.setLineDash([]);

        if (!points.length) {{
          ctx.fillStyle = "#7d7870";
          ctx.font = "12px 'IBM Plex Mono', monospace";
          ctx.fillText("no data", 12, h / 2 + 4);
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
        grad.addColorStop(0, hexRgba(color, 0.09));
        grad.addColorStop(1, hexRgba(color, 0.02));
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = "round";
        ctx.beginPath();
        points.forEach(([x, y], i) => {{
          if (i === 0) ctx.moveTo(px(x), py(y));
          else ctx.lineTo(px(x), py(y));
        }});
        ctx.stroke();

        ctx.fillStyle = "#7d7870";
        ctx.font = "10px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(fmt(maxY), 4, 10);
        ctx.fillText(fmt(minY), 4, h - 4);
      }}

      refresh();
      setInterval(refresh, 1000);
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

# ruff: noqa: E501

import json
from pathlib import Path
from typing import Any


def write_training_dashboard(run_dir: str | Path) -> Path:
    """Write a self-contained HTML dashboard for a training run."""

    run_path = Path(run_dir)
    summary = _read_json(run_path / "metrics" / "train_summary.json")
    events = _read_jsonl(run_path / "metrics" / "train_events.jsonl")
    step_events = [event for event in events if event.get("type") == "step"]
    episode_events = [event for event in events if event.get("type") == "episode"]
    dashboard_path = run_path / "dashboard.html"
    dashboard_path.write_text(
        _render_dashboard_html(
            run_name=run_path.name,
            summary=summary,
            step_events=step_events,
            episode_events=episode_events,
        ),
        encoding="utf-8",
    )
    return dashboard_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _render_dashboard_html(
    *, run_name: str, summary: dict[str, Any], step_events: list[dict[str, Any]], episode_events: list[dict[str, Any]]
) -> str:
    payload = {
        "runName": run_name,
        "summary": summary,
        "stepEvents": step_events,
        "episodeEvents": episode_events,
    }
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>JEPA-RL Training Dashboard - {run_name}</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #0f1117;
        --panel: #181c25;
        --panel-2: #202634;
        --text: #f4f7fb;
        --muted: #a8b2c5;
        --line: #31394b;
        --green: #34d399;
        --blue: #60a5fa;
        --yellow: #fbbf24;
        --red: #fb7185;
      }}

      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}

      header {{
        padding: 22px 28px;
        border-bottom: 1px solid var(--line);
        background: #121722;
      }}

      h1 {{
        margin: 0 0 6px;
        font-size: 22px;
        line-height: 1.2;
      }}

      .subtle {{
        color: var(--muted);
        font-size: 13px;
      }}

      main {{
        display: grid;
        gap: 18px;
        padding: 20px 28px 28px;
      }}

      .cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
      }}

      .card, .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
      }}

      .card {{
        padding: 14px;
        min-height: 82px;
      }}

      .label {{
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
      }}

      .value {{
        font-variant-numeric: tabular-nums;
        font-size: 24px;
        font-weight: 750;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 18px;
      }}

      .panel {{
        min-height: 260px;
        padding: 14px;
      }}

      .panel h2 {{
        margin: 0 0 12px;
        font-size: 15px;
      }}

      canvas {{
        width: 100%;
        height: 210px;
        display: block;
        background: var(--panel-2);
        border-radius: 6px;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}

      th, td {{
        padding: 8px 6px;
        border-bottom: 1px solid var(--line);
        text-align: right;
        font-variant-numeric: tabular-nums;
      }}

      th:first-child, td:first-child {{ text-align: left; }}
    </style>
  </head>
  <body>
    <header>
      <h1>Training Dashboard: {run_name}</h1>
      <div class="subtle">Self-contained snapshot generated from metrics/train_events.jsonl and metrics/train_summary.json.</div>
    </header>
    <main>
      <section class="cards" id="cards"></section>
      <section class="grid">
        <div class="panel"><h2>Score</h2><canvas id="scoreChart"></canvas></div>
        <div class="panel"><h2>Loss</h2><canvas id="lossChart"></canvas></div>
        <div class="panel"><h2>Epsilon</h2><canvas id="epsilonChart"></canvas></div>
        <div class="panel"><h2>TD Error</h2><canvas id="tdChart"></canvas></div>
      </section>
      <section class="panel">
        <h2>Episodes</h2>
        <table>
          <thead><tr><th>Episode</th><th>Step</th><th>Return</th><th>Score</th></tr></thead>
          <tbody id="episodes"></tbody>
        </table>
      </section>
    </main>
    <script>
      const payload = {payload_json};
      const summary = payload.summary || {{}};
      const steps = payload.stepEvents || [];
      const episodes = payload.episodeEvents || [];

      const cards = [
        ["Steps", summary.steps],
        ["Episodes", summary.episodes],
        ["Best Score", summary.best_score],
        ["Mean Loss", summary.mean_loss],
        ["Mean TD Error", summary.mean_td_error],
        ["Updates", summary.update_count],
        ["Replay Size", summary.replay_size],
        ["Weight Delta", summary.weight_delta_norm],
      ];

      document.getElementById("cards").innerHTML = cards.map(([label, value]) => `
        <div class="card">
          <div class="label">${{label}}</div>
          <div class="value">${{formatValue(value)}}</div>
        </div>
      `).join("");

      document.getElementById("episodes").innerHTML = episodes.slice(-30).reverse().map((event) => `
        <tr>
          <td>${{event.episode ?? ""}}</td>
          <td>${{event.step ?? ""}}</td>
          <td>${{formatValue(event.return)}}</td>
          <td>${{formatValue(event.score)}}</td>
        </tr>
      `).join("");

      drawChart("scoreChart", steps.map(e => [e.step, e.score]), "#34d399");
      drawChart("lossChart", steps.filter(e => e.loss !== null).map(e => [e.step, e.loss]), "#60a5fa");
      drawChart("epsilonChart", steps.map(e => [e.step, e.epsilon]), "#fbbf24");
      drawChart("tdChart", steps.filter(e => e.td_error !== null).map(e => [e.step, e.td_error]), "#fb7185");

      function formatValue(value) {{
        if (value === null || value === undefined || Number.isNaN(value)) return "-";
        if (typeof value === "number") {{
          if (Math.abs(value) >= 1000) return value.toFixed(0);
          if (Math.abs(value) >= 10) return value.toFixed(2);
          return value.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
        }}
        return String(value);
      }}

      function drawChart(id, points, color) {{
        const canvas = document.getElementById(id);
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * ratio));
        canvas.height = Math.max(1, Math.floor(rect.height * ratio));
        const ctx = canvas.getContext("2d");
        ctx.scale(ratio, ratio);
        const w = rect.width;
        const h = rect.height;
        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = "#31394b";
        ctx.lineWidth = 1;
        for (let i = 0; i < 4; i += 1) {{
          const y = 12 + i * (h - 24) / 3;
          ctx.beginPath();
          ctx.moveTo(12, y);
          ctx.lineTo(w - 12, y);
          ctx.stroke();
        }}
        if (!points.length) {{
          ctx.fillStyle = "#a8b2c5";
          ctx.font = "13px system-ui";
          ctx.fillText("No data yet", 14, 26);
          return;
        }}
        const xs = points.map(p => p[0]);
        const ys = points.map(p => p[1]).filter(v => Number.isFinite(v));
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        let minY = Math.min(...ys);
        let maxY = Math.max(...ys);
        if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return;
        if (Math.abs(maxY - minY) < 1e-9) {{
          minY -= 1;
          maxY += 1;
        }}
        const px = (x) => 12 + ((x - minX) / Math.max(1, maxX - minX)) * (w - 24);
        const py = (y) => h - 12 - ((y - minY) / (maxY - minY)) * (h - 24);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach(([x, y], index) => {{
          const xx = px(x);
          const yy = py(y);
          if (index === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        }});
        ctx.stroke();
        ctx.fillStyle = "#a8b2c5";
        ctx.font = "12px system-ui";
        ctx.fillText(formatValue(maxY), 14, 18);
        ctx.fillText(formatValue(minY), 14, h - 14);
      }}
    </script>
  </body>
</html>
"""

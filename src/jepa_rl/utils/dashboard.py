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
      .badge {{
        margin-left: 8px;
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 2px 8px;
      }}
      main {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 20px 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
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
        .charts {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <header>
      <span class="brand">jepa-rl</span>
      <span class="sep"></span>
      <span class="run-name">{run_name}</span>
      <span class="badge">static snapshot</span>
    </header>
    <main>
      <div class="metrics" id="metrics"></div>
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
      const payload = {payload_json};
      const summary = payload.summary || {{}};
      const steps = payload.stepEvents || [];
      const episodes = payload.episodeEvents || [];

      const items = [
        ["steps", summary.steps],
        ["episodes", summary.episodes],
        ["best score", summary.best_score],
        ["mean loss", summary.mean_loss],
        ["mean td error", summary.mean_td_error],
        ["updates", summary.update_count],
        ["replay size", summary.replay_size],
        ["weight delta", summary.weight_delta_norm],
      ];

      document.getElementById("metrics").innerHTML = items.map(([label, value]) =>
        '<div class="metric"><div class="m-label">' + label + '</div><div class="m-val">' + fmt(value) + '</div></div>'
      ).join("");

      document.getElementById("episodes").innerHTML = episodes.slice(-30).reverse().map((e) =>
        "<tr><td>" + (e.episode ?? "") + "</td><td>" + (e.step ?? "") + "</td><td>" + fmt(e.return) + "</td><td>" + fmt(e.score) + "</td></tr>"
      ).join("");

      drawChart("scoreChart", steps.map(e => [e.step, e.score]), "#5d9e5d");
      drawChart("lossChart", steps.filter(e => e.loss !== null).map(e => [e.step, e.loss]), "#4e89ba");
      drawChart("epsilonChart", steps.map(e => [e.step, e.epsilon]), "#bfa03e");
      drawChart("tdChart", steps.filter(e => e.td_error !== null).map(e => [e.step, e.td_error]), "#b9524c");

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
    </script>
  </body>
</html>
"""

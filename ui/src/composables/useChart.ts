import { watch, ref, type Ref } from "vue";

const ACCENT_HUES: Record<string, string> = {
  "#5d9e5d": "#48c9a0",
  "#4e89ba": "#8b5ecf",
  "#bfa03e": "#d4764e",
  "#b9524c": "#cf5e8b",
};

let noisePattern: CanvasPattern | null = null;
function getNoise(ctx: CanvasRenderingContext2D) {
  if (noisePattern) return noisePattern;
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 128;
  const cx = c.getContext("2d");
  if (!cx) return null;
  const img = cx.createImageData(128, 128);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255;
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
    img.data[i + 3] = 10;
  }
  cx.putImageData(img, 0, 0);
  noisePattern = ctx.createPattern(c, "repeat");
  return noisePattern;
}

export function useChart(
  canvasRef: Ref<HTMLCanvasElement | null>,
  points: Ref<[number, number][]>,
  color: Ref<string>,
) {
  const minYPx = ref(0);
  const maxYPx = ref(0);
  const minVal = ref<number | null>(null);
  const maxVal = ref<number | null>(null);

  function draw() {
    const canvas = canvasRef.value;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const pts = points.value;
    if (!pts.length) {
      ctx.fillStyle = "#7d7870";
      ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("no data", w / 2, h / 2 + 4);
      return;
    }

    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]).filter(Number.isFinite);
    let dataMinY = Math.min(...ys);
    let dataMaxY = Math.max(...ys);
    if (!Number.isFinite(dataMinY)) return;
    if (dataMaxY - dataMinY < 1e-9) {
      dataMinY -= 1;
      dataMaxY += 1;
    }
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);

    const yRange = dataMaxY - dataMinY;
    const minY = dataMinY - yRange * 0.1;
    const maxY = dataMaxY + yRange * 0.15;

    const dataRangeX = Math.max(1, maxX - minX);
    const projMaxX = pts.length >= 8 ? maxX + dataRangeX * 0.01 : maxX;

    const px = (x: number) => ((x - minX) / Math.max(1, projMaxX - minX)) * w;
    const py = (y: number) => h - ((y - minY) / (maxY - minY)) * h;

    const accent = ACCENT_HUES[color.value] || color.value;

    // Subtle grid lines with a bit of gradient
    const gridGrad = ctx.createLinearGradient(0, 0, w, 0);
    gridGrad.addColorStop(0, "rgba(255,255,255,0.01)");
    gridGrad.addColorStop(0.5, "rgba(255,255,255,0.05)");
    gridGrad.addColorStop(1, "rgba(255,255,255,0.01)");
    ctx.strokeStyle = gridGrad;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    for (let i = 1; i < 4; i++) {
      const gy = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Path for the main line
    const linePath = new Path2D();
    pts.forEach(([x, y], i) => {
      if (i === 0) linePath.moveTo(px(x), py(y));
      else linePath.lineTo(px(x), py(y));
    });

    // Gradient fill with noise
    const fillPath = new Path2D(linePath);
    fillPath.lineTo(px(xs[xs.length - 1]), h + 10);
    fillPath.lineTo(px(xs[0]), h + 10);
    fillPath.closePath();

    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, hexRgba(accent, 0.35));
    fillGrad.addColorStop(0.3, hexRgba(color.value, 0.15));
    fillGrad.addColorStop(0.8, hexRgba(color.value, 0.02));
    fillGrad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = fillGrad;
    ctx.fill(fillPath);

    const noise = getNoise(ctx);
    if (noise) {
      ctx.globalCompositeOperation = "overlay";
      ctx.fillStyle = noise;
      ctx.fill(fillPath);
      ctx.globalCompositeOperation = "source-over";
    }

    // Trendline & estimates
    if (pts.length >= 8) {
      const { slope, intercept } = linReg(pts);

      const spread = yRange * 0.15;
      ctx.fillStyle = hexRgba(color.value, 0.03);
      ctx.beginPath();
      ctx.moveTo(px(minX), py(slope * minX + intercept + spread));
      ctx.lineTo(px(projMaxX), py(slope * projMaxX + intercept + spread * 1.5));
      ctx.lineTo(px(projMaxX), py(slope * projMaxX + intercept - spread * 1.5));
      ctx.lineTo(px(minX), py(slope * minX + intercept - spread));
      ctx.closePath();
      ctx.fill();

      ctx.save();
      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = hexRgba(color.value, 0.4);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px(minX), py(slope * minX + intercept));
      ctx.lineTo(px(maxX), py(slope * maxX + intercept));
      ctx.stroke();

      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = hexRgba(accent, 0.3);
      ctx.beginPath();
      ctx.moveTo(px(maxX), py(slope * maxX + intercept));
      ctx.lineTo(px(projMaxX), py(slope * projMaxX + intercept));
      ctx.stroke();
      ctx.restore();
    }

    // Line with extra gradient stroke and glow
    ctx.save();
    ctx.shadowColor = hexRgba(color.value, 0.6);
    ctx.shadowBlur = 8;

    const lineStrokeGrad = ctx.createLinearGradient(0, 0, w, 0);
    lineStrokeGrad.addColorStop(0, color.value);
    lineStrokeGrad.addColorStop(0.5, accent);
    lineStrokeGrad.addColorStop(1, color.value);

    ctx.strokeStyle = lineStrokeGrad;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke(linePath);
    ctx.restore();

    // Min/Max checkpoints with highlighted dots
    let maxPt = pts[0],
      minPt = pts[0];
    pts.forEach((p) => {
      if (p[1] > maxPt[1]) maxPt = p;
      if (p[1] < minPt[1]) minPt = p;
    });

    const drawCheckpoint = (pt: [number, number], c: string) => {
      const cx = px(pt[0]);
      const cy = py(pt[1]);
      // Outer glow
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = hexRgba(c, 0.25);
      ctx.fill();
      // Ring
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.strokeStyle = c;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Center dot
      ctx.beginPath();
      ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    };

    if (maxPt && dataMaxY - dataMinY > 0) drawCheckpoint(maxPt, accent);
    if (minPt && dataMaxY - dataMinY > 0 && minPt !== maxPt)
      drawCheckpoint(minPt, color.value);

    // Expose min/max for external labels
    maxYPx.value = py(dataMaxY);
    maxVal.value = dataMaxY;
    minYPx.value = py(dataMinY);
    minVal.value = dataMinY;

    // Last-point marker with animated/cool ring
    const last = pts[pts.length - 1];
    const lx = px(last[0]);
    const ly = py(last[1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 8, 0, Math.PI * 2);
    ctx.fillStyle = hexRgba(accent, 0.15);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fillStyle = hexRgba(color.value, 0.8);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lx, ly, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();

    // Min/Max reference lines
    ctx.save();
    ctx.setLineDash([1, 3]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = hexRgba(accent, 0.15);
    ctx.beginPath();
    ctx.moveTo(0, py(dataMaxY));
    ctx.lineTo(w, py(dataMaxY));
    ctx.stroke();
    ctx.strokeStyle = hexRgba(color.value, 0.1);
    ctx.beginPath();
    ctx.moveTo(0, py(dataMinY));
    ctx.lineTo(w, py(dataMinY));
    ctx.stroke();
    ctx.restore();
  }

  watch([points, color], draw);
  return { draw, minYPx, maxYPx, minVal, maxVal };
}

function hexRgba(hex: string, a: number): string {
  if (!hex) return `rgba(255,255,255,${a})`;
  let normalizedHex = hex;
  if (hex.length === 4) {
    normalizedHex = "#" + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3];
  }
  const r = parseInt(normalizedHex.slice(1, 3), 16) || 0;
  const g = parseInt(normalizedHex.slice(3, 5), 16) || 0;
  const b = parseInt(normalizedHex.slice(5, 7), 16) || 0;
  return `rgba(${r},${g},${b},${a})`;
}

function linReg(pts: [number, number][]): { slope: number; intercept: number } {
  const n = pts.length;
  let sx = 0,
    sy = 0,
    sxy = 0,
    sxx = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
    sxy += x * y;
    sxx += x * x;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return { slope: 0, intercept: sy / n };
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return { slope, intercept };
}

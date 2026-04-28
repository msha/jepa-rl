import { watch, type Ref } from 'vue'

export function useChart(canvasRef: Ref<HTMLCanvasElement | null>, points: Ref<[number, number][]>, color: Ref<string>) {
  function draw() {
    const canvas = canvasRef.value
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.floor(rect.width * dpr))
    canvas.height = Math.max(1, Math.floor(rect.height * dpr))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height
    ctx.clearRect(0, 0, w, h)

    const pts = points.value
    if (!pts.length) {
      ctx.fillStyle = '#7d7870'
      ctx.font = "11px 'IBM Plex Mono', monospace"
      ctx.textAlign = 'center'
      ctx.fillText('no data', w / 2, h / 2 + 4)
      return
    }

    const xs = pts.map(p => p[0])
    const ys = pts.map(p => p[1]).filter(Number.isFinite)
    let minY = Math.min(...ys)
    let maxY = Math.max(...ys)
    if (!Number.isFinite(minY)) return
    if (maxY - minY < 1e-9) { minY -= 1; maxY += 1 }
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)

    const px = (x: number) => ((x - minX) / Math.max(1, maxX - minX)) * w
    const py = (y: number) => h - ((y - minY) / (maxY - minY)) * h

    // Fill
    ctx.beginPath()
    pts.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(px(x), py(y))
      else ctx.lineTo(px(x), py(y))
    })
    ctx.lineTo(px(xs[xs.length - 1]), h)
    ctx.lineTo(px(xs[0]), h)
    ctx.closePath()
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, hexRgba(color.value, 0.15))
    grad.addColorStop(1, hexRgba(color.value, 0.01))
    ctx.fillStyle = grad
    ctx.fill()

    // Line
    ctx.strokeStyle = color.value
    ctx.lineWidth = 1.8
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.beginPath()
    pts.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(px(x), py(y))
      else ctx.lineTo(px(x), py(y))
    })
    ctx.stroke()

    // Last point marker
    const last = pts[pts.length - 1]
    ctx.beginPath()
    ctx.arc(px(last[0]), py(last[1]), 3, 0, Math.PI * 2)
    ctx.fillStyle = color.value
    ctx.fill()

    // Min/max labels
    ctx.fillStyle = '#7d7870'
    ctx.font = "9px 'IBM Plex Mono', monospace"
    ctx.textAlign = 'left'
    ctx.fillText(fmt(maxY), 3, 9)
    ctx.textAlign = 'right'
    ctx.fillText(fmt(minY), w - 3, h - 3)
  }

  watch([points, color], draw)
  return { draw }
}

function hexRgba(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${a})`
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

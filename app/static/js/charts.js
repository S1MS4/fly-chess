// The "How the flies trained" overlay: one small line chart per metric, one line per
// run, read from runs/*/metrics.jsonl through /api/metrics. Hover for the values.

import { api } from './api.js'

const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'] // fixed order, never cycled; steps for the dark panel
const RUN_ORDER = ['dl', 'dl-shuffled', 'bio', 'bio-shuffled', 'dl-day'] // a run keeps its colour whatever else is there
const RUN_NAMES = { dl: 'Backprop fly', 'dl-shuffled': 'Backprop fly, shuffled wiring', bio: 'Dopamine fly',
  'bio-shuffled': 'Dopamine fly, shuffled wiring', 'bio-rules': 'Dopamine rule experiments', 'dl-day': 'Day-long fly',
  'dl-pictures': 'Backprop fly, 10 minutes with snapshots' } // runs/ folder -> legend
const TITLES = {
  policy_loss: 'Policy loss (lower is better)', value_loss: 'Value loss', top1: "Top-1: the fly's favourite is Stockfish's move",
  good_moves: "Good moves: within 1% win chance of Stockfish's best", positions_per_s: 'Speed (positions per second)',
  held_out_loss: 'Policy loss on positions it never trained on',
  val_pair_acc: 'Dopamine fly: prefers the good move of a pair', pair_acc: 'Dopamine fly: pairs right while learning',
  finish_top1: 'Finishing: picks the fastest win in a won position', finish_good_moves: 'Finishing: a move about as good as the fastest win',
  finish_held_out_loss: 'Finishing: loss on won positions it never trained on',
}
const SKIP = new Set(['step', 'positions', 'minutes', 'lessons', 'game', 'level', 'lr', 'blind_top1', 'blind_good_moves',
  'grad_readout', 'grad_synapses', 'grad_neurons', 'best_val_loss'])

function colorOf(run, runs) {
  const i = RUN_ORDER.indexOf(run)
  return SERIES[i >= 0 ? i : RUN_ORDER.length + runs.filter((r) => !RUN_ORDER.includes(r)).indexOf(run)] ?? '#898781'
}

function chart(title, series, xLabel) {
  const wrap = document.createElement('div')
  wrap.className = 'chart'
  const h = document.createElement('h3')
  h.textContent = title
  const cv = document.createElement('canvas')
  const legend = document.createElement('div')
  legend.className = 'legend'
  for (const s of series) {
    const item = document.createElement('span')
    const swatch = document.createElement('i')
    swatch.style.background = s.color
    item.append(swatch, s.name)
    legend.append(item)
  }
  const tip = document.createElement('div')
  tip.className = 'chart-tip'
  tip.hidden = true
  wrap.append(h, cv, legend, tip)
  requestAnimationFrame(() => draw(cv, series, xLabel, null))
  cv.addEventListener('pointermove', (e) => {
    const x = draw(cv, series, xLabel, e.offsetX)
    if (x === null) return (tip.hidden = true)
    tip.replaceChildren()
    const head = document.createElement('div')
    head.textContent = `${xLabel} ${x.toFixed(1)}`
    tip.append(head)
    for (const s of series) {
      const p = nearest(s.points, x)
      if (!p) continue
      const row = document.createElement('div')
      const v = document.createElement('b')
      v.textContent = p[1] < 1 ? p[1].toFixed(3) : p[1].toFixed(1)
      row.append(v, ` ${s.name}`)
      tip.append(row)
    }
    tip.hidden = false
    tip.style.left = `${Math.min(e.offsetX + 14, cv.clientWidth - 170)}px`
    tip.style.top = `${e.offsetY + 30}px`
  })
  cv.addEventListener('pointerleave', () => { tip.hidden = true; draw(cv, series, xLabel, null) })
  return wrap
}

function nearest(points, x) {
  let best = null
  for (const p of points) if (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p
  return best
}

/** Draws the chart; with a pointer x it adds the crosshair and returns the x value under it. */
function draw(cv, series, xLabel, px) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, hgt = cv.clientHeight
  cv.width = w * dpr
  cv.height = hgt * dpr
  const g = cv.getContext('2d')
  g.scale(dpr, dpr)
  const pts = series.flatMap((s) => s.points)
  if (!pts.length) return null
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const L = 46, R = 10, T = 8, B = 22
  const X = (x) => L + ((x - x0) / Math.max(1e-9, x1 - x0)) * (w - L - R)
  const Y = (y) => hgt - B - ((y - y0) / Math.max(1e-9, y1 - y0)) * (hgt - T - B)
  g.font = '11px system-ui'
  g.fillStyle = '#898781'
  g.strokeStyle = '#3a3936'
  g.lineWidth = 1
  const digits = Math.min(4, Math.max(0, Math.ceil(-Math.log10(Math.max(1e-9, (y1 - y0) / 3))) + 1)) // enough that the labels differ
  for (let i = 0; i <= 3; i++) { // recessive grid with y labels
    const y = y0 + ((y1 - y0) * i) / 3
    g.beginPath(); g.moveTo(L, Y(y)); g.lineTo(w - R, Y(y)); g.stroke()
    g.fillText(y.toFixed(digits), 4, Y(y) + 4)
  }
  g.fillText(`${xLabel} ${x0.toFixed(0)}`, L, hgt - 6)
  g.fillText(x1.toFixed(0), w - R - 24, hgt - 6)
  for (const s of series) {
    g.strokeStyle = s.color
    g.lineWidth = 2
    g.lineJoin = 'round'
    g.beginPath()
    s.points.forEach(([x, y], j) => (j ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))))
    g.stroke()
  }
  if (px === null || px < L || px > w - R) return null
  g.strokeStyle = '#6b6a66'
  g.lineWidth = 1
  g.beginPath(); g.moveTo(px, T); g.lineTo(px, hgt - B); g.stroke()
  return x0 + ((px - L) / (w - L - R)) * (x1 - x0)
}

export async function refreshCharts(host) {
  const data = await api.metrics()
  const runs = Object.keys(data)
  const metrics = {}
  for (const [run, rows] of Object.entries(data)) {
    for (const row of rows) {
      const x = row.minutes ?? row.lessons ?? row.game ?? row.step
      for (const [k, v] of Object.entries(row)) {
        if (SKIP.has(k) || typeof v !== 'number') continue
        ;((metrics[k] ??= {})[run] ??= []).push([x, v])
      }
    }
  }
  host.replaceChildren()
  if (!runs.length) host.textContent = 'No runs yet. Start training and this fills in.'
  for (const [k, perRun] of Object.entries(metrics)) {
    const series = Object.entries(perRun).map(([run, points]) => ({ name: RUN_NAMES[run] ?? run, points, color: colorOf(run, runs) }))
    host.append(chart(TITLES[k] ?? k, series, 'minute'))
  }
}

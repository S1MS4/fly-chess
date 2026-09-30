// fly-chess interface: board, brain map, training charts. No build step.

const game = new Chess();
let board, activity = [], busy = false;

// Brain map layout: the signal roughly flows left to right, eye to legs.
const LAYOUT = {
  "photoreceptors":   [0.02, 0.40],
  "optic lobe":       [0.17, 0.40],
  "visual projection":[0.33, 0.40],
  "antennal lobe":    [0.33, 0.75],
  "kenyon cells":     [0.50, 0.15],
  "MBON":             [0.67, 0.15],
  "dopamine (PAM)":   [0.60, 0.02],
  "dopamine (PPL1)":  [0.75, 0.02],
  "central brain":    [0.50, 0.50],
  "central complex":  [0.67, 0.40],
  "descending":       [0.84, 0.50],
  "ascending":        [0.67, 0.75],
  "nerve cord":       [0.84, 0.80],
  "other":            [0.50, 0.85],
};
const FLOW = [["photoreceptors","optic lobe"],["optic lobe","visual projection"],["visual projection","central brain"],
  ["visual projection","kenyon cells"],["antennal lobe","kenyon cells"],["kenyon cells","MBON"],["dopamine (PAM)","MBON"],
  ["dopamine (PPL1)","MBON"],["MBON","central brain"],["central brain","central complex"],["central complex","descending"],
  ["central brain","descending"],["descending","nerve cord"],["nerve cord","ascending"],["ascending","central brain"]];

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawBrain(step) {
  const cv = document.getElementById("brain"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height, bw = 88, bh = 42;
  g.clearRect(0, 0, W, H);
  const frame = activity[step] || {};
  const vals = Object.values(frame), max = Math.max(1e-6, ...vals);
  const pos = r => [LAYOUT[r][0] * W, LAYOUT[r][1] * H];

  g.strokeStyle = css("--line"); g.lineWidth = 2;
  for (const [a, b] of FLOW) {
    const [x1, y1] = pos(a), [x2, y2] = pos(b);
    g.beginPath(); g.moveTo(x1 + bw / 2, y1 + bh / 2); g.lineTo(x2 + bw / 2, y2 + bh / 2); g.stroke();
  }
  for (const r of Object.keys(LAYOUT)) {
    const [x, y] = pos(r), v = (frame[r] ?? 0) / max;
    g.fillStyle = `rgba(217, 122, 30, ${0.08 + 0.92 * v})`;
    g.strokeStyle = css("--line");
    g.beginPath(); g.roundRect(x, y, bw, bh, 8); g.fill(); g.stroke();
    g.fillStyle = v > 0.55 ? "#fff" : css("--ink");
    g.font = "11px system-ui"; g.textAlign = "center";
    g.fillText(r, x + bw / 2, y + 18);
    g.fillText(frame[r] !== undefined ? frame[r].toFixed(3) : "-", x + bw / 2, y + 33);
  }
}

function setStatus(t) { document.getElementById("status").textContent = t; }

async function flyMove() {
  if (game.game_over()) return setStatus("Game over: " + (game.in_checkmate() ? "checkmate." : "draw."));
  busy = true; setStatus("The fly is thinking...");
  const res = await fetch("/move", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen: game.fen(), fly: document.getElementById("fly").value }) });
  if (!res.ok) { busy = false; return setStatus("Server error: " + (await res.text())); }
  const d = await res.json();
  game.move(d.move, { sloppy: true });
  board.position(game.fen());
  activity = d.activity;
  const slider = document.getElementById("step");
  slider.max = activity.length - 1; slider.value = activity.length - 1;
  showStep(activity.length - 1);
  document.getElementById("top").innerHTML = d.top.map(([m, s]) => `<li>${m} <span>${s.toFixed(3)}</span></li>`).join("");
  document.getElementById("value").textContent = (d.value !== undefined ? `Fly thinks its position is ${d.value.toFixed(2)} (-1 lost, +1 won). ` : "")
    + (d.trained ? "" : "No checkpoint found, this fly is untrained.");
  busy = false;
  setStatus(game.game_over() ? "Game over." : `Fly played ${d.san}. Your move.`);
}

function showStep(i) {
  drawBrain(i);
  document.getElementById("step-label").textContent = `step ${+i + 1} of ${activity.length || 12}`;
}

function newGame() {
  game.reset();
  const side = document.getElementById("side").value;
  board.orientation(side === "w" ? "white" : "black");
  board.position("start");
  activity = []; drawBrain(0);
  setStatus("Your move.");
  if (side === "b") flyMove();
}

board = Chessboard("board", {
  draggable: true, position: "start",
  pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
  onDragStart: (src, piece) => !busy && !game.game_over() && piece[0] === game.turn() && game.turn() === document.getElementById("side").value,
  onDrop: (src, dst) => {
    const m = game.move({ from: src, to: dst, promotion: "q" });
    if (!m) return "snapback";
    setTimeout(flyMove, 50);
  },
  onSnapEnd: () => board.position(game.fen()),
});

document.getElementById("new").onclick = newGame;
document.getElementById("step").oninput = e => showStep(e.target.value);
window.addEventListener("resize", () => board.resize());
drawBrain(0);

// Training charts: one small line chart per metric, one line per run.
const COLOURS = ["#d97a1e", "#2f7fc1", "#3a9d5d", "#b04a8f", "#7a6a3a"];
function lineChart(title, series) {
  const wrap = document.createElement("div"); wrap.className = "chart";
  wrap.innerHTML = `<h3>${title}</h3>`;
  const cv = document.createElement("canvas"); cv.width = 900; cv.height = 220; wrap.appendChild(cv);
  const g = cv.getContext("2d"), pts = series.flatMap(s => s.points);
  if (!pts.length) return wrap;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const X = x => 40 + (x - x0) / Math.max(1e-9, x1 - x0) * (cv.width - 50);
  const Y = y => cv.height - 20 - (y - y0) / Math.max(1e-9, y1 - y0) * (cv.height - 40);
  g.fillStyle = css("--muted"); g.font = "11px system-ui";
  g.fillText(y1.toFixed(3), 0, 14); g.fillText(y0.toFixed(3), 0, cv.height - 22);
  series.forEach((s, i) => {
    g.strokeStyle = COLOURS[i % COLOURS.length]; g.lineWidth = 1.5; g.beginPath();
    s.points.forEach(([x, y], j) => j ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)));
    g.stroke(); g.fillStyle = g.strokeStyle; g.fillText(s.name, 50 + i * 120, cv.height - 4);
  });
  return wrap;
}

async function refreshCharts() {
  const data = await (await fetch("/metrics")).json();
  const charts = document.getElementById("charts"); charts.innerHTML = "";
  const metrics = {};
  for (const [run, rows] of Object.entries(data)) {
    for (const row of rows) {
      const x = row.step ?? row.game;
      for (const [k, v] of Object.entries(row)) {
        if (["step", "game", "level"].includes(k) || typeof v !== "number") continue;
        ((metrics[k] ??= {})[run] ??= []).push([x, v]);
      }
    }
  }
  for (const [k, runs] of Object.entries(metrics))
    charts.appendChild(lineChart(k, Object.entries(runs).map(([name, points]) => ({ name, points }))));
  if (!Object.keys(metrics).length) charts.textContent = "No runs yet. Start training and this fills in.";
}

document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button, .tab").forEach(e => e.classList.remove("on"));
  b.classList.add("on"); document.getElementById(b.dataset.tab).classList.add("on");
  if (b.dataset.tab === "train") refreshCharts();
});
setInterval(() => document.getElementById("train").classList.contains("on") && refreshCharts(), 10000);

// Talking to app/server.py. Every chess rule lives there (python-chess), so the
// browser only animates what the server says.

async function json(res) {
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

const post = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(json)

export const api = {
  legal: (fen) => fetch(`/api/legal?fen=${encodeURIComponent(fen)}`).then(json),
  play: (fen, uci, fly, seen, imagine) => post('/api/play', { fen, uci, fly, seen, imagine: !!imagine }),
  analyse: (fen, moves) => post('/api/analyse', { fen, moves }), // no moves: just Stockfish's best one, for the hint
  opening: (ucis) => fetch(`/api/opening?moves=${ucis.join(',')}`).then(json),
  warm: (fly) => fetch(`/api/warm/${fly}`, { method: 'POST' }).catch(() => {}), // load the fly before its first move
  flies: () => fetch('/api/flies').then(json), // which flies are trained yet
  brain: () => fetch('/api/brain').then(json),
  points: () => fetch('/api/brain/points').then((r) => r.arrayBuffer()),
  activity: (id) => fetch(`/api/activity/${id}`).then((r) => (r.ok ? r.arrayBuffer() : null)),
  metrics: () => fetch('/api/metrics').then(json),
}

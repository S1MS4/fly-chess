import { api } from './api.js'
import { Board, parseFen } from './board.js'
import { BrainView } from './brain.js'
import { refreshCharts } from './charts.js'
import { Panel, pct } from './panel.js'

// The game, laid out like chess.com: pick a fly and a colour, then play. Your move goes to the
// server, the fly thinks (its brain replays the think in the panel) and its piece slides over.
// While it thinks you can premove. You can take moves back, ask Stockfish for a hint, flip the
// board and click back through the game. Every position the board shows is a FEN from the server.

const $ = (id) => document.getElementById(id)
const saved = (key, fallback) => { try { return localStorage.getItem(key) ?? fallback } catch { return fallback } }
const save = (key, value) => { try { localStorage.setItem(key, value) } catch { /* private window */ } }
const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
const REPLAY = 1.3 // seconds the brain takes to replay a think before the fly moves

// The flies you can play (FLIES in flychess/play.py). The lab flies are the control brains from the
// README's experiments. Their Elo comes from ratings.json, written by python -m flychess.rating.
const FLIES = {
  dl: {
    name: 'Backprop fly', group: 0, color: '#4f7fc4',
    hello: "Hi! I'm a fruit fly brain, wired neuron for neuron like a real one. Backprop taught me chess for 25 minutes.",
    stat: (m) => (m.dl?.top1 ? `Finds Stockfish's move ${(m.dl.top1 * 100).toFixed(1)}% of the time` : 'Trained by backprop'),
  },
  day: {
    name: 'Day-long fly', group: 0, color: '#3a6fc7',
    hello: 'Same brain as the backprop fly, but I trained for a whole day, with an eye that sees every square and lessons in finishing won games.',
    stat: (m) => (m.day?.top1 ? `Finds Stockfish's move ${(m.day.top1 * 100).toFixed(1)}% of the time` : 'Trained for a day'),
  },
  bio: {
    name: 'Dopamine fly', group: 0, color: '#c0558a',
    hello: "Same brain, but I learned the way real flies do: sugar for Stockfish's move, a shock for the other one.",
    stat: (m) => (m.bio?.val_pair_acc ? `Tells Stockfish's move from a random one ${(m.bio.val_pair_acc * 100).toFixed(1)}% of the time` : 'Trained with dopamine'),
  },
  quick: {
    name: '3-minute fly', group: 1, color: '#2f9c8e',
    hello: "I'm the backprop fly after 3 minutes of training instead of 25.",
    stat: () => "Finds Stockfish's move 18.0% of the time",
  },
  shuffled: {
    name: 'Shuffled fly', group: 1, color: '#7d5cc4',
    hello: 'My wiring is scrambled: the same neurons and synapses, pointing at random targets. I read chess better than the real fly anyway.',
    stat: () => "Finds Stockfish's move 20.9% of the time",
  },
  'no-optic-lobe': {
    name: 'Fly without optic lobes', group: 1, color: '#77756f',
    hello: 'Someone cut out my optic lobes, more than half of all my neurons. The board still gets through, just less of it.',
    stat: () => "Finds Stockfish's move 15.1% of the time",
  },
  'no-central-brain': {
    name: 'Fly without a central brain', group: 1, color: '#c8742f',
    hello: "No central brain, the part where a real fly makes its decisions. Without it I still pick Stockfish's move as often.",
    stat: () => "Finds Stockfish's move 17.9% of the time",
  },
  newborn: {
    name: 'Newborn fly', group: 1, color: '#b59a35',
    hello: "I've never seen a chessboard and nobody has taught me anything. Every move is a guess :D",
    stat: () => 'Never trained',
  },
}
const GROUPS = ['Trained flies', 'Lab flies']
const REASONS = {
  checkmate: 'by checkmate', stalemate: 'by stalemate', 'insufficient material': 'by insufficient material',
  'threefold repetition': 'by repetition', 'fivefold repetition': 'by repetition', 'fifty moves': 'by the 50-move rule',
  'seventyfive moves': 'by the 75-move rule', resignation: 'by resignation',
}

const key = (fen) => fen.split(' ').slice(0, 4).join(' ') // the same position, whatever the move number
const turnOf = (fen) => fen.split(' ')[1]
const other = (color) => (color === 'w' ? 'b' : 'w')
const colorName = (color) => (color === 'w' ? 'white' : 'black')

function el(tag, cls, text) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

const setup = { fly: saved('fly', 'dl') in FLIES ? saved('fly', 'dl') : 'dl', pick: saved('side', 'w'), imagine: saved('imagine', '0') === '1' }
const game = {
  id: 0, // bumped by every new game and takeback, so a think for an old position gets ignored
  state: 'setup', // setup, playing, over
  started: false,
  fly: setup.fly,
  side: setup.pick === 'b' ? 'b' : 'w',
  plies: [], // {by, san, uci, from, to, before, fen, check, captured, opening, verdict, accuracy, think?, expect?}
  live: null, // the server's latest position: fen, legal moves, check, opening, over
  view: null, // the ply you're looking back at, or null for the live position
  thinking: false,
  pending: null, // the fly's move while its brain replays the think
  hint: null,
  result: '*',
}
const board = new Board($('board'))
const panel = new Panel({ onGoto: (i) => goto(i), onArrow: (a) => board.setArrows(a ? [a] : game.view === null && game.hint ? [game.hint] : []) })
let brain = null, metrics = {}, ratings = {}, chartTimer = null, tabTimer = null, resignTimer = null, hintPending = false

const shownIndex = () => game.view ?? game.plies.length - 1 // -1 is the start position
const fenAt = (i) => (i < 0 ? START : game.plies[i].fen)
const canMove = () => game.state === 'playing' && !game.thinking && !!game.live && turnOf(game.live.fen) === game.side
const seenBefore = (n) => [key(START), ...game.plies.slice(0, n).map((p) => key(p.fen))] // the positions so far, for the repetition rule
const elo = (kind) => ratings[kind]?.elo ?? null // you stay anonymous: only the flies have a rating
const looksAhead = (kind) => kind !== 'bio' // the dopamine fly always looks at the boards its moves lead to, by design
const ratedAs = (kind, imagine) => (imagine && looksAhead(kind) ? `${kind}-imagine` : kind) // with imagination it has its own Elo
const flyName = () => FLIES[game.fly].name + (game.imagine ? ' + imagination' : '')
const flyElo = () => elo(ratedAs(game.fly, game.imagine))

function openingAt(i) {
  for (let j = i; j >= 0; j--) if (game.plies[j].name) return game.plies[j].name
  return null
}

/** Ask for the opening's name once a ply is played; it's named by move order, like chess.com does. */
function nameOpening(ply) {
  const n = game.plies.indexOf(ply) + 1
  if (n > 40) return // long out of the book; the last name stays
  api.opening(game.plies.slice(0, n).map((p) => p.uci)).then((op) => {
    ply.name = op
    refresh()
  }).catch(() => {})
}

// ---- drawing the panel and the player bars -------------------------------------------------

function refresh() {
  const i = shownIndex()
  panel.renderMoves(game.plies, i)
  panel.renderAccuracy(game.plies, game.side)
  panel.setOpening(openingAt(i))
  renderGuesses(i)
  renderPlayers()
  renderFooter()
}

function renderGuesses(i) {
  let flyIdx = -1
  for (let j = Math.min(i, game.plies.length - 1); j >= 0; j--) if (game.plies[j].by === 'fly') { flyIdx = j; break }
  const flyPly = game.plies[flyIdx]
  let flyTab, youTab
  if (game.view === null && game.thinking) {
    const p = game.pending
    flyTab = p ? { title: 'What the fly is choosing between', candidates: p.think.candidates, verdicts: p.think.verdicts }
      : { title: 'The fly is thinking…' }
  } else if (flyPly) {
    flyTab = { title: `Before ${flyPly.san}, the fly was choosing between`, candidates: flyPly.think.candidates,
      verdicts: flyPly.think.verdicts, played: flyPly.uci, value: flyPly.think.value }
  } else {
    flyTab = { title: "Make a move. What the fly chooses between shows up here, with Stockfish's verdict on each." }
  }
  if (flyPly?.expect) {
    const reply = flyIdx < i ? game.plies[flyIdx + 1] : null
    youTab = reply
      ? { title: `What it expected you to play. You played ${reply.san}.`, candidates: flyPly.expect, verdicts: flyPly.expectVerdicts, played: reply.uci }
      : { title: 'What the fly expects you to play', candidates: flyPly.expect, hidden: true } // grading your options would be a free hint
  } else {
    youTab = { title: 'Its guess at your next move shows up here after its first move.' }
  }
  panel.setGuesses(flyTab, youTab)
}

function renderPlayers() {
  const top = other(board.orientation)
  playerBar($('top-player'), top)
  playerBar($('bottom-player'), other(top))
}

function playerBar(bar, color) {
  const i = shownIndex(), isFly = color !== game.side
  const avatar = el('span', isFly ? 'avatar' : 'avatar you')
  if (isFly) avatar.dataset.fly = game.fly
  const who = el('div', 'who'), taken = el('div', 'taken'), name = el('div', 'name', isFly ? flyName() : 'You')
  if (isFly && flyElo() !== null) name.append(' ', el('small', '', `(${flyElo()})`)) // like chess.com's 'Maria (1000)'
  who.append(name, taken)
  // The pieces this side took, grouped like chess.com does, and how much material it's up.
  const caps = game.plies.slice(0, i + 1).filter((p, j) => (j % 2 ? 'b' : 'w') === color && p.captured).map((p) => p.captured)
  caps.sort((a, b) => 'pnbrq'.indexOf(a.toLowerCase()) - 'pnbrq'.indexOf(b.toLowerCase()))
  let group = null
  caps.forEach((code, j) => {
    if (code !== caps[j - 1]) taken.append((group = el('span', 'group')))
    const icon = el('i')
    icon.style.backgroundImage = `url(pieces/Chess_${code.toLowerCase()}${code === code.toUpperCase() ? 'l' : 'd'}t45.svg)`
    group.append(icon)
  })
  let up = 0
  for (const code of parseFen(fenAt(i)).values()) up += VALUES[code.toLowerCase()] * ((code === code.toUpperCase()) === (color === 'w') ? 1 : -1)
  if (up > 0) taken.append(el('span', 'adv', `+${up}`))
  bar.replaceChildren(avatar, who)
  if (game.state !== 'playing' || game.view !== null || !game.live) return
  const toMove = game.thinking ? other(game.side) : turnOf(game.live.fen)
  if (toMove !== color) return
  if (isFly && game.thinking) {
    const pill = el('span', 'turn fly', 'Thinking')
    pill.prepend(Object.assign(el('span', 'dots'), { innerHTML: '<i></i><i></i><i></i>' }))
    bar.append(pill)
  } else if (!isFly) bar.append(el('span', 'turn you', 'Your move'))
}

function renderFooter() {
  const over = game.state === 'over'
  $('play-buttons').hidden = over
  $('over-buttons').hidden = !over
  $('resign').disabled = game.state !== 'playing'
  $('takeback').disabled = game.state === 'setup' || !game.plies.some((p) => p.by === 'you')
  $('hint').disabled = !canMove() || hintPending
  $('prev').disabled = shownIndex() < 0
  $('next').disabled = game.view === null
  $('pgn').disabled = !game.plies.length
}

function updateBoardInput() {
  if (game.view === null && canMove()) board.enable(game.live.legal)
  else if (game.view === null && game.state === 'playing' && game.thinking) board.enablePremove(game.side) // while it thinks
  else {
    board.disable()
    board.clearPremove()
  }
}

function setFlyColor() {
  document.documentElement.style.setProperty('--fly', FLIES[game.fly].color)
}

// ---- setup: pick a fly and a colour -------------------------------------------------------

/** The fly list, like chess.com's bot list: a box per group, weakest first, the rating under each fly. */
function buildFlyBox() {
  $('fly-box').replaceChildren(...GROUPS.map((title, g) => {
    const kinds = Object.keys(FLIES).filter((k) => FLIES[k].group === g).sort((a, b) => (elo(a) ?? Infinity) - (elo(b) ?? Infinity))
    const section = el('section', 'bot-group'), head = el('header'), grid = el('div', 'bot-grid')
    head.append(el('h3', '', title), el('span', '', `${kinds.length} flies`))
    for (const kind of kinds) {
      const tile = el('button', 'bot-tile'), avatar = el('span', 'avatar')
      Object.assign(tile, { type: 'button', title: FLIES[kind].name + (elo(kind) !== null ? `, Elo ${elo(kind)}` : '') })
      tile.dataset.fly = avatar.dataset.fly = kind
      tile.setAttribute('role', 'radio')
      tile.append(avatar, el('span', 'tile-elo', elo(kind) ?? '?'))
      tile.onclick = () => {
        setup.fly = kind
        renderSetup()
      }
      grid.append(tile)
    }
    section.append(head, grid)
    return section
  }))
  renderSetup()
}

function renderSetup() {
  const f = FLIES[setup.fly], r = ratings[ratedAs(setup.fly, setup.imagine)]
  $('imagine-row').hidden = !looksAhead(setup.fly)
  $('imagine').checked = setup.imagine
  $('hero-avatar').dataset.fly = setup.fly
  $('hero-name').textContent = f.name
  $('hero-elo').textContent = r ? r.elo : ''
  $('hero-elo').title = r ? `Elo ${r.elo}, 95% sure it's between ${r.low} and ${r.high} (anchored at Stockfish 1320)` : ''
  $('hero-stat').textContent = f.stat(metrics)
  $('hero-bubble').textContent = f.hello
  for (const t of document.querySelectorAll('.bot-tile')) t.setAttribute('aria-checked', String(t.dataset.fly === setup.fly))
  for (const b of document.querySelectorAll('.side-grid button')) b.setAttribute('aria-checked', String(b.dataset.side === setup.pick))
  if (!game.started) { // before the first game the board shows what you're picking
    game.fly = setup.fly
    game.imagine = setup.imagine && looksAhead(setup.fly)
    game.side = setup.pick === 'b' ? 'b' : 'w'
    board.setOrientation(game.side)
    setFlyColor()
    renderPlayers()
  }
}

function showSetup() { // mid-game it's New game and Continue, like chess.com, starting from the fly you're playing
  if (game.started) Object.assign(setup, { fly: game.fly, pick: game.pick })
  $('setup').hidden = $('setup-foot').hidden = false
  $('game').hidden = $('game-foot').hidden = true
  $('back-btn').hidden = !game.started
  $('back-btn').textContent = game.state === 'over' ? 'Back to the game' : 'Continue'
  $('play-btn').textContent = game.started ? 'New game' : 'Play'
  $('play-btn').className = game.started ? 'btn big' : 'btn-green big'
  renderSetup()
}

function showGame() {
  $('setup').hidden = $('setup-foot').hidden = true
  $('game').hidden = $('game-foot').hidden = false
}

// ---- the game --------------------------------------------------------------------------------

async function startGame() {
  const id = ++game.id
  board.disable()
  clearTimeout(tabTimer)
  resetResign()
  const side = setup.pick === 'r' ? (Math.random() < 0.5 ? 'w' : 'b') : setup.pick
  Object.assign(game, { state: 'playing', started: true, fly: setup.fly, pick: setup.pick, side, plies: [], live: null, view: null,
    thinking: false, pending: null, hint: null, result: '*', warned: false, ending: null, imagine: setup.imagine && looksAhead(setup.fly) })
  save('fly', setup.fly)
  save('side', setup.pick)
  save('imagine', setup.imagine ? '1' : '0')
  api.warm(setup.fly)
  setFlyColor()
  hideResult()
  board.clearPremove(false)
  board.setOrientation(side)
  board.clearAnnotations()
  board.setArrows([])
  board.set(START, { animate: false })
  showGame()
  if ($('play').getBoundingClientRect().top < 0) window.scrollTo({ top: 0, behavior: 'smooth' }) // on a phone Play sits below the board
  panel.reset(game.fly)
  panel.say(FLIES[game.fly].hello)
  panel.say(side === 'w' ? "You're white, your move." : "I'm white, so I'll start.")
  refresh()
  if (side === 'b') return flyTurn(null, START, id)
  try {
    const pos = await api.legal(START)
    if (id === game.id) yourTurn(pos)
  } catch (err) {
    if (id === game.id) serverError(err)
  }
}

function yourTurn(pos) {
  game.live = pos
  game.thinking = false
  if (pos.over) return finish(pos.over)
  refresh()
  if (!tryPremove(pos)) updateBoardInput()
}

/** A premove made while the fly thought goes now if it's legal (a pawn on the last rank becomes a
 *  queen, as on chess.com), otherwise it slides back. */
function tryPremove(pos) {
  const pre = board.premove
  if (!pre) return false
  const options = pos.legal.filter((m) => m.from === pre.from && m.to === pre.to)
  const move = options.find((m) => m.promo === 'q') ?? options[0]
  if (!move) {
    board.clearPremove()
    return false
  }
  board.disable() // no new premove while this one goes through
  setTimeout(() => { // a beat later, so you see the fly's move first
    if (board.premove !== pre || game.view !== null || game.live !== pos || !canMove()) return // called off meanwhile
    board.premove = null // the piece already stands on its new square
    board.dropped = board.pieces.get(pre.from)
    yourMove(move)
  }, 120)
  return true
}

function yourMove(move) {
  const before = game.live.fen
  const ply = { by: 'you', san: move.san, uci: move.uci, from: move.from, to: move.to, before, fen: move.after,
    check: /[+#]$/.test(move.san), captured: move.capture ? parseFen(before).get(move.capture) : null, opening: move.opening }
  game.plies.push(ply)
  game.view = null
  game.hint = null
  board.setArrows([])
  board.set(ply.fen, { last: ply, check: ply.check })
  nameOpening(ply)
  react(ply)
  api.analyse(before, [move.uci]).then((a) => grade(ply, a.moves[move.uci])).catch(() => {})
  const guessed = game.plies.findLast((p) => p.by === 'fly')
  if (guessed?.expect && !guessed.expectVerdicts) { // now that you've moved, Stockfish may grade what it expected
    api.analyse(before, guessed.expect.map((c) => c.uci)).then((a) => {
      guessed.expectVerdicts = a.moves
      refresh()
    }).catch(() => {})
  }
  flyTurn(move.uci, before, game.id)
}

function grade(ply, v) {
  if (!v) return
  ply.verdict = v.verdict
  ply.accuracy = v.accuracy
  if (game.ending) renderResult() // the accuracies on the result card
  refresh()
}

/** The fly's reaction to your move: did it see it coming? */
function react(ply) {
  const guesses = game.plies.findLast((p) => p.by === 'fly')?.expect
  if (!guesses) return
  const rank = guesses.findIndex((c) => c.uci === ply.uci)
  if (rank === 0) panel.say(`I expected ${ply.san}. I gave it ${pct(guesses[0].p)}.`)
  else if (rank > 0 && rank < 5) panel.say(`${ply.san} was my number ${rank + 1} guess, ${pct(guesses[rank].p)}.`)
  else panel.say(`I didn't see ${ply.san} coming.`)
}

async function flyTurn(uci, fen, id) {
  game.thinking = true
  game.pending = null
  clearTimeout(tabTimer)
  panel.userTab = false
  panel.showTab('fly')
  panel.typing(true)
  updateBoardInput() // premoves while it thinks
  refresh()
  let res
  try {
    res = await api.play(fen, uci, game.fly, seenBefore(uci ? game.plies.length - 1 : game.plies.length), game.imagine)
  } catch (err) {
    if (id !== game.id) return
    if (uci) { // the server never took your move, so it comes off the board again
      game.plies.pop()
      const last = game.plies.at(-1)
      board.set(fen, { last: last ?? null, check: last?.check ?? false })
    }
    return serverError(err)
  }
  if (id !== game.id) return
  const f = res.fly
  if (!f) { // your move ended the game
    panel.typing(false)
    return yourTurn(res)
  }
  const ply = { by: 'fly', san: f.move.san, uci: f.move.uci, from: f.move.from, to: f.move.to, before: f.fen, fen: res.fen,
    check: !!res.check, captured: f.move.capture ? parseFen(f.fen).get(f.move.capture) : null, opening: res.opening,
    think: { candidates: f.candidates, value: f.value, trained: f.trained, imagined: f.imagined, verdicts: null }, expect: res.expect,
    expectVerdicts: null }
  game.pending = ply
  refresh()
  api.analyse(f.fen, f.candidates.map((c) => c.uci)).then((a) => {
    ply.think.verdicts = a.moves
    grade(ply, a.moves[ply.uci])
  }).catch(() => {})
  const bytes = await api.activity(f.trace).catch(() => null)
  if (id !== game.id) return
  game.activity = f.activity
  if (brain) {
    if (bytes) brain.setActivity(new Uint8Array(bytes))
    await brain.play(REPLAY) // watch the board travel from the eye to the legs
  }
  if (id !== game.id) return
  game.pending = null
  game.plies.push(ply)
  game.view = null
  board.set(ply.fen, { last: ply, check: ply.check })
  nameOpening(ply)
  panel.say(comment(ply))
  if (res.expect) {
    tabTimer = setTimeout(() => { // then show what it expects from you, unless you picked a tab yourself
      if (id === game.id && !panel.userTab && game.plies.at(-1) === ply) panel.showTab('you')
    }, 1500)
  }
  yourTurn(res)
}

/** What the fly says about its own move. */
function comment(ply) {
  const { candidates, value, trained, imagined } = ply.think // imagined: per move, what it saw one move ahead
  const top = candidates[0], mine = candidates.find((c) => c.uci === ply.uci)
  let text
  if (ply.san.endsWith('#')) text = imagined ? `${ply.san}! I imagined the board after it: checkmate.` : `${ply.san}!`
  else if (top.uci === ply.uci) text = `${ply.san}. I gave it ${pct(top.p)}.`
  else if (imagined?.[top.uci]?.end === 'draw') text = `${top.san} was my favourite (${pct(top.p)}), but I imagined the board after it: a draw. So ${ply.san}.`
  else if (imagined) text = `${top.san} was my favourite (${pct(top.p)}), but one move ahead ${ply.san} looked better to me.`
  else text = `${top.san} was my favourite (${pct(top.p)}), but it repeats a position. So ${ply.san}${mine ? `, ${pct(mine.p)}` : ''}.`
  if (value !== null && value !== undefined && !ply.san.endsWith('#')) {
    text += value > 0.5 ? " I think I'm winning." : value > 0.15 ? " I think I'm a bit better." : value >= -0.15 ? " I think it's about even."
      : value >= -0.5 ? " I think I'm a bit worse." : " I think I'm losing."
  }
  if (!trained && !game.warned) {
    text += " Nobody has trained me yet, so I'm guessing :c"
    game.warned = true
  }
  return text
}

function finish(over) {
  const [how, result] = over.split(', ')
  const winner = result === 'white wins' ? 'w' : result === 'black wins' ? 'b' : null
  Object.assign(game, { state: 'over', thinking: false, pending: null, hint: null, result: winner ? (winner === 'w' ? '1-0' : '0-1') : '1/2-1/2' })
  clearTimeout(tabTimer)
  resetResign()
  panel.typing(false)
  board.disable()
  board.clearPremove()
  board.setArrows([])
  const youWon = winner === game.side
  const lines = {
    checkmate: youWon ? "Checkmate. Well played! I only ever look one move ahead :c" : 'Checkmate! Not bad for a brain the size of a poppy seed :D',
    resignation: 'You resigned. Thanks for the game!',
    stalemate: "Stalemate, so it's a draw.",
    'insufficient material': 'Nobody can checkmate any more. A draw.',
    'threefold repetition': "The same position three times, so it's a draw.",
    'fifty moves': 'Fifty moves without a capture or a pawn move. A draw.',
  }
  panel.say(lines[how] ?? `${how}, ${result}.`)
  showResult(winner === null ? 'Draw' : youWon ? 'You won!' : `${flyName()} won`, REASONS[how] ?? how, winner)
  refresh()
}

function showResult(title, reason, winner) {
  game.ending = { title, reason, winner }
  renderResult()
  $('result').hidden = false
  $('result-rematch').focus({ preventScroll: true })
}

function renderResult() {
  const { title, reason, winner } = game.ending
  $('result-title').textContent = title
  $('result-reason').textContent = reason
  const acc = (by) => {
    const list = game.plies.filter((p) => p.by === by && p.accuracy != null).map((p) => p.accuracy)
    return list.length ? (list.reduce((a, b) => a + b, 0) / list.length).toFixed(1) : null
  }
  const figure = (isFly) => {
    const fig = el('figure', (winner && (winner === game.side) !== isFly) ? 'won' : '')
    const avatar = el('span', isFly ? 'avatar' : 'avatar you')
    if (isFly) avatar.dataset.fly = game.fly
    const cap = el('figcaption'), accuracy = acc(isFly ? 'fly' : 'you')
    cap.append(el('b', '', isFly ? flyName() + (flyElo() !== null ? ` (${flyElo()})` : '') : 'You'))
    if (accuracy) cap.append(`accuracy ${accuracy}`)
    fig.append(avatar, cap)
    return fig
  }
  $('result-players').replaceChildren(figure(false), figure(true))
}

function hideResult() {
  $('result').hidden = true
}

async function takeback() {
  const cut = game.plies.findLastIndex((p) => p.by === 'you')
  if (cut < 0 || game.state === 'setup') return
  const id = ++game.id
  board.disable() // until the server has sent the legal moves again
  board.clearPremove(false)
  clearTimeout(tabTimer)
  resetResign()
  game.plies = game.plies.slice(0, cut)
  Object.assign(game, { state: 'playing', view: null, thinking: false, pending: null, hint: null, result: '*', live: null, ending: null })
  const last = game.plies.at(-1)
  hideResult()
  board.setArrows([])
  board.set(fenAt(game.plies.length - 1), { last: last ?? null, check: last?.check ?? false })
  panel.typing(false)
  panel.say("Takeback? Sure. I only see the board, so I won't remember anyway.")
  panel.showTab(last ? 'you' : 'fly')
  refresh()
  try {
    const pos = await api.legal(fenAt(game.plies.length - 1))
    if (id === game.id) yourTurn(pos)
  } catch (err) {
    if (id === game.id) serverError(err)
  }
}

async function hint() {
  if (!canMove() || hintPending) return
  goto(game.plies.length - 1)
  const id = game.id, n = game.plies.length, fen = game.live.fen
  hintPending = true
  renderFooter()
  const a = await api.analyse(fen, []).catch(() => null)
  hintPending = false
  renderFooter()
  if (id !== game.id || game.plies.length !== n || !canMove() || !a?.best) return // you moved, or the game changed, meanwhile
  game.hint = { from: a.best.uci.slice(0, 2), to: a.best.uci.slice(2, 4), kind: 'hint' }
  board.setArrows([game.hint])
  panel.say(`Psst. Stockfish would play ${a.best.san} here.`)
}

function resign() {
  const b = $('resign')
  if (!b.classList.contains('confirm')) { // one more click to be sure
    b.classList.add('confirm')
    b.querySelector('span').textContent = 'Resign?'
    resignTimer = setTimeout(resetResign, 3000)
    return
  }
  resetResign()
  game.id++ // a think still on its way is ignored
  finish(`resignation, ${colorName(other(game.side))} wins`)
}

function resetResign() {
  clearTimeout(resignTimer)
  $('resign').classList.remove('confirm')
  $('resign').querySelector('span').textContent = 'Resign'
}

/** Look at the position after ply i (-1: the start). One step at a time slides, a jump doesn't. */
function goto(i) {
  if (!game.started) return
  const lastIdx = game.plies.length - 1, from = shownIndex()
  i = Math.max(-1, Math.min(lastIdx, i))
  if (i === from) return
  game.view = i === lastIdx ? null : i
  const ply = game.plies[i]
  board.set(fenAt(i), { last: ply ?? null, check: ply?.check ?? false, animate: Math.abs(i - from) === 1 })
  board.setArrows(game.view === null && game.hint ? [game.hint] : [])
  updateBoardInput()
  refresh()
}

function flip() {
  board.setOrientation(other(board.orientation))
  renderPlayers()
}

function serverError(err) {
  game.thinking = false
  game.pending = null
  board.clearPremove()
  panel.typing(false)
  panel.say(`The server said no: ${String(err.message).slice(0, 140)}`, 'error')
  refresh()
  updateBoardInput()
}

function downloadPgn() {
  if (!game.plies.length) return
  const fly = flyName(), d = new Date(), pad = (n) => String(n).padStart(2, '0')
  const date = `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}`
  const tags = [['Event', 'fly-chess'], ['Site', location.host], ['Date', date], ['Round', '-'], ['White', game.side === 'w' ? 'You' : fly],
    ['Black', game.side === 'b' ? 'You' : fly], ['Result', game.result]]
  if (flyElo() !== null) tags.push([game.side === 'w' ? 'BlackElo' : 'WhiteElo', String(flyElo())])
  const opening = openingAt(game.plies.length - 1)
  if (opening) tags.push(['ECO', opening.eco], ['Opening', opening.name])
  if (game.state === 'over' && game.ending) {
    const { title, reason, winner } = game.ending
    tags.push(['Termination', winner ? `${title.replace('!', '')} ${reason}` : `Game drawn ${reason}`])
  }
  const words = [...game.plies.map((p, i) => (i % 2 ? p.san : `${i / 2 + 1}. ${p.san}`)), game.result]
  const lines = [''] // moves wrapped at 80 characters, as PGN likes
  for (const w of words) lines[lines.length - 1].length + w.length + 1 > 80 ? lines.push(w) : (lines[lines.length - 1] += (lines.at(-1) ? ' ' : '') + w)
  const text = `${tags.map(([k, v]) => `[${k} "${v.replace(/["\\]/g, '\\$&')}"]`).join('\n')}\n\n${lines.join('\n')}\n`
  const a = el('a')
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/x-chess-pgn' }))
  a.download = `fly-chess-${date.replaceAll('.', '-')}.pgn`
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
}

// ---- controls -------------------------------------------------------------------------------

// The second click of a double click lands on whatever button took this one's place: ignore it.
const click = (fn) => (e) => { if (e.detail < 2) fn() }
board.onMove = (move) => yourMove(move)
for (const b of document.querySelectorAll('.side-grid button')) b.onclick = () => { setup.pick = b.dataset.side; renderSetup() }
$('imagine').onchange = (e) => { setup.imagine = e.target.checked; renderSetup() }
$('play-btn').onclick = click(() => startGame())
$('back-btn').onclick = click(() => showGame())
$('new').onclick = $('new-game').onclick = $('result-new').onclick = click(() => { hideResult(); showSetup() })
$('rematch').onclick = $('result-rematch').onclick = click(() => {
  Object.assign(setup, { fly: game.fly, pick: game.pick }) // the same fly and colour choice, whatever the picker shows
  startGame()
})
$('result-close').onclick = () => hideResult()
$('resign').onclick = click(() => resign())
$('takeback').onclick = click(() => takeback())
$('hint').onclick = () => hint()
$('flip').onclick = () => flip()
$('prev').onclick = () => goto(shownIndex() - 1)
$('next').onclick = () => goto(shownIndex() + 1)
$('pgn').onclick = () => downloadPgn()
$('step').oninput = (e) => brain?.setStep(Number(e.target.value))
const foldBrain = (folded) => { // on a short screen the brain can fold away, so the move list fits
  $('brain-card').classList.toggle('folded', folded)
  $('brain-body').hidden = folded
  $('brain-fold').setAttribute('aria-expanded', String(!folded))
  $('brain-fold').title = folded ? 'Show the brain' : 'Hide the brain'
  $('brain-fold').setAttribute('aria-label', $('brain-fold').title)
  save('brain', folded ? 'folded' : 'open')
}
$('brain-fold').onclick = () => foldBrain(!$('brain-body').hidden)
foldBrain(saved('brain', innerHeight < 800 || innerWidth <= 960 ? 'folded' : 'open') === 'folded')
$('train-btn').onclick = () => {
  $('training').hidden = false
  $('train-close').focus({ preventScroll: true })
  refreshCharts($('charts'))
  chartTimer = setInterval(() => refreshCharts($('charts')), 10_000)
}
const closeTraining = () => {
  $('training').hidden = true
  clearInterval(chartTimer)
}
$('train-close').onclick = closeTraining
$('training').onclick = (e) => { if (e.target === $('training')) closeTraining() }
addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return
  if (e.key === 'Escape') { // the topmost thing closes first
    if (!$('training').hidden) return closeTraining()
    if (board.promoting) return board.cancelPromo?.()
    if (!$('result').hidden) return hideResult()
    return board.select(null)
  }
  if (!$('training').hidden) return
  const keys = { ArrowLeft: () => goto(shownIndex() - 1), ArrowRight: () => goto(shownIndex() + 1), Home: () => goto(-1),
    End: () => goto(game.plies.length - 1), f: flip }
  if (!keys[e.key] || (e.target.closest?.('input') && e.key !== 'f')) return // the brain's step slider keeps its arrow keys
  e.preventDefault()
  keys[e.key]()
})

// ---- boot -------------------------------------------------------------------------------------

async function boot() {
  board.set(START, { animate: false })
  buildFlyBox()
  showSetup()
  fetch('ratings.json').then((r) => (r.ok ? r.json() : null)).then((data) => { // each fly's Elo
    ratings = data?.ratings ?? {}
    buildFlyBox()
    renderPlayers()
  }).catch(() => {})
  api.metrics().then((m) => { // the latest numbers from training, for the fly picker
    const last = (run, field) => m[run]?.findLast((row) => row[field] !== undefined)?.[field]
    metrics = { dl: { top1: last('dl', 'top1') }, day: { top1: last('dl-day', 'top1') }, bio: { val_pair_acc: last('bio', 'val_pair_acc') } }
    renderSetup()
  }).catch(() => {})
  api.flies().then((ready) => { // a fly whose training hasn't finished yet stays out of the picker
    for (const kind of Object.keys(FLIES)) if (ready[kind] === false) delete FLIES[kind]
    if (!(setup.fly in FLIES)) setup.fly = 'dl'
    buildFlyBox()
  }).catch(() => {})
  window.flyChess = { board, panel, game, goto } // to poke at it from the browser console
  const [meta, points] = await Promise.all([api.brain(), api.points()])
  try {
    brain = new BrainView($('brain'), meta, points)
  } catch (err) { // no WebGL: the game works without the brain
    console.error(err)
    $('brain-card').hidden = true
    return
  }
  window.flyChess.brain = brain
  panel.regions = meta.regions
  panel.showRegions(null, 0)
  let shown = null
  brain.onStep = (s) => { // called every frame of a replay; the text and bars only change once a step
    const step = Math.round(s)
    if (shown && shown.step === step && shown.activity === game.activity) return
    shown = { step, activity: game.activity }
    $('step-label').textContent = `step ${step + 1} of ${meta.steps}`
    $('step').value = String(step)
    panel.showRegions(game.activity, step)
  }
  let last = performance.now()
  const frame = (now) => {
    requestAnimationFrame(frame)
    const dt = Math.min((now - last) / 1000, 0.05)
    last = now
    brain.render(dt)
  }
  requestAnimationFrame(frame)
}

boot().catch((err) => {
  console.error(err)
  $('hero-bubble').textContent = `Could not reach the server: ${err.message}`
  $('hero-bubble').classList.add('error')
})

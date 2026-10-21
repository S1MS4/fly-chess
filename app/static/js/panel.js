// The right panel during a game: the fly's chat bubbles, what it was choosing between and
// what it expects you to play (with Stockfish's verdict on each), the opening, both players'
// accuracy and the move list, which you can click to look back through the game.

import { pieceUrl } from './board.js'

const VERDICT = { best: 'Best move', good: 'Good', inaccuracy: 'Inaccuracy', mistake: 'Mistake', blunder: 'Blunder' }
const GLYPH = { best: '★', good: '✓', inaccuracy: '?!', mistake: '?', blunder: '??' }
const PIECE = { K: 'king', Q: 'queen', R: 'rook', B: 'bishop', N: 'knight' }
const SHOWN = 5 // guesses listed
const $ = (id) => document.getElementById(id)

function el(tag, cls, text) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

export const pct = (p) => (p >= 0.995 ? '>99%' : p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`)

/** 'Nxf3+' -> a knight figurine and 'xf3+', the way chess.com writes moves. */
function figurine(san) {
  const frag = document.createDocumentFragment()
  const [, piece, rest, promo] = /^([KQRBN])?(.*?)(?:=([QRBN]))?([+#]?)$/.exec(san) ?? [, null, san]
  const fig = (code) => {
    const i = el('i', 'fig')
    i.style.setProperty('--piece', `url(${pieceUrl(code)})`)
    i.setAttribute('role', 'img')
    i.setAttribute('aria-label', PIECE[code])
    return i
  }
  if (piece) frag.append(fig(piece))
  frag.append(promo ? `${rest}=` : rest)
  if (promo) frag.append(fig(promo))
  if (/[+#]$/.test(san)) frag.append(san.at(-1))
  return frag
}

function verdictIcon(kind) {
  const i = el('i', `verdict ${kind}`)
  i.setAttribute('aria-hidden', 'true') // the move's label says it in words
  if (kind === 'book') i.innerHTML = '<svg><use href="#i-book"/></svg>'
  else i.textContent = GLYPH[kind]
  return i
}

export class Panel {
  constructor({ onGoto, onArrow }) {
    this.onGoto = onGoto
    this.onArrow = onArrow
    this.tab = 'fly'
    this.userTab = false // you picked a tab during this turn, so it doesn't switch by itself
    this.guesses = { fly: null, you: null }
    this.blurred = (() => { try { return localStorage.getItem('blur') === '1' } catch { return false } })()
    $('blur').onclick = () => this.setBlur(!this.blurred)
    $('tab-fly').onclick = () => this.showTab('fly', true)
    $('tab-you').onclick = () => this.showTab('you', true)
    this.setBlur(this.blurred)
  }

  reset(fly) {
    $('chat-avatar').dataset.fly = fly
    $('bubbles').replaceChildren()
    this.userTab = false
    this.showTab('fly')
  }

  // ---- the fly's chat ----------------------------------------------------------------

  say(text, kind = '') {
    this.typing(false)
    $('bubbles').append(el('p', `bubble ${kind}`, text))
    this.trim()
  }

  typing(on) {
    $('bubbles').querySelector('.typing')?.remove()
    if (!on) return
    const b = el('p', 'bubble typing')
    b.innerHTML = '<span class="dots" aria-label="thinking"><i></i><i></i><i></i></span>'
    $('bubbles').append(b)
    this.trim()
  }

  trim() { // two bubbles at most, the dots while it thinks count as one, so the panel doesn't jump
    const box = $('bubbles')
    while (box.children.length > 2) box.firstElementChild.remove()
  }

  // ---- what the fly was choosing between, and what it expects from you -------------------

  setBlur(on) {
    this.blurred = on
    $('blur').setAttribute('aria-pressed', String(on))
    $('blur').querySelector('use').setAttribute('href', on ? '#i-eye-off' : '#i-eye')
    $('guesses').classList.toggle('blurred', on)
    if (on) this.onArrow(null)
    try { localStorage.setItem('blur', on ? '1' : '0') } catch { /* private window: fine */ }
  }

  showTab(tab, byYou = false) {
    this.tab = tab
    if (byYou) this.userTab = true
    $('tab-fly').setAttribute('aria-pressed', String(tab === 'fly'))
    $('tab-you').setAttribute('aria-pressed', String(tab === 'you'))
    this.renderGuesses()
  }

  /** Each: {title, candidates: [{uci, san, p}], verdicts: {uci: {accuracy, verdict}} or null while Stockfish looks,
   *  hidden (grades kept back until you've moved), played, value (the fly's own rating of the position)}. */
  setGuesses(fly, you) {
    this.guesses = { fly, you }
    this.renderGuesses()
  }

  renderGuesses() {
    const g = this.guesses[this.tab]
    const list = $('guesses')
    list.replaceChildren()
    this.onArrow(null)
    $('guess-title').textContent = g?.title ?? ''
    const value = g?.value
    $('guess-value').hidden = value === null || value === undefined
    if (!$('guess-value').hidden) $('guess-value').textContent = `eval ${value >= 0 ? '+' : ''}${value.toFixed(2)}`
    if (!g?.candidates?.length) return
    const shown = g.candidates.slice(0, SHOWN)
    for (const c of shown) {
      const li = el('li', c.uci === g.played ? 'move played' : 'move')
      const san = el('span', 'san')
      san.append(figurine(c.san))
      const bar = el('span', 'bar'), fill = el('i')
      fill.style.width = `${Math.max(2, c.p * 100)}%`
      bar.append(fill)
      const v = g.hidden ? null : g.verdicts?.[c.uci]
      const chip = el('span', `chip ${v ? v.verdict : g.hidden ? 'hidden' : g.verdicts ? '' : 'wait'}`)
      if (v) chip.append(el('i', '', GLYPH[v.verdict]), String(Math.round(v.accuracy)))
      else chip.textContent = g.hidden ? '?' : g.verdicts ? '-' : '…'
      chip.title = v ? `${VERDICT[v.verdict]}: ${Math.round(v.accuracy)}% accuracy (Stockfish, Lichess's formula)`
        : g.hidden ? "Stockfish's grade shows up once you've moved" : g.verdicts ? 'not checked' : 'Stockfish is looking at it'
      li.title = `${c.san}: the fly gives it ${(c.p * 100).toFixed(1)}%${c.uci === g.played ? ', and played it' : ''}`
      li.append(san, bar, el('span', 'pct', pct(c.p)), chip)
      li.onmouseenter = () => { if (!this.blurred) this.onArrow({ from: c.uci.slice(0, 2), to: c.uci.slice(2, 4), kind: 'fly' }) }
      li.onmouseleave = () => this.onArrow(null)
      list.append(li)
    }
    const rest = 1 - shown.reduce((s, c) => s + c.p, 0)
    if (rest >= 0.005) list.append(el('li', 'note', `Every other move together: ${pct(rest)}`))
  }

  // ---- opening, accuracy, moves --------------------------------------------------------

  setOpening(op) {
    $('opening-name').textContent = op ? op.name : 'Starting position'
    $('opening').title = op ? `${op.eco}, from Lichess's opening names` : ''
  }

  renderAccuracy(plies, side) {
    const avg = (by) => {
      const list = plies.filter((p) => p.by === by && p.accuracy != null).map((p) => p.accuracy)
      return list.length ? (list.reduce((a, b) => a + b, 0) / list.length).toFixed(1) : '-'
    }
    const chip = (who, label, color) => { // white's chip is white, black's is black, like chess.com's game review
      const c = el('span', color)
      c.append(`${label} `, el('b', '', avg(who)))
      c.title = `${label === 'You' ? 'Your' : "The fly's"} accuracy: the average of each move's, from Stockfish with Lichess's formula`
      return c
    }
    const you = chip('you', 'You', side), fly = chip('fly', 'Fly', side === 'w' ? 'b' : 'w')
    $('accuracy').replaceChildren(...(side === 'w' ? [you, fly] : [fly, you]))
  }

  renderMoves(plies, current) {
    const box = $('history')
    box.replaceChildren()
    let currentButton = null
    for (let i = 0; i < plies.length; i += 2) {
      const row = el('div', 'mrow')
      row.setAttribute('role', 'listitem')
      row.append(el('span', 'mnum', `${i / 2 + 1}.`))
      for (const j of [i, i + 1]) {
        const p = plies[j]
        if (!p) break
        const b = el('button', j === current ? 'mply current' : 'mply')
        b.type = 'button'
        b.append(figurine(p.san))
        const kind = p.opening ? 'book' : p.verdict
        if (kind) b.append(verdictIcon(kind))
        const graded = p.verdict ? `Stockfish: ${VERDICT[p.verdict]}, ${Math.round(p.accuracy)}% accuracy` : ''
        b.title = [p.opening ? 'Book move: a named opening position' : '', graded].filter(Boolean).join('. ') || p.san
        b.setAttribute('aria-label', [p.san, p.opening ? 'book move' : '', p.verdict ? VERDICT[p.verdict] : ''].filter(Boolean).join(', '))
        b.onclick = () => this.onGoto(j)
        row.append(b)
        if (j === current) currentButton = b
      }
      box.append(row)
    }
    if (currentButton && box.scrollHeight > box.clientHeight) { // keep the move you're on in sight, inside the list only
      const r = currentButton.getBoundingClientRect(), s = box.getBoundingClientRect()
      if (r.bottom > s.bottom) box.scrollTop += r.bottom - s.bottom + 6
      else if (r.top < s.top) box.scrollTop -= s.top - r.top + 6
    } else if (!currentButton && current === plies.length - 1) box.scrollTop = box.scrollHeight
  }

  /** Region bars: how much each region's firing moved away from usual at this step. Before the first
   *  think, just the names. */
  showRegions(activity, step) {
    const frame = activity?.[Math.round(step)] ?? Object.fromEntries((this.regions ?? []).map((r) => [r, 0]))
    const top = Math.max(1e-6, ...(activity ?? []).flatMap((f) => Object.values(f)))
    $('regions').replaceChildren(...Object.entries(frame).map(([name, v]) => {
      const li = el('li'), bar = el('span', 'bar'), fill = el('i')
      fill.style.width = `${(v / top) * 100}%`
      bar.append(fill)
      li.append(el('span', 'name', name), bar)
      li.title = `${name}: ${v.toFixed(4)} average change in firing rate`
      return li
    }))
  }
}

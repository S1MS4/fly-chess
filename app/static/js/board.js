// A flat board in chess.com's colours with Cburnett's pieces. It draws whatever FEN it's given
// and slides the pieces that moved (it compares the old position with the new one), so moves,
// takebacks and stepping through the game all animate the same way. Every chess rule lives on
// the server; the board only knows where pieces stand and which moves it was told are legal.
// You move by clicking a piece and then a square, or by dragging it. While the fly thinks you can
// premove: pick a piece up and put it where you'll want it, and it's played the moment the fly has
// moved. Right-click marks a square, right-drag draws an arrow, like on chess.com.

const FILES = 'abcdefgh'
const SLIDE = 200 // ms, how long a move slides; the same as .piece's transition in style.css
const LIGHT = '#ebecd0', DARK = '#739552'
const ARROWS = { you: ['#ffaa00', 0.8], hint: ['#81b64c', 0.9], fly: ['#4f9de0', 0.85] } // colour, opacity
const NS = 'http://www.w3.org/2000/svg'

export const pieceUrl = (code) => `pieces/Chess_${code.toLowerCase()}${code === code.toUpperCase() ? 'l' : 'd'}t45.svg`

/** 'rnbqkbnr/... w ...' -> Map {'a8' => 'r', ...}, uppercase for white. */
export function parseFen(fen) {
  const out = new Map()
  fen.split(' ')[0].split('/').forEach((row, i) => {
    let f = 0
    for (const ch of row) {
      if (/\d/.test(ch)) f += Number(ch)
      else out.set(FILES[f++] + (8 - i), ch)
    }
  })
  return out
}

function el(tag, cls, parent) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  parent?.append(e)
  return e
}

function svg(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag)
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v)
  parent?.append(e)
  return e
}

const distance = (a, b) => Math.hypot(FILES.indexOf(a[0]) - FILES.indexOf(b[0]), a[1] - b[1])
const colorOf = (code) => (code === code.toUpperCase() ? 'w' : 'b')
const ORTHO = [[1, 0], [-1, 0], [0, 1], [0, -1]], DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
const JUMPS = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]]

export class Board {
  constructor(root) {
    this.root = root
    this.orientation = 'w' // the colour at the bottom
    this.pieces = new Map() // square -> element
    this.legal = [] // the moves you may play right now, from the server
    this.selected = null
    this.drag = null
    this.marks = new Set() // right-clicked squares
    this.userArrows = new Map() // 'e2e4' -> arrow, right-dragged
    this.shown = [] // arrows from outside: the hint, the fly's guess under the mouse
    this.last = null
    this.checkSquare = null
    this.leaving = new Set() // taken pieces, waiting for the taker to land
    this.premoveColor = null // set while you may premove: your colour
    this.premove = null // {from, to, code}: the move waiting for the fly
    this.onMove = () => {}

    // Layers, bottom to top: squares, highlights, coordinates, pieces, move dots, arrows.
    const squares = svg('svg', { class: 'squares', viewBox: '0 0 8 8', 'shape-rendering': 'crispEdges', 'aria-hidden': 'true' }, root)
    svg('rect', { width: 8, height: 8, fill: LIGHT }, squares)
    for (let i = 0; i < 64; i++) if ((i % 8 + (i >> 3)) % 2) svg('rect', { x: i % 8, y: i >> 3, width: 1, height: 1, fill: DARK }, squares)
    this.highlights = el('div', 'layer highlights', root)
    this.coords = svg('svg', { class: 'coords', viewBox: '0 0 100 100', 'aria-hidden': 'true' }, root)
    this.pieceLayer = el('div', 'layer pieces', root)
    this.hintLayer = el('div', 'layer hints', root)
    this.arrowLayer = svg('svg', { class: 'arrows', viewBox: '0 0 8 8', 'aria-hidden': 'true' }, root)
    this.drawCoords()

    root.addEventListener('pointerdown', (e) => this.down(e))
    root.addEventListener('pointermove', (e) => this.move(e))
    root.addEventListener('pointerup', (e) => this.up(e))
    root.addEventListener('pointercancel', () => this.cancelDrag())
    root.addEventListener('contextmenu', (e) => e.preventDefault())
    new ResizeObserver(() => root.style.setProperty('--sq', `${root.clientWidth / 8}px`)).observe(root)
  }

  // ---- squares and positions -------------------------------------------------------

  /** Column and row on screen, 0..7 from the top left. */
  xy(sq) {
    const f = FILES.indexOf(sq[0]), r = Number(sq[1]) - 1
    return this.orientation === 'w' ? [f, 7 - r] : [7 - f, r]
  }

  put(node, sq) {
    const [x, y] = this.xy(sq)
    node.style.transform = `translate(${x * 100}%, ${y * 100}%)`
  }

  squareAt(clientX, clientY) {
    const r = this.root.getBoundingClientRect()
    const x = Math.floor(((clientX - r.left) / r.width) * 8), y = Math.floor(((clientY - r.top) / r.height) * 8)
    if (x < 0 || x > 7 || y < 0 || y > 7) return null
    return this.orientation === 'w' ? FILES[x] + (8 - y) : FILES[7 - x] + (y + 1)
  }

  codeAt(sq) {
    return this.pieces.get(sq)?.dataset.code ?? null
  }

  drawCoords() {
    this.coords.replaceChildren()
    for (let i = 0; i < 8; i++) { // ranks down the left edge, files along the bottom, in the other square's colour
      const rank = this.orientation === 'w' ? 8 - i : i + 1, file = this.orientation === 'w' ? FILES[i] : FILES[7 - i]
      svg('text', { x: 0.8, y: i * 12.5 + 3.4, fill: i % 2 ? LIGHT : DARK }, this.coords).textContent = rank
      svg('text', { x: i * 12.5 + 11.9, y: 99, 'text-anchor': 'end', fill: i % 2 ? DARK : LIGHT }, this.coords).textContent = file
    }
  }

  setOrientation(color) {
    if (color === this.orientation) return
    this.cancelPromo?.()
    for (const piece of this.leaving) piece.remove()
    this.leaving.clear()
    this.orientation = color
    this.snap([...this.pieces].map(([sq, piece]) => [piece, this.premove?.from === sq ? this.premove.to : sq]))
    this.drawCoords()
    this.paint()
    this.drawArrows()
  }

  /** Move pieces without the slide (a flip, a drop). */
  snap(list) {
    for (const [piece, sq] of list) {
      piece.classList.add('instant')
      this.put(piece, sq)
    }
    void this.root.offsetWidth // apply the jump now, so turning the transition back on doesn't animate it
    for (const [piece] of list) piece.classList.remove('instant')
  }

  // ---- drawing a position ----------------------------------------------------------

  /** Show `fen`. The pieces that moved slide there; `last` is the move to highlight and `check`
   *  says the side to move is in check. A piece you just dropped lands without sliding. */
  set(fen, { last = null, check = false, animate = true } = {}) {
    this.cancelPromo?.()
    if (!animate) {
      for (const piece of this.leaving) piece.remove()
      this.leaving.clear()
    }
    const pre = this.premove, waiting = pre && this.pieces.get(pre.from)
    if (waiting) this.snap([[waiting, pre.from]]) // a premoved piece stands at home while the position changes
    const next = parseFen(fen), old = this.pieces, kept = new Map()
    for (const [sq, code] of next) if (old.get(sq)?.dataset.code === code) kept.set(sq, old.get(sq))
    const gone = [...old].filter(([sq, piece]) => kept.get(sq) !== piece)
    const take = (code, sq) => { // the vanished piece of this kind closest to sq
      let best = null
      for (const g of gone) if (g[1]?.dataset.code === code && (!best || distance(g[0], sq) < distance(best[0], sq))) best = g
      const piece = best?.[1] ?? null
      if (best) best[1] = null
      return piece
    }
    const slides = [], jumps = [] // jumps: new pieces, a piece you dropped, everything when not animating
    for (const [sq, code] of next) {
      if (kept.has(sq)) continue
      let piece = take(code, sq)
      if (!piece && /[QRBNqrbn]/.test(code) && /[18]/.test(sq[1])) { // a promotion: the pawn walks on and turns into the new piece
        piece = take(code === code.toUpperCase() ? 'P' : 'p', sq)
        if (piece) this.dress(piece, code)
      }
      if (piece) (animate && piece !== this.dropped ? slides : jumps).push([piece, sq])
      else jumps.push([(piece = this.spawn(code)), sq])
      kept.set(sq, piece)
    }
    for (const [, piece] of gone) { // taken pieces go once the taker lands
      if (!piece) continue
      if (!animate) piece.remove()
      else {
        this.leaving.add(piece)
        setTimeout(() => this.leaving.delete(piece) && piece.remove(), SLIDE)
      }
    }
    this.snap(jumps)
    for (const [piece, sq] of [...slides, ...jumps.filter(([p]) => p === this.dropped)]) {
      piece.classList.add('moving') // above the piece it takes
      if (piece !== this.dropped) this.put(piece, sq)
      setTimeout(() => piece.classList.remove('moving'), SLIDE)
    }
    this.dropped = null
    this.pieces = kept
    if (pre) { // and goes back to its new square, unless it was taken
      const piece = kept.get(pre.from)
      if (piece?.dataset.code === pre.code) this.snap([[piece, pre.to]])
      else this.premove = null
    }
    this.last = last
    const king = fen.split(' ')[1] === 'b' ? 'k' : 'K'
    this.checkSquare = check ? [...next].find(([, code]) => code === king)?.[0] ?? null : null
    this.select(null)
  }

  spawn(code) {
    const piece = el('div', 'piece', this.pieceLayer)
    this.dress(piece, code)
    return piece
  }

  dress(piece, code) {
    piece.dataset.code = code
    piece.style.backgroundImage = `url(${pieceUrl(code)})`
  }

  /** Yellow for the last move and the picked-up piece, red for a king in check and right-clicked squares. */
  paint() {
    const yellow = new Set([this.last?.from, this.last?.to, this.selected].filter(Boolean))
    const marks = [...yellow].map((sq) => [sq, 'last'])
    if (this.checkSquare) marks.push([this.checkSquare, 'check'])
    if (this.premove) marks.push([this.premove.from, 'premove'], [this.premove.to, 'premove'])
    for (const sq of this.marks) marks.push([sq, 'mark'])
    if (this.drag?.over) marks.push([this.drag.over, 'hover'])
    this.highlights.replaceChildren(...marks.map(([sq, kind]) => {
      const node = el('div', `hl ${kind}`)
      this.put(node, sq)
      return node
    }))
    this.hintLayer.replaceChildren()
    for (const to of this.dests(this.selected)) { // a dot on an empty square, a ring round a piece you can take
      const capture = this.targets(this.selected).some((m) => m.to === to && m.capture === to)
      this.put(el('div', capture ? 'hint capture' : 'hint', this.hintLayer), to)
    }
  }

  // ---- arrows ----------------------------------------------------------------------

  /** Arrows from outside, [{from, to, kind}] with kind 'hint' or 'fly'. Your own right-drawn ones stay. */
  setArrows(list) {
    this.shown = list
    this.drawArrows()
  }

  drawArrows() {
    const all = [...this.shown, ...this.userArrows.values()]
    if (this.right?.over && this.right.over !== this.right.from) all.push({ from: this.right.from, to: this.right.over, kind: 'you' })
    this.arrowLayer.replaceChildren(...all.map((a) => this.arrow(a)))
  }

  /** chess.com's arrow: straight, or an L for a knight's jump, with the head on the target square's centre. */
  arrow({ from, to, kind }) {
    const [x0, y0] = this.xy(from).map((v) => v + 0.5), [x1, y1] = this.xy(to).map((v) => v + 0.5)
    const [color, opacity] = ARROWS[kind]
    const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0)
    const path = dx * dy === 2 ? [[x0, y0], dy === 2 ? [x0, y1] : [x1, y0], [x1, y1]] : [[x0, y0], [x1, y1]]
    const [ax, ay] = path.at(-2), len = Math.hypot(x1 - ax, y1 - ay), ux = (x1 - ax) / len, uy = (y1 - ay) / len
    const bx = x1 - ux * 0.42, by = y1 - uy * 0.42 // where the head starts
    const g = svg('g', { opacity })
    const shaft = [...path.slice(0, -1), [bx, by]].map((p) => p.join(',')).join(' ')
    svg('polyline', { points: shaft, fill: 'none', stroke: color, 'stroke-width': 0.2, 'stroke-linejoin': 'miter' }, g)
    const head = [[x1, y1], [bx - uy * 0.26, by + ux * 0.26], [bx + uy * 0.26, by - ux * 0.26]]
    svg('polygon', { points: head.map((p) => p.join(',')).join(' '), fill: color }, g)
    return g
  }

  clearAnnotations() {
    if (!this.marks.size && !this.userArrows.size) return
    this.marks.clear()
    this.userArrows.clear()
    this.paint()
    this.drawArrows()
  }

  // ---- your moves ------------------------------------------------------------------

  /** Let you move: `legal` are the server's move descriptions for this position. */
  enable(legal) {
    this.legal = legal
    this.premoveColor = null
    this.root.classList.add('movable')
    for (const [sq, piece] of this.pieces) piece.classList.toggle('can-move', this.targets(sq).length > 0)
  }

  /** Let you premove while the fly thinks: pick up a `color` piece and put it on any square it could
   *  reach. It's played the moment the fly has moved, if it's legal then. */
  enablePremove(color) {
    this.legal = []
    this.premoveColor = color
    this.root.classList.add('movable')
    for (const [, piece] of this.pieces) piece.classList.toggle('can-move', colorOf(piece.dataset.code) === color)
  }

  /** Call a premove off; the piece slides home (or jumps, with animate false). */
  clearPremove(animate = true) {
    const pre = this.premove
    if (!pre) return
    this.premove = null
    const piece = this.pieces.get(pre.from)
    if (piece) animate ? this.put(piece, pre.from) : this.snap([[piece, pre.from]])
    this.paint()
  }

  setPremove(from, to) {
    const piece = this.pieces.get(from)
    this.premove = { from, to, code: piece.dataset.code }
    this.snap([[piece, to]]) // it waits on its new square, the way chess.com shows a premove
    this.select(null)
  }

  /** Every square the piece on `from` could reach on some board, the way chessground does premoves:
   *  a pawn also diagonally, sliding pieces through anything, the king to its castling squares. */
  premoveDests(from) {
    const code = this.codeAt(from)
    if (!code) return []
    const type = code.toLowerCase(), white = colorOf(code) === 'w'
    const f = FILES.indexOf(from[0]), r = Number(from[1]) - 1, out = []
    const add = (df, dr) => {
      if (f + df >= 0 && f + df < 8 && r + dr >= 0 && r + dr < 8) out.push(FILES[f + df] + (r + dr + 1))
    }
    const lines = (dirs) => { for (const [a, b] of dirs) for (let k = 1; k < 8; k++) add(a * k, b * k) }
    if (type === 'p') {
      const dir = white ? 1 : -1
      add(0, dir)
      add(-1, dir)
      add(1, dir)
      if (r === (white ? 1 : 6)) add(0, 2 * dir)
    } else if (type === 'n') JUMPS.forEach(([a, b]) => add(a, b))
    else if (type === 'b') lines(DIAG)
    else if (type === 'r') lines(ORTHO)
    else if (type === 'q') lines([...ORTHO, ...DIAG])
    else {
      [...ORTHO, ...DIAG].forEach(([a, b]) => add(a, b))
      const home = white ? '1' : '8', rook = white ? 'R' : 'r'
      if (from === `e${home}` && this.codeAt(`h${home}`) === rook) out.push(`g${home}`)
      if (from === `e${home}` && this.codeAt(`a${home}`) === rook) out.push(`c${home}`)
    }
    return out
  }

  disable() {
    this.cancelPromo?.()
    this.premoveColor = null
    this.legal = []
    this.root.classList.remove('movable')
    this.cancelDrag()
    this.select(null)
  }

  targets(from) {
    return from ? this.legal.filter((m) => m.from === from) : []
  }

  /** Where the piece on `from` may go now: its legal moves, or while premoving, every square it could reach. */
  dests(from) {
    if (!from) return []
    return this.premoveColor ? this.premoveDests(from) : [...new Set(this.targets(from).map((m) => m.to))]
  }

  canPick(sq) {
    const code = this.codeAt(sq)
    return !!code && (this.premoveColor ? colorOf(code) === this.premoveColor : this.targets(sq).length > 0)
  }

  select(sq) {
    this.selected = sq
    this.paint()
  }

  down(e) {
    if (this.promoting) return
    const sq = this.squareAt(e.clientX, e.clientY)
    if (e.button === 2) { // start an arrow or a mark
      if (this.premove) return this.clearPremove() // or call a premove off, like on chess.com
      if (sq) {
        this.right = { from: sq, over: sq }
        this.root.setPointerCapture(e.pointerId)
      }
      return
    }
    if (e.button !== 0) return
    this.clearAnnotations()
    if (this.premoveColor && this.premove) this.clearPremove(false) // a new premove replaces the old one
    if (!sq) return this.select(null)
    if (this.selected && this.dests(this.selected).includes(sq)) return this.commit(this.selected, sq)
    if (!this.canPick(sq)) return this.select(null)
    const again = this.selected === sq
    this.select(sq)
    this.drag = { piece: this.pieces.get(sq), from: sq, x: e.clientX, y: e.clientY, moving: false, again, over: null }
    this.root.setPointerCapture(e.pointerId)
    e.preventDefault()
  }

  move(e) {
    if (this.right) {
      if (!(e.buttons & 2)) { // the right button went up in a chord with the left one: no arrow
        this.right = null
        return this.drawArrows()
      }
      const over = this.squareAt(e.clientX, e.clientY)
      if (over !== this.right.over) {
        this.right.over = over
        this.drawArrows()
      }
      return
    }
    const d = this.drag
    if (!d) return
    if (!d.moving && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 4) return
    d.moving = true
    d.piece.classList.add('dragging')
    this.root.classList.add('dragging')
    const r = this.root.getBoundingClientRect(), size = r.width / 8
    d.piece.style.transform = `translate(${e.clientX - r.left - size / 2}px, ${e.clientY - r.top - size / 2}px)` // centred under the pointer
    const over = this.squareAt(e.clientX, e.clientY)
    if (over !== d.over) {
      d.over = over
      this.paint()
    }
  }

  up(e) {
    if (this.right && e.button !== 2) { // released in a chord: forget the arrow
      this.right = null
      this.drawArrows()
    }
    if (this.right && e.button === 2) {
      const { from } = this.right, to = this.squareAt(e.clientX, e.clientY)
      this.right = null
      if (from && to === from) this.marks.has(to) ? this.marks.delete(to) : this.marks.add(to)
      else if (from && to) {
        const key = from + to
        this.userArrows.has(key) ? this.userArrows.delete(key) : this.userArrows.set(key, { from, to, kind: 'you' })
      }
      this.paint()
      this.drawArrows()
      return
    }
    const d = this.drag
    if (!d || e.button !== 0) return
    this.drag = null
    d.piece.classList.remove('dragging')
    this.root.classList.remove('dragging')
    if (!d.moving) { // a click: a second click on the picked-up piece puts it down
      if (d.again) this.select(null)
      return
    }
    const to = this.squareAt(e.clientX, e.clientY)
    if (to && to !== d.from && this.dests(d.from).includes(to)) return this.commit(d.from, to, d.piece)
    this.put(d.piece, d.from) // not a legal square: slide back
    this.paint()
  }

  cancelDrag() {
    const d = this.drag
    this.drag = null
    this.right = null
    this.root.classList.remove('dragging')
    if (!d) return
    d.piece.classList.remove('dragging')
    if (this.pieces.get(d.from) === d.piece) this.put(d.piece, d.from)
  }

  async commit(from, to, dropped = null) {
    if (this.premoveColor) return this.setPremove(from, to)
    const options = this.targets(from).filter((m) => m.to === to)
    if (dropped) this.snap([[dropped, to]])
    const move = options.length > 1 ? await this.promote(to, options) : options[0]
    if (!move || !this.legal.includes(move)) { // promotion called off, or the game moved on while you chose
      if (dropped && this.pieces.get(from) === dropped) this.put(dropped, from)
      return this.select(null)
    }
    this.dropped = dropped
    this.disable()
    this.onMove(move, !!dropped)
  }

  /** chess.com's promotion picker: a column of queen, knight, rook, bishop running from the promotion square. */
  promote(to, options) {
    return new Promise((done) => {
      const white = to[1] === '8', [x, y] = this.xy(to)
      const veil = el('div', 'promo', this.root)
      const column = el('div', y === 0 ? 'promo-col' : 'promo-col up', veil)
      column.style.left = `${x * 12.5}%`
      const finish = (move) => {
        veil.remove()
        this.promoting = false
        this.cancelPromo = null
        done(move)
      }
      this.cancelPromo = () => finish(null)
      for (const kind of 'qnrb') {
        const b = el('button', 'promo-piece', column)
        b.style.backgroundImage = `url(${pieceUrl(white ? kind.toUpperCase() : kind)})`
        b.title = { q: 'Queen', n: 'Knight', r: 'Rook', b: 'Bishop' }[kind]
        b.onclick = () => finish(options.find((m) => m.promo === kind))
      }
      const cancel = el('button', 'promo-cancel', column)
      cancel.textContent = '×'
      cancel.title = 'Cancel'
      cancel.onclick = () => finish(null)
      veil.addEventListener('pointerdown', (e) => {
        e.stopPropagation()
        if (e.target === veil) finish(null)
      })
      this.promoting = true
    })
  }
}

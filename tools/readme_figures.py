"""Make the charts in the README (docs/img/*.png) from the brain and the training logs.

  python tools/readme_figures.py

The brain simulations here run on the CPU with scipy, on 48 positions from the
training data, so they don't need TensorFlow or a trained fly.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.io_maps import eye_input, motor_output

OUT = Path("docs/img")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SHOWN = ["photoreceptors", "optic lobe", "visual projection", "kenyon cells", "MBON", "central complex",
         "central brain", "descending"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight"})


def boards(n: int = 48) -> np.ndarray:
    d = np.load("data/positions/packed.npz")
    return np.unpackbits(d["boards"][5000:5000 + n], axis=1).astype(np.float64).T   # (768, n)


def simulate(c, x, version: str, gain: float = 10.0, steps: int = 12):
    """Rates after `steps` steps for three versions of the brain:
    'dead'   the first version: histamine inhibits, inputs add up to 1
    'gain'   photoreceptors excite, inputs add up to `gain`
    'homeo'  photoreceptors excite, every input centred and scaled per step (what the model uses now)"""
    sign = c.sign.copy()
    sign[np.isin(c.pre, c.where("photoreceptors"))] = -1.0 if version == "dead" else 1.0  # histamine, as the first version had it
    m = np.log1p(c.synapses.astype(np.float64))
    m /= np.maximum(np.bincount(c.post, weights=m, minlength=c.n)[c.post], 1e-9)
    scale = gain if version == "gain" else 1.0
    W = sp.csr_matrix((scale * sign * m, (c.post, c.pre)), shape=(c.n, c.n))
    n_idx, f_idx = eye_input(c)
    eye = sp.csr_matrix((np.ones(len(n_idx)), (n_idx, f_idx)), shape=(c.n, 768)) @ x
    r = np.zeros_like(eye)
    for _ in range(steps):
        u = W @ r + eye
        if version == "homeo":
            u = 0.5 * (u - u.mean(1, keepdims=True)) / np.maximum(u.std(1, keepdims=True), 0.02) + 0.3
        r = 0.5 * r + 0.5 * np.maximum(np.tanh(u), 0)
    return r


def signal_figure(c, x):
    """How much the board moves each region at step 12 (spread across positions), per version."""
    versions = [("dead", "First version (histamine inhibits)", MUTED), ("gain", "Photoreceptors excite, gain 10", ORANGE),
                ("homeo", "Homeostasis (what it uses now)", BLUE)]
    spread = {v: [simulate(c, x, v)[c.where(r)].std(1).mean() for r in SHOWN] for v, _, _ in versions}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(SHOWN))
    for k, (v, label, color) in enumerate(versions):
        ax.barh(y + (k - 1) * 0.27, spread[v], height=0.25, color=color, label=label)
    ax.set_yticks(y, SHOWN)
    ax.invert_yaxis()
    ax.text(0.002, 1 - 0.27, "first version: exactly 0 from here down", color=MUTED, fontsize=8.5, va="center")
    ax.set_xlabel("how much the board changes the firing (spread across positions, step 12)")
    ax.set_title("The board has to reach the legs before anything can learn")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "signal.png")
    return spread


def gain_figure(c, x):
    dn = motor_output(c)
    gains = [1, 2, 3, 4, 6, 8, 10, 12]
    sig = [simulate(c, x, "gain", g)[dn].std(1).mean() for g in gains]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(gains, sig, color=ORANGE, lw=2, marker="o", ms=5)
    ax.set_xlabel("GAIN (what every neuron's inputs add up to)")
    ax.set_ylabel("board signal at the\ndescending neurons")
    ax.set_title("Turning the volume up")
    fig.savefig(OUT / "gain.png")


def hops_figure(c):
    A = sp.csr_matrix((np.ones(len(c.pre)), (c.pre, c.post)), shape=(c.n, c.n))

    def bfs(start, M, k=12):
        d = np.full(c.n, 99)
        d[start] = 0
        front = np.zeros(c.n, bool)
        front[start] = True
        for i in range(1, k + 1):
            nxt = ((M.T @ front.astype(np.float32)) > 0) & (d == 99)
            d[nxt] = i
            front = nxt
        return d
    eye, dn = bfs(c.where("photoreceptors"), A), bfs(motor_output(c), A.T.tocsr())
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for d, label, color, off in ((eye, "synapses from the eye", BLUE, -0.2), (dn, "synapses to the descending neurons", ORANGE, 0.2)):
        counts = np.bincount(np.minimum(d, 9), minlength=10)[:9]
        ax.bar(np.arange(9) + off, counts, width=0.38, color=color, label=label)
    ax.set_xlabel("hops")
    ax.set_ylabel("neurons")
    ax.set_title("A small world: almost every neuron is 2 to 4 hops from both ends")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "hops.png")


def regions_figure(c):
    names = connectome.REGIONS
    counts = [len(c.where(r)) for r in names]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(names, counts, color=BLUE, height=0.65)
    for i, v in enumerate(counts):
        ax.text(v + 600, i, f"{v:,}", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("neurons")
    ax.set_title(f"{c.n:,} neurons, {len(c.pre):,} connections")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.savefig(OUT / "regions.png")


def accuracy_figure():
    drop = np.linspace(0, 40, 200)
    acc = np.clip(103.1668100711649 * np.exp(-0.04354415386753951 * drop) - 3.166924740191411 + 1, 0, 100)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(drop, acc, color=BLUE, lw=2)
    for x, name in ((5, "inaccuracy"), (10, "mistake"), (15, "blunder")):
        ax.axvline(x, color=GRID, lw=1.2, zorder=0)
        ax.text(x + 0.4, 96, name, color=MUTED, fontsize=9)
    ax.set_xlabel("win% the move gave away")
    ax.set_ylabel("move accuracy (%)")
    ax.set_title("How Lichess turns Stockfish's opinion into an accuracy")
    fig.savefig(OUT / "accuracy.png")


def runs(name):
    f = Path("runs") / name / "metrics.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def training_figure():
    dl = runs("dl")
    if not dl:
        return
    blind = next((r for r in dl if "blind_top1" in r), {})
    checks, final = [r for r in dl if "top1" in r][:-1], dl[-1]           # checks use 2,048 positions, the final all 4,096
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axes[0], "top1", "Top-1: picks Stockfish's move"),
                           (axes[1], "good_moves", "Good moves: within 1% win chance of it")):
        ax.plot([r["minutes"] for r in checks], [100 * r[key] for r in checks], color=BLUE, lw=2, marker="o", ms=4,
                label="checks during the run (2,048 positions)")
        end = (final["minutes"], 100 * final[key])
        ax.plot(*end, marker="D", ms=7, color=INK, ls="none", label="final (all 4,096)")
        ax.annotate(f"{math.floor(end[1] * 10 + 0.5) / 10:.1f}%", end, textcoords="offset points", xytext=(-14, -16), color=INK, fontsize=9)
        ax.legend(loc="center right", fontsize=8.5)
        ax.axhline(100 * blind[f"blind_{key}"], color=MUTED, lw=1.2, ls="--")
        ax.text(0.3, 100 * blind[f"blind_{key}"] + 0.3, "never looks at the board", color=MUTED, fontsize=9)
        ax.set_xlabel("minutes of training on my laptop (0 = after the readout probe)")
        ax.set_ylabel("% of held-out positions")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "training.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    pts = [(r["minutes"], r["policy_loss"]) for r in dl if "policy_loss" in r]
    ax.plot(*zip(*pts), color=BLUE, lw=1, alpha=0.5, label="training batches")
    ax.plot([r["minutes"] for r in checks], [r["held_out_loss"] for r in checks], color=BLUE, lw=2, marker="o", ms=4,
            label="held-out positions (2,048)")
    ax.plot(final["minutes"], final["held_out_loss"], marker="D", ms=7, color=INK, ls="none", label="final (all 4,096)")
    ax.set_xlabel("minutes")
    ax.set_ylabel("policy loss")
    ax.set_title("How far the fly's probabilities are from Stockfish's")
    ax.legend()
    fig.savefig(OUT / "loss.png")


def controls_figure():
    """The science controls: the same readout probe on the real brain, a shuffled one and lesioned ones."""
    names = [("probe-real", "real wiring"), ("probe-shuffled", "shuffled wiring"),
             ("probe-no-optic-lobe", "optic lobe cut out"), ("probe-no-central-brain", "central brain cut out")]
    found = [(label, runs(n)[-1]) for n, label in names if runs(n)]
    if not found:
        return
    blind = runs(names[0][0])[0] if runs(names[0][0]) else {}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(found))
    ax.barh(y, [100 * r["top1"] for _, r in found], height=0.6, color=[BLUE] + [ORANGE] * (len(found) - 1))
    for i, (_, r) in enumerate(found):
        ax.text(100 * r["top1"] + 0.2, i, f"{100 * r['top1']:.1f}%", va="center", color=INK, fontsize=9)
    if "blind_top1" in blind:
        ax.axvline(100 * blind["blind_top1"], color=MUTED, lw=1.2, ls="--")
        ax.text(100 * blind["blind_top1"] + 0.2, -0.5, "never looks at the board", color=MUTED, fontsize=9, va="bottom")
    ax.set_yticks(y, [label for label, _ in found])
    ax.invert_yaxis()
    ax.set_xlabel("top-1 on 4,096 held-out positions (readout probe only, same recipe)")
    ax.set_title("Does the real wiring matter?")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "controls.png")


def bio_rules_figure():
    """The dopamine rules compared on the same held-out pairs (tools/bio_rules.py writes the data)."""
    f = Path("runs/bio-rules/results.json")
    if not f.exists():
        return
    data = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for v, color in zip(data["rules"], (MUTED, ORANGE, YELLOW, AQUA, BLUE)):
        x, y = zip(*[(k * 64 / 1000, 100 * a) for k, a in v["curve"]])
        ax.plot(x, y, color=color, lw=1.6, marker="o", ms=3, label=v["name"])
    edge = ax.get_yaxis_transform()                                    # x in axes units, y in data units
    ax.axhline(50, color=MUTED, lw=1.2, ls="--")
    ax.text(0.99, 49.6, "coin flip", color=MUTED, fontsize=9, ha="right", va="top", transform=edge)
    kc, dn = (100 * data[k]["held_out"] for k in ("ceiling_kenyon_cells", "ceiling_descending_neurons"))
    lines = [(kc, "best linear readout of the Kenyon cells"), (dn, "best linear readout of the descending neurons")]
    if abs(kc - dn) < 0.5:
        lines = [(kc, f"best linear readout of the Kenyon cells, or of the descending neurons: {kc:.1f}%")]
    for y, label in lines:
        ax.axhline(y, color=INK, lw=1, ls=":")
        ax.text(0.99, y + 0.4, label, color=INK, fontsize=8.5, ha="right", transform=edge)
    ax.set_xlabel("lessons (thousands)")
    ax.set_ylabel("% of held-out pairs right")
    ax.set_title("Which dopamine rule teaches the fly?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5)
    fig.savefig(OUT / "bio_rules.png")


def bio_figure():
    bio = runs("bio")
    if not bio:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    pts = [(r["lessons"] / 1000, 100 * r["val_pair_acc"]) for r in bio if "val_pair_acc" in r]
    ax.plot(*zip(*pts), color=BLUE, lw=2, marker="o", ms=3)
    ax.annotate(f"{pts[-1][1]:.1f}%", pts[-1], textcoords="offset points", xytext=(-10, 8), color=INK, fontsize=9)
    edge = ax.get_yaxis_transform()
    ax.axhline(50, color=MUTED, lw=1.2, ls="--")
    ax.text(0.99, 49.6, "coin flip", color=MUTED, fontsize=9, ha="right", va="top", transform=edge)
    rules = Path("runs/bio-rules/results.json")
    if rules.exists():                                                 # the best linear readout, for scale
        y = 100 * json.loads(rules.read_text())["ceiling_kenyon_cells"]["held_out"]
        ax.axhline(y, color=INK, lw=1, ls=":")
        ax.text(0.99, y + 0.4, "best linear readout of the Kenyon cells", color=INK, fontsize=8.5, ha="right", transform=edge)
    ax.set_ylim(44, 63)
    ax.set_xlabel("lessons (thousands)")
    ax.set_ylabel("% of 1,024 held-out pairs right")
    ax.set_title("The dopamine fly: Stockfish's move or a random one?")
    fig.savefig(OUT / "bio.png")


PIECE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d55181", "#7d5cc4"]   # pawn, knight, bishop, rook, queen, king
GLYPH = dict(zip("KQRBNP", "♚♛♜♝♞♟"))                                  # solid glyphs, filled white or dark below


def draw_board(ax, board, arrows=(), shade=None, size: float = 25, color=ORANGE) -> None:
    """A board like the app's: green squares, solid pieces. arrows: [(move, strength 0..1)] drawn as thick as their
    strength; shade: 64 numbers 0..1 to tint the squares (for a heat map)."""
    import chess
    import matplotlib.patheffects as pe
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ax.add_patch(plt.Rectangle((f, r), 1, 1, color="#ebecd0" if (f + r) % 2 else "#739552", lw=0))
        if shade is not None and shade[sq] > 0:
            ax.add_patch(plt.Rectangle((f, r), 1, 1, color=ORANGE, alpha=0.85 * min(1.0, shade[sq]), lw=0))
        p = board.piece_at(sq)
        if p:
            ax.text(f + 0.5, r + 0.45, GLYPH[p.symbol().upper()], ha="center", va="center", fontsize=size,
                    color="white" if p.color else "#2b2b2b", family="DejaVu Sans",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#2b2b2b")])
    for f in range(8):
        ax.text(f + 0.94, 0.06, "abcdefgh"[f], ha="right", va="bottom", fontsize=7.5, color="#739552" if f % 2 == 0 else "#ebecd0")
    for m, p in arrows:
        a, b = m.from_square, m.to_square
        x0, y0 = chess.square_file(a) + 0.5, chess.square_rank(a) + 0.5
        x1, y1 = chess.square_file(b) + 0.5, chess.square_rank(b) + 0.5
        w = 0.05 + 0.55 * p
        ax.arrow(x0, y0, (x1 - x0) * 0.82, (y1 - y0) * 0.82, width=w, head_width=w * 2.2 + 0.08, head_length=0.28,
                 length_includes_head=False, color=color, alpha=0.35 + 0.6 * min(1.0, p * 3), lw=0, zorder=4)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")


def eye_cells(c, layout):
    """Every placed photoreceptor: index, square, role (0-5 piece type, 6 own, 7 opponent), and where it
    sits on the round eye (hex1 - hex2, hex1 + hex2), spread a little around its column so each one shows."""
    n_idx, f_idx = eye_input(c, layout)
    cells = np.unique(n_idx)
    first = np.searchsorted(n_idx, cells)                              # eye_input lists each cell's pairs together
    square = f_idx[first] % 64
    role = np.where(np.char.startswith(c.type[cells].astype(str), "R1-R6"), f_idx[first] // 64,
                    np.where(np.char.startswith(c.type[cells].astype(str), "R7"), 6, 7))
    cx, cy = eye_xy(c.hex1[cells], c.hex2[cells])
    rng = np.random.default_rng(0)
    angle, radius = rng.uniform(0, 2 * np.pi, len(cells)), 0.3 * np.sqrt(rng.uniform(0, 1, len(cells)))
    return cells, square, role, cx + radius * np.cos(angle), cy + radius * np.sin(angle), n_idx, f_idx


def eye_xy(hex1, hex2):
    """Column coordinates -> a flat picture of the eye. The two hex axes are 120 degrees apart."""
    return np.sqrt(3) / 2 * (hex1 - hex2), (hex1 + hex2) / 2


def eye_coverage_figure(c=None):
    """Photoreceptors per square with the first eye and with the fair one, and which squares can't tell
    the piece types apart."""
    import matplotlib.patheffects as pe
    c = c or connectome.load()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for ax, layout, title in ((axes[0], "ranks", "The first eye: ranks of hex1 and hex2"), (axes[1], "fair", "The fair eye")):
        _, square, role, *_ = eye_cells(c, layout)
        cells = np.bincount(square, minlength=64)
        types = np.array([len(set(role[(square == s) & (role < 6)])) for s in range(64)])
        ax.imshow(cells.reshape(8, 8), origin="lower", cmap="Blues", vmin=0, vmax=120)
        for s in range(64):
            f, r = s % 8, s // 8
            if cells[s] == 0:
                ax.add_patch(plt.Rectangle((f - 0.5, r - 0.5), 1, 1, color="#fbe3d9"))
                ax.text(f, r, "blind", ha="center", va="center", color=ORANGE, fontsize=8.5, weight="bold")
                continue
            ax.text(f, r + 0.13, str(cells[s]), ha="center", va="center", fontsize=9, color=INK,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            ax.text(f, r - 0.24, f"{types[s]}/6", ha="center", va="center", fontsize=7.5,
                    color=INK if types[s] == 6 else ORANGE, weight="normal" if types[s] == 6 else "bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.set_xticks(range(8), list("abcdefgh"))
        ax.set_yticks(range(8), range(1, 9))
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel(f"{(cells == 0).sum()} squares blind, {(types < 6).sum()} can't tell every piece type apart")
    fig.text(0.5, 0.005, "Each square: how many photoreceptors see it, and how many of the 6 piece types it can tell apart",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "eye_coverage.png")


def eye_figure(c=None):
    """How the fly sees a board with the fair eye: every square is one patch of the eye, inside a patch
    the cells share out the piece types, and a position is a pattern of lit cells."""
    import chess
    from flychess.encode import planes
    import matplotlib.patheffects as pe
    from matplotlib.patches import RegularPolygon
    c = c or connectome.load()
    cells, square, role, x, y, n_idx, f_idx = eye_cells(c, "fair")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 0.85, 1]})
    light, dark = "#d8d3a8", "#739552"

    ax = axes[0]
    shade = np.where((square // 8 + square % 8) % 2, light, dark)
    ax.scatter(x, y, s=6, c=shade, lw=0)
    for s in range(64):
        m = square == s
        ax.text(np.mean(x[m]), np.mean(y[m]), "abcdefgh"[s % 8] + str(s // 8 + 1), ha="center", va="center", fontsize=6.5,
                color=INK, path_effects=[pe.withStroke(linewidth=1.8, foreground=SURFACE)])
    ax.set_title("Every square is a patch of the eye")

    ax = axes[1]
    m = square == chess.E4
    for hx, hy in {(a, b) for a, b in zip(c.hex1[cells[m]], c.hex2[cells[m]])}:   # one hexagon per column (ommatidium)
        ax.add_patch(RegularPolygon(eye_xy(hx, hy), 6, radius=0.56, orientation=np.pi / 6, facecolor="#f1f0ea",
                                    edgecolor=GRID, lw=1))
    names = ["pawn", "knight", "bishop", "rook", "queen", "king", "own piece (R7)", "opponent's piece (R8)"]
    colors = PIECE_COLORS + [INK, MUTED]
    for k in range(8):
        sel = m & (role == k)
        ax.scatter(x[sel], y[sel], s=70 if k < 6 else 40, c=colors[k], marker="o" if k < 6 else ("s" if k == 6 else "^"),
                   label=f"{names[k]}: {sel.sum()}", edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_title("Inside one patch: e4")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    ax.set_aspect("equal")
    ax.autoscale_view()

    ax = axes[2]
    board = chess.Board()
    board.push_uci("e2e4")
    lit_features = np.where(planes(board).ravel() > 0)[0]              # the fly is black here: planes() mirrors the board
    lit = np.isin(f_idx, lit_features)
    lit_cells = np.unique(n_idx[lit])
    own_cells = np.unique(n_idx[lit & (f_idx // 64 < 6)])
    on = np.isin(cells, lit_cells)
    ax.scatter(x[~on], y[~on], s=4, c=GRID, lw=0)
    mine = on & np.isin(cells, own_cells)
    ax.scatter(x[mine], y[mine], s=7, c=BLUE, lw=0, label=f"{mine.sum()} cells see its own pieces")
    ax.scatter(x[on & ~mine], y[on & ~mine], s=7, c=ORANGE, lw=0, label=f"{(on & ~mine).sum()} see the opponent's")
    ax.set_title("1.e4, as the fly sees it (it plays black)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_aspect("equal")
    axes[2].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "eye.png")


def packed_board(bits):
    """A packed training position back as a board, from the mover's side (the mover plays white here).
    Castling rights and en passant aren't in the planes, so moves like those can't be named."""
    import chess
    from flychess.encode import PIECES
    planes = np.unpackbits(bits).reshape(12, 64)
    board = chess.Board(None)
    for k, sq in zip(*np.nonzero(planes)):
        board.set_piece_at(int(sq), chess.Piece(PIECES[k % 6], chess.WHITE if k < 6 else chess.BLACK))
    board.turn = chess.WHITE
    return board


def slot_move(code: int):
    import chess
    from flychess.encode import PROMO
    return chess.Move(int(code) // 64 % 64, int(code) % 64, promotion=PROMO[int(code) // 4096])


def named(board, move) -> str:
    return board.san(move) if move in board.legal_moves else move.uci()


def dl_fly(checkpoint: str = "checkpoints/dl"):
    from flychess.play import DLFly
    return DLFly(checkpoint)


def top1_figure():
    """One held-out position: the fly's favourite, Stockfish's favourite, and the moves within 1 win% of it.
    Picks the first position where the fly's favourite is a good move but not Stockfish's exact one."""
    import tensorflow as tf
    from flychess.train_dl import move_logits
    d = np.load("data/positions/packed.npz")
    n = 400                                                            # the first held-out positions
    bits, legal, ci, cw, best = (d[k][:n] for k in ("boards", "legal", "cand_idx", "cand_win", "best"))
    fly = dl_fly()
    x = np.unpackbits(bits, axis=1).astype(np.float32)
    probs = np.concatenate([tf.nn.softmax(move_logits(fly.model(x[i:i + 50]), legal[i:i + 50])).numpy() for i in range(0, n, 50)])
    good, nice = [], []
    for i in range(n):
        win = cw[i].astype(np.float32)
        fav = int(probs[i].argmax())
        fav_win = win[ci[i] == fav].max() if (ci[i] == fav).any() else -1
        if fav != best[i] and fav_win >= win.max() - 1:                # a good move, not the top-1 one
            good.append(i)
            if 3 <= (win >= win.max() - 1).sum() <= 5 and len(packed_board(bits[i]).piece_map()) >= 18:
                nice.append(i)
    i = (nice or good)[0]
    fav = int(probs[i].argmax())
    board = packed_board(bits[i])
    order = [j for j in np.argsort(-cw[i].astype(np.float32)) if cw[i][j] >= 0][:7]
    moves = [slot_move(legal[i][ci[i][j]]) for j in order]
    wins = [float(cw[i][j]) for j in order]
    fly_p = [float(probs[i][ci[i][j]]) for j in order]
    fav_move = slot_move(legal[i][fav])
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes((0.01, 0.05, 0.36, 0.84))
    draw_board(ax, board, [(fav_move, 0.5)], size=21, color=BLUE)
    draw_board(ax, board, [(moves[0], 0.5)], size=21, color=AQUA)
    ax.set_title("The fly's favourite (blue), Stockfish's (green)", fontsize=10.5)
    labels = [named(board, m) for m in moves]
    for k, (pos, values, title, color, fmt) in enumerate(((0.45, wins, "Stockfish: win% after the move", AQUA, "{:.1f}%"),
                                                         (0.77, [100 * p for p in fly_p], "The fly: its probability", BLUE, "{:.0f}%"))):
        ax = fig.add_axes((pos, 0.12, 0.2, 0.74))
        y = np.arange(len(labels))
        ax.barh(y, values, color=color, height=0.6)
        if k == 0:
            ax.axvspan(wins[0] - 1, wins[0], color=GRID, zorder=0)
            ax.text(wins[0] - 0.5, len(labels) - 0.7, "good moves: within 1 win%", ha="center", va="bottom", fontsize=8,
                    color=MUTED, rotation=90)
            ax.set_xlim(min(wins) - 2, wins[0] + 0.6)
            ax.set_yticks(y, labels, fontsize=10)
        else:
            ax.set_yticks(y, [""] * len(y))
            ax.set_xlim(0, max(values) * 1.35)
        for yy, v in zip(y, values):
            ax.text(v + (0.05 if k == 0 else 0.6), yy, fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.text(0.47, 0.01, f"The fly's favourite {named(board, fav_move)} isn't Stockfish's {labels[0]}, so it doesn't count for top-1, "
             f"but it's within 1 win%, so it counts as a good move.", fontsize=9, color=INK)
    fig.savefig(OUT / "top1.png")
    print(f"top1 figure: held-out position {i}, fly {named(board, fav_move)}, Stockfish {labels[0]}")


def occlusion_figure():
    """What the fly looked at: take one piece off the board at a time and see how much its chosen move loses.
    One batch of boards, a lesion experiment on the board instead of the brain."""
    import chess
    from flychess.encode import planes, policy
    fly = dl_fly()
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 1 5")   # Italian Game, 4...Nf6
    fl, tl, pl, _ = (h.numpy() for h in fly.model(planes(board).reshape(1, 768)))
    ranked = policy(board, fl[0], tl[0], pl[0])
    chosen, p0 = ranked[0]
    squares = [sq for sq, p in board.piece_map().items() if p.piece_type != chess.KING and sq != chosen.from_square]
    variants = []
    for sq in squares:
        b = board.copy()
        b.remove_piece_at(sq)
        variants.append(b)
    x = np.stack([planes(b).ravel() for b in variants])
    outs = [h.numpy() for h in fly.model(x)]
    drop = np.zeros(64)
    for k, (sq, b) in enumerate(zip(squares, variants)):
        p = dict(policy(b, outs[0][k], outs[1][k], outs[2][k]))
        drop[sq] = p0 - p.get(chosen, 0.0)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    draw_board(ax, board, [(chosen, 0.6)], shade=np.maximum(drop, 0) / max(drop.max(), 1e-9), size=24, color=BLUE)
    for sq in squares:
        if abs(drop[sq]) >= 0.005:
            ax.text(chess.square_file(sq) + 0.96, chess.square_rank(sq) + 0.96, f"{100 * drop[sq]:+.0f}", ha="right", va="top",
                    fontsize=8, color=INK, weight="bold")
    ax.set_title(f"Without each piece, how much {board.san(chosen)} ({100 * p0:.0f}%) loses", fontsize=11)
    fig.savefig(OUT / "occlusion.png")
    top = sorted(((drop[sq], chess.square_name(sq)) for sq in squares), reverse=True)[:5]
    print(f"occlusion: {board.san(chosen)} at {p0:.1%}; biggest drops {[(s, round(100 * v, 1)) for v, s in top]}")


def kc_mbon_figure():
    """The dopamine fly after training: how much each MBON type's Kenyon cell input changed, and which dopamine
    reaches it (reward PAM, punishment PPL1)."""
    from flychess.bio import BioFly
    c = connectome.load()
    fly = BioFly(c)
    w = np.load("checkpoints/bio/kc_mbon.npz")["w"]
    change = (w.sum(0) - fly.w0.sum(0)) / np.maximum(fly.w0.sum(0), 1e-9)
    names = np.array([str(t).split("_")[0] for t in c.type[fly.mbon]])
    rows = []
    for t in np.unique(names):
        m = names == t
        if fly.w0[:, m].sum() == 0:
            continue
        rows.append((float((w[:, m].sum() - fly.w0[:, m].sum()) / fly.w0[:, m].sum()), t,
                     float(fly.pam_reach[m].mean()), float(fly.ppl1_reach[m].mean())))
    rows = sorted(rows)[:22]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(rows) + 1.3))
    for i, (ch, t, pam, ppl1) in enumerate(rows):
        color = BLUE if pam > ppl1 else ORANGE if ppl1 > pam else MUTED
        ax.barh(i, 100 * ch, color=color, height=0.65)
    ax.set_yticks(range(len(rows)), [t for _, t, _, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("change in Kenyon cell input after 15 minutes of lessons (%)")
    ax.set_title("Where dopamine changed the dopamine fly")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="reached by reward dopamine (PAM): an avoid MBON"),
                       Patch(color=ORANGE, label="reached by punishment dopamine (PPL1): an approach MBON")],
              fontsize=8, loc="lower left")
    fig.savefig(OUT / "kc_mbon.png")
    print(f"kc_mbon: {len(rows)} types shown, biggest change {rows[0][1]} {100 * rows[0][0]:.1f}%; all MBONs: "
          f"{100 * change.min():.1f}% to {100 * change.max():.1f}%")


def lesson_figure():
    """A comic of one tutor lesson: sugar after Stockfish's move, a shock after a random one, and which synapses
    change. The Kenyon cell counts are the dopamine fly's own, for these two positions."""
    import chess
    from flychess.encode import planes
    from flychess.play import BioPlayer
    player = BioPlayer("checkpoints/bio/kc_mbon.npz")
    fly = player.fly
    start = chess.Board()
    start.push_uci("e2e4")
    after = {}
    for uci in ("c7c5", "h7h5"):
        b = start.copy()
        b.push_uci(uci)
        after[uci] = b
    x = np.stack([planes(after[u], chess.BLACK).ravel() for u in after])
    kc = fly.sparse_kc(fly.run(x)).numpy()                             # (Kenyon cells, 2)
    on = kc > 0.5 * kc.max()
    both = int((on[:, 0] & on[:, 1]).sum())
    only = [int((on[:, k] & ~on[:, 1 - k]).sum()) for k in (0, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, (uci, b), text, color in zip(axes[:2], after.items(), ("sugar: reward dopamine (PAM)", "a shock: punishment dopamine (PPL1)"),
                                         (AQUA, ORANGE)):
        draw_board(ax, b, size=17)
        k = 0 if uci == "c7c5" else 1
        who = "Stockfish's move" if k == 0 else "a random move"
        ax.set_title(f"{k + 1}. After 1.e4 {start.san(chess.Move.from_uci(uci))}, {who}", fontsize=10.5)
        ax.text(4, -0.55, f"{int(on[:, k].sum())} Kenyon cells fire\n{text}", ha="center", va="top", fontsize=9.5, color=color,
                weight="bold")
        ax.set_ylim(-1.6, 8)
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("3. The synapses of the cells that fired get weaker", fontsize=10.5, loc="left")
    groups = ((8.0, f"{only[0]} only for c5", AQUA, 0.45), (5.0, f"{both} for both", MUTED, 1.35), (2.0, f"{only[1]} only for h5", ORANGE, 0.45))
    for y, label, color, r in groups:                                 # the Kenyon cells, as three groups
        ax.add_patch(plt.Circle((1.6, y), r, color=color, alpha=0.85))
        ax.text(1.6 + r + 0.25, y, label, va="center", fontsize=9, color=INK)
    boxes = ((8.3, "avoid MBONs", BLUE, "sugar: PAM"), (1.7, "approach MBONs", ORANGE, "shock: PPL1"))
    for y, label, color, drug in boxes:
        ax.add_patch(plt.Rectangle((6.6, y - 0.7), 3.2, 1.4, facecolor=color, alpha=0.16, edgecolor=color))
        ax.text(8.2, y + 0.18, label, ha="center", va="center", fontsize=9.5, color=INK, weight="bold")
        ax.text(8.2, y - 0.35, drug, ha="center", va="center", fontsize=8.5, color=color)
    for y0, y1, color in ((8.0, 8.3, AQUA), (5.0, 8.3, AQUA), (5.0, 1.7, ORANGE), (2.0, 1.7, ORANGE)):
        ax.annotate("", (6.6, y1), (4.9, y0), arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2, "ls": "--"})
    ax.text(5.0, -0.2, f"{both} of the cells fire for both positions, so most\nof the sugar and the shock land on the same synapses",
            ha="center", va="top", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "lesson.png")
    print(f"lesson: {int(on[:, 0].sum())} and {int(on[:, 1].sum())} Kenyon cells, {both} in both")


def curve_figure():
    """The readout's learning curve on cached brain activity: held-out loss after fitting to the end on 25,000 to
    500,000 positions (tools/readout_lab.py curve). Is more data still worth a day?"""
    f = Path("runs/readout-lab/fair.json")
    if not f.exists() or "curve" not in json.loads(f.read_text()):
        return
    curve = json.loads(f.read_text())["curve"]
    n = [c["train_positions"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, key, label in ((axes[0], "held_out_loss", "held-out loss (lower is better)"), (axes[1], "top1", "top-1 (%)")):
        y = [c[key] * (100 if key == "top1" else 1) for c in curve]
        ax.plot(n, y, color=BLUE, lw=2, marker="o", ms=5)
        for x, v in zip(n, y):
            ax.annotate(f"{v:.3f}" if key == "held_out_loss" else f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(n, [f"{x // 1000}k" if x < 1_000_000 else f"{x / 1e6:g}M" for x in n])
        ax.set_xlabel("different positions the readout learned from")
        ax.set_ylabel(label)
    axes[0].set_title("More positions, fitted to the end")
    axes[1].set_title("Top-1 on the same 4,096 held-out positions")
    fig.tight_layout()
    fig.savefig(OUT / "curve.png")


def plasticity_figure(run: str = "pilot-synapses"):
    """For a run that trained the brain: how far every synapse moved from its measured strength, and which
    regions' neurons changed their inputs most."""
    import tensorflow as tf
    from flychess.model import FlyChess
    ckpt = Path("checkpoints") / run
    if not (ckpt / "checkpoint").exists():
        return
    cfg = json.loads((Path("runs") / run / "config.json").read_text())
    c = connectome.load()
    model = FlyChess(c, eye=cfg.get("eye", "ranks"))
    model(np.zeros((1, 768), np.float32))
    w0 = tf.nn.softplus(model.brain.theta).numpy()                     # as measured, before the checkpoint
    tf.train.Checkpoint(model=model).restore(tf.train.latest_checkpoint(str(ckpt))).expect_partial()
    w1 = tf.nn.softplus(model.brain.theta).numpy()
    change = (w1 - w0) / w0
    order = np.lexsort((c.pre, c.post))                                # the brain's edge order
    post_region = c.region[c.post[order]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.hist(100 * change, bins=np.linspace(-25, 25, 101), color=BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("change from the measured strength (%)")
    ax.set_ylabel("synapses")
    ax.set_title(f"{len(change):,} synapses after {cfg['minutes']:g} minutes")
    ax = axes[1]
    regions = [r for r in SHOWN if (post_region == r).any()]
    moved = [100 * np.abs(change[post_region == r]).mean() for r in regions]
    ax.barh(regions, moved, color=BLUE, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("average change of the synapses onto it (%)")
    ax.set_title("Which regions' inputs moved")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / f"plasticity_{run}.png")
    print(f"plasticity {run}: median |change| {100 * np.median(np.abs(change)):.3f}%, "
          f"{(np.abs(change) > 0.01).mean():.1%} moved more than 1%, at the leash: {(np.abs(change) > 0.199).mean():.2%}")


def steps_figure():
    """Why 12 steps: the board's signal per region for 20 steps (homeostasis calibrated for all 20), and how well a
    readout does when it reads the descending neurons at step 12, 16 or 20 (tools/readout_lab.py beyond)."""
    f = Path("runs/readout-lab/fair.json")
    if not f.exists() or "beyond_spread" not in json.loads(f.read_text()):
        return
    data = json.loads(f.read_text())
    spread, readout = data["beyond_spread"], data.get("beyond_readout", {})
    steps = np.arange(1, len(spread) + 1)
    if readout:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    else:
        fig, ax0 = plt.subplots(figsize=(7.5, 3.9))
        axes = [ax0, None]
    ax = axes[0]
    for r, color in (("optic lobe", YELLOW), ("visual projection", AQUA), ("kenyon cells", ORANGE), ("central brain", MUTED),
                     ("descending", BLUE)):
        ax.plot(steps, [s[r] for s in spread], color=color, lw=2.4 if r == "descending" else 1.6, marker="o", ms=3, label=r)
    ax.axvline(12, color=INK, lw=1, ls=":")
    ax.text(12.2, ax.get_ylim()[1] * 0.97, "the flies read here", fontsize=8.5, color=INK, va="top")
    ax.set_xticks(steps)
    ax.set_xlabel("step")
    ax.set_ylabel("how much the board changes the firing\n(spread across 256 positions)")
    ax.set_title("The board's signal for 20 steps")
    ax.legend(fontsize=8.5, loc="lower right")
    if readout:
        ax = axes[1]
        names = list(readout)
        loss = [readout[k]["held_out_loss"] for k in names]
        ax.bar(range(len(names)), loss, color=[BLUE if k == "step 12" else MUTED for k in names], width=0.6)
        for i, (k, v) in enumerate(zip(names, loss)):
            ax.text(i, v + 0.002, f"{v:.3f}\n{100 * readout[k]['top1']:.1f}% top-1", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(names)), [f"read at {k}" for k in names])
        ax.set_ylim(min(loss) - 0.03, max(loss) + 0.03)
        ax.set_ylabel("held-out loss (lower is better)")
        ax.set_title("A readout fitted at each step")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "steps.png")


def blame_figure():
    """Backprop through time on one position (tools/bptt.py): how big the blame is at every step on the way
    back, next to how many neurons homeostasis gives its biggest gain at that step."""
    f = Path("runs/bptt/e4.npz")
    if not f.exists():
        return
    d = np.load(f)
    blame, floor = json.loads(str(d["per_region"])), json.loads(str(d["at_floor"]))
    steps = np.arange(1, 13)
    shown = (("descending", BLUE), ("central brain", ORANGE), ("nerve cord", AQUA), ("optic lobe", YELLOW), ("photoreceptors", MUTED))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for r, color in shown:
        pts = [(s, blame[s - 1][r]) for s in steps if blame[s - 1][r] > 0]
        ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=3.5, label=r)
    ax.set_yscale("log")
    ax.set_ylabel("blame: average |d loss / d input|")
    ax.set_title("The error on its way back from step 12")
    ax.legend(fontsize=8.5, loc="upper right")
    ax = axes[1]
    ax.plot(steps, [100 * f["all"] for f in floor], color=INK, lw=2.4, marker="o", ms=3.5, label="all neurons")
    for r, color in shown[1:4]:
        ax.plot(steps, [100 * f[r] for f in floor], color=color, lw=1.6, marker="o", ms=3, label=r)
    ax.set_ylabel("% of neurons at the floor")
    ax.set_title("Neurons the board hasn't reached yet")
    ax.legend(fontsize=8.5, loc="center left")
    for ax in axes:
        ax.set_xlim(12.5, 0.5)                                         # going back in time, left to right
        ax.set_xticks(steps)
        ax.axvspan(6.5, 12.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.set_xlabel("step (the backward pass goes from 12 to 1)")
    axes[0].text(9.5, axes[0].get_ylim()[1] / 20, "what training looks at\n(--window 6)", color=MUTED, fontsize=8.5,
                 ha="center", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "blame.png")


RATED = {"dl": "Backprop fly", "day": "Day-long fly", "day-imagine": "Day-long fly + imagination", "bio": "Dopamine fly", "quick": "3-minute fly", "shuffled": "Shuffled fly",
         "no-optic-lobe": "Fly without optic lobes", "no-central-brain": "Fly without a central brain", "newborn": "Newborn fly",
         "stockfish-1320": "Stockfish 1320", "stockfish-1320-25": "Stockfish 1320, 25% random moves",
         "stockfish-1320-50": "Stockfish 1320, 50% random moves", "stockfish-1320-75": "Stockfish 1320, 75% random moves",
         "greedy": "Greedy (takes the biggest piece)", "random": "Random mover"}


def progress_figure():
    """The day-long fly at every fit of the long run (the "final" lines of runs/dl-day/metrics.jsonl), after the
    25-minute backprop fly: top-1, good moves and the held-out loss, all on the same 4,096 held-out positions."""
    f = Path("runs/dl-day/metrics.jsonl")
    if not f.exists():
        return
    fits = [json.loads(line) for line in f.read_text().splitlines() if '"final"' in line]
    n = [527_000] + [r["positions"] for r in fits]                     # the 25-minute fly streamed 527,000 positions
    top1 = [19.9] + [100 * r["top1"] for r in fits]
    good = [29.2] + [100 * r["good_moves"] for r in fits]
    loss = [2.963] + [r["held_out_loss"] for r in fits]
    stage = ["25-minute fly", "new recipe", "+7 files", "cycle 1, old fit", "cycle 2, fixed fit"][:len(n)]
    x = list(range(len(n)))                                            # one step per fit, like a timeline
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(x, good, color=MUTED, lw=2, marker="o", ms=5, label="good move (within 1 win% of Stockfish's)")
    ax.plot(x, top1, color=BLUE, lw=2, marker="o", ms=5, label="top-1 (Stockfish's own move)")
    for x_, a, b in zip(x, top1, good):
        ax.annotate(f"{a:.1f}", (x_, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=BLUE)
        ax.annotate(f"{b:.1f}", (x_, b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylabel("% of held-out positions")
    ax.set_ylim(16, 37)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("Picking Stockfish's move")
    ax = axes[1]
    ax.plot(x, loss, color=BLUE, lw=2, marker="o", ms=5)
    for x_, v in zip(x, loss):
        ax.annotate(f"{v:.3f}", (x_, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=INK)
    if len(n) >= 4:
        ax.annotate("stalled: the fit,\nnot the data", (x[3], loss[3]), textcoords="offset points", xytext=(-10, 30),
                    ha="center", fontsize=8.5, color=ORANGE, arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set_ylabel("held-out loss (lower is better)")
    ax.set_title("The loss on the same positions")
    for ax in axes:
        ax.set_xticks(x, [f"{s}\n{p / 1e6:.1f}M" for s, p in zip(stage, n)], fontsize=8)
        ax.set_xlim(-0.4, len(n) - 0.6)
        ax.set_xlabel("each fit, and the positions it learned from")
    fig.tight_layout()
    fig.savefig(OUT / "progress.png")


def ratings_figure():
    """Every fly's Elo with its 95% interval, next to the ladder it was measured on (flychess/rating.py)."""
    f = Path("app/static/ratings.json")
    if not f.exists():
        return
    rows = sorted(json.loads(f.read_text())["ratings"].items(), key=lambda kv: kv[1]["elo"])
    is_fly = lambda p: not p.startswith(("stockfish", "random", "greedy"))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 1.1))
    for i, (p, r) in enumerate(rows):
        color = BLUE if is_fly(p) else MUTED
        ax.plot([r["low"], r["high"]], [i, i], color=color, lw=2.2, alpha=0.4, solid_capstyle="round")
        ax.plot(r["elo"], i, "o", color=color, ms=7, mec=SURFACE, mew=1.5)
        ax.text(r["high"] + 18, i, f"{r['elo']}", va="center", fontsize=9, color=INK if is_fly(p) else MUTED)
    ax.set_yticks(range(len(rows)), [RATED.get(p, p) for p, _ in rows])
    for label, (p, _) in zip(ax.get_yticklabels(), rows):
        label.set_color(INK if is_fly(p) else MUTED)
    ax.set_xlabel("Elo with its 95% interval (Stockfish 1320 held at 1320). Blue: the flies, grey: the ladder")
    ax.set_title("How strong is each fly?")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "ratings.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = [a for a in sys.argv[1:] if not a.startswith("--")]        # e.g. python tools/readme_figures.py blame
    if names:
        for name in names:
            globals()[f"{name}_figure"]()
        print(f"{', '.join(names)} in {OUT}")
        sys.exit()
    ratings_figure()
    accuracy_figure()
    training_figure()
    controls_figure()
    bio_rules_figure()
    bio_figure()
    blame_figure()
    if "--brain" in sys.argv:                                          # the slow ones, about a minute
        c = connectome.load()
        x = boards()
        regions_figure(c)
        hops_figure(c)
        print(json.dumps({k: [round(v, 4) for v in s] for k, s in signal_figure(c, x).items()}))
        gain_figure(c, x)
    print(f"figures in {OUT}")

"""Draw the pictures in the README into docs/img/.

The data figures use the real MaleCNS tables, so run scripts/download_connectome.py
first (only the two small tables are needed). The rest are diagrams.

Usage: pip install matplotlib && python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flychess.connectome import REGIONS, RAW, region_of  # noqa: E402

OUT = Path("docs/img")
INK, MUTED, LINE, ACCENT, BLUE, GREEN, RED = "#1d1d1b", "#6b6a66", "#d8d6cf", "#d97a1e", "#2f7fc1", "#3a9d5d", "#c0443a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": LINE,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, text, color=ACCENT, sub=None, alpha=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=color, alpha=alpha, lw=1.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="none", ec=color, lw=1.5))
    ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), text, ha="center", va="center", fontsize=11, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center", fontsize=7.8, color=MUTED)


def arrow(ax, a, b, color=INK, style="-|>", ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def pipeline():
    """Board -> eye -> brain -> descending neurons -> move."""
    fig, ax = canvas(12, 3.2)
    steps = [("Chess board", "12 planes x 64 squares", BLUE), ("Eye", "photoreceptors R1-R8", ACCENT),
             ("Optic lobe", "edges, shapes", ACCENT), ("Central brain", "mushroom body, CX", ACCENT),
             ("Descending", "~1,300 neurons", ACCENT), ("Move", "from + to + promotion", GREEN)]
    for i, (t, s, c) in enumerate(steps):
        x = 0.1 + i * 2
        box(ax, x, 1.1, 1.8, 1.0, t, c, s)
        if i:
            arrow(ax, (x - 0.2, 1.6), (x - 0.02, 1.6))
    ax.text(6.0, 2.75, "the fly's own wiring, 12 steps of activity", ha="center", color=MUTED, fontsize=10)
    ax.plot([2.2, 9.8], [2.5, 2.5], color=ACCENT, lw=1, alpha=0.6)
    ax.text(1.0, 0.55, "input", ha="center", color=MUTED, fontsize=9)
    ax.text(11.0, 0.55, "readout (learned)", ha="center", color=MUTED, fontsize=9)
    fig.savefig(OUT / "pipeline.png")


def regions(ann):
    """Real neuron counts per region."""
    ann = ann[ann.status.isin(["Traced", "Anchor"]) & ann.superclass.notna()].copy()
    ann["region"] = ann.apply(region_of, axis=1)
    counts = ann.region.value_counts().reindex(REGIONS).fillna(0).astype(int)
    counts = counts[counts > 0].sort_values()
    used = {"photoreceptors", "optic lobe", "kenyon cells", "MBON", "dopamine (PAM)", "dopamine (PPL1)", "descending"}
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=[ACCENT if r in used else LINE for r in counts.index])
    ax.set_xscale("log")
    for b, v in zip(bars, counts.values):
        ax.text(v * 1.08, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("neurons (log scale)")
    ax.set_title(f"MaleCNS v1.0: {len(ann):,} traced neurons by region", loc="left", fontsize=12)
    ax.text(0, -0.16, "orange = regions this project reads from, writes into or trains", transform=ax.transAxes,
            fontsize=9, color=MUTED)
    fig.savefig(OUT / "regions.png")


def transmitters(ann, nt):
    """Real transmitter split, which decides the sign of every connection."""
    ids = ann[ann.status.isin(["Traced", "Anchor"]) & ann.superclass.notna()].bodyId
    c = nt.set_index("body").consensus_nt.reindex(ids).fillna("unknown").str.lower().value_counts()
    sign = {"acetylcholine": GREEN, "gaba": RED, "glutamate": RED, "histamine": RED}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(c.index, c.values, color=[sign.get(k, LINE) for k in c.index])
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_ylabel("neurons")
    ax.set_title("Predicted transmitter per neuron: green excites, red inhibits", loc="left", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(OUT / "transmitters.png")


def eye(ann):
    """Real eye columns (from L1 lamina cells, one per column) cut into 64 chess squares."""
    l1 = ann[(ann.type == "L1") & ann.assignedOlHex1.notna()]
    side = l1.rootSide.fillna(l1.somaSide)
    l1 = l1[side == side.mode()[0]]
    h1, h2 = l1.assignedOlHex1.values, l1.assignedOlHex2.values

    def bins(x):
        return np.minimum((x.argsort().argsort() * 8) // len(x), 7)
    f, r = bins(h1), bins(h2)
    # Hex grid to screen: offset every other column.
    x = h1 + 0.5 * (h2 % 2)
    y = h2 * 0.87
    light = (f + r) % 2 == 1
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x[light], y[light], s=26, c="#efd9b5", edgecolors="#c9a877", lw=0.5)
    ax.scatter(x[~light], y[~light], s=26, c="#b58863", edgecolors="#8a6446", lw=0.5)
    for sq in [(4, 3), (6, 5)]:  # label two squares
        m = (f == sq[0]) & (r == sq[1])
        ax.text(x[m].mean(), y[m].mean(), "abcdefgh"[sq[0]] + str(sq[1] + 1), ha="center", va="center",
                fontsize=11, weight="bold", color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"One eye, {len(l1)} real columns, cut into 64 squares", loc="left", fontsize=12)
    fig.savefig(OUT / "eye-board.png")


def square_cells():
    """Inside one square: which photoreceptor reports what."""
    fig, ax = canvas(10, 3.6)
    names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    ax.text(0.2, 3.2, "Inside the patch of eye for one square (say e4)", fontsize=12, weight="bold")
    for i, n in enumerate(names):
        box(ax, 0.2 + i * 1.3, 1.7, 1.15, 0.9, "R1-R6", ACCENT, n)
    box(ax, 0.2, 0.4, 3.8, 0.9, "R7", GREEN, "lights up for the fly's own pieces")
    box(ax, 4.1, 0.4, 3.7, 0.9, "R8", RED, "lights up for the opponent's pieces")
    ax.text(8.2, 1.9, "Own knight on e4:\n  knight R1-R6 cell\n  + R7 cells", fontsize=10, va="center")
    ax.text(8.2, 0.85, "Enemy knight on e4:\n  knight R1-R6 cell\n  + R8 cells", fontsize=10, va="center")
    fig.savefig(OUT / "square-cells.png")


def activity_steps():
    """Illustrative: when each region wakes up over the 12 steps."""
    steps = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, onset, color in [("photoreceptors", 1, BLUE), ("optic lobe", 3, ACCENT), ("central brain", 6, GREEN),
                               ("mushroom body", 7, "#b04a8f"), ("descending", 9, INK)]:
        y = 1 / (1 + np.exp(-(steps - onset - 1) * 1.3))
        ax.plot(steps, y, lw=2.2, color=color, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("activity (sketch)")
    ax.set_xticks(steps)
    ax.set_title("How the signal travels: each region wakes up a few steps after the one before it", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right")
    ax.text(1, -0.28, "Sketch of the expected pattern. The brain viewer in the interface shows the real one for any position.",
            transform=ax.transData, fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "activity-steps.png")


def backprop():
    """Forward through 12 steps, gradient back through the same 12."""
    fig, ax = canvas(12, 4.2)
    xs = [0.3, 2.1, 3.9, 6.9, 8.7, 10.5]
    labels = [("board", BLUE), ("step 1", ACCENT), ("step 2", ACCENT), ("step 12", ACCENT), ("readout", GREEN), ("loss", RED)]
    for x, (t, c) in zip(xs, labels):
        box(ax, x, 1.7, 1.3, 0.8, t, c)
    ax.text(5.9, 2.1, ". . .", ha="center", va="center", fontsize=16, color=MUTED)
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a < 2.5:
            arrow(ax, (a + 1.35, 2.25), (b - 0.05, 2.25), INK)
            arrow(ax, (b - 0.05, 1.95), (a + 1.35, 1.95), RED, ls="--")
    ax.text(6.0, 3.3, "forward: guess a move", ha="center", fontsize=11)
    ax.text(6.0, 0.9, "backward: every connection learns how much it caused the mistake (chain rule, back through time)",
            ha="center", fontsize=10, color=RED)
    ax.text(6.0, 0.35, "the same weights W are used at every step, so their gradient adds up across all 12",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(OUT / "backprop.png")


def loss_curve():
    """What the loss number means for one guess."""
    p = np.linspace(0.005, 1, 300)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(p, -np.log(p), color=ACCENT, lw=2.2)
    for q in [0.02, 0.3, 0.9]:
        ax.scatter([q], [-np.log(q)], color=INK, zorder=3)
        ax.annotate(f"p = {q}: loss {-np.log(q):.1f}", (q, -np.log(q)), xytext=(12, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("probability the fly gave to Stockfish's move")
    ax.set_ylabel("loss = -log(p)")
    ax.set_title("Cross-entropy: confident and right is cheap, confident and wrong is expensive", loc="left", fontsize=11)
    fig.savefig(OUT / "loss.png")


def mushroom_body():
    """KC -> MBON with dopamine compartments."""
    fig, ax = canvas(12, 5.8)
    ax.text(0.2, 5.45, "Mushroom body: the one place the biological fly learns", fontsize=12, weight="bold")
    box(ax, 0.3, 1.9, 2.2, 1.2, "Position", BLUE, "the move's result")
    for i in range(6):
        on = i in (1, 4)
        ax.add_patch(plt.Circle((4.0, 0.9 + i * 0.55), 0.18, color=ACCENT if on else LINE))
    ax.text(4.0, 4.35, "Kenyon cells\n(a sparse few fire)", ha="center", fontsize=9.5)
    arrow(ax, (2.55, 2.5), (3.7, 2.5))
    box(ax, 7.0, 2.9, 2.0, 0.9, "MBON approach", GREEN)
    box(ax, 7.0, 1.2, 2.0, 0.9, "MBON avoid", RED)
    for i in (1, 4):
        arrow(ax, (4.2, 0.9 + i * 0.55), (6.95, 3.35), ACCENT, lw=1.6)
        arrow(ax, (4.2, 0.9 + i * 0.55), (6.95, 1.65), ACCENT, lw=1.6)
    box(ax, 6.2, 4.2, 1.6, 0.6, "PPL1", RED, None, 0.25)
    box(ax, 6.2, 0.1, 1.6, 0.6, "PAM", GREEN, None, 0.25)
    arrow(ax, (7.0, 4.2), (7.6, 3.85), RED, ls="--")
    arrow(ax, (7.0, 0.7), (7.6, 1.15), GREEN, ls="--")
    ax.text(8.0, 4.5, "bad outcome", ha="left", va="center", fontsize=9, color=RED)
    ax.text(8.0, 0.4, "good outcome", ha="left", va="center", fontsize=9, color=GREEN)
    arrow(ax, (9.05, 3.35), (10.0, 2.55))
    arrow(ax, (9.05, 1.65), (10.0, 2.35))
    box(ax, 10.0, 2.0, 1.8, 0.9, "Choice", INK, "approach - avoid", 0.08)
    ax.text(0.3, 0.9, "Dopamine weakens the synapses\nfrom Kenyon cells that just fired,\nonly in its own compartment.",
            fontsize=9.5, color=MUTED)
    fig.savefig(OUT / "mushroom-body.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = pd.read_feather(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = pd.read_feather(RAW / "body-neurotransmitters-male-cns-v1.0.feather")
    for f in (pipeline, square_cells, activity_steps, backprop, loss_curve, mushroom_body):
        f()
    regions(ann)
    transmitters(ann, nt)
    eye(ann)
    print("figures in", OUT)


if __name__ == "__main__":
    main()

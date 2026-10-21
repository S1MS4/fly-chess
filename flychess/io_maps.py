"""Where the board goes in and where the moves come out.

In: the eye. Each photoreceptor sits in one column of the compound eye. The eye's
columns are split into an 8 x 8 grid and each grid cell is one chess square. Inside
a square:
  - R1-R6 cells (motion and brightness) are shared out between the 6 piece types
  - R7 cells light up for the fly's own pieces
  - R8 cells light up for the opponent's pieces
So a knight on e4 lights up the R1-R6 "knight" cells in the e4 patch of the eye,
plus R7 or R8 depending on whose knight it is.

Two ways to cut the eye into squares:
  - "ranks", the first one: files from the ranks of hex1, ranks from the ranks of hex2. The eye is
    a tilted oval in those coordinates, so two corners of the grid come out empty: 12 squares get
    no photoreceptor at all and 15 can't tell one piece type from another.
  - "fair": the eye is round in (hex1 - hex2, hex1 + hex2), so cut it into 8 bands of equal size
    and every band into 8 squares of equal size. Every square gets 1/64 of the eye, and the six
    piece types are shared out inside each square. The trained flies say which one they use.

Out: the descending neurons, about 1300 cells that carry commands from the brain
down to the body. In a real fly they start walking, turning or take-off. Here a
small readout turns their firing into move scores.
"""
import numpy as np

from .connectome import Connectome


def square_of(hex1: np.ndarray, hex2: np.ndarray) -> np.ndarray:
    """Map eye column coordinates onto 64 squares using rank bins, so every square gets
    roughly the same number of columns even though the eye is not a square."""
    def bins(x):
        ranks = x.argsort().argsort()
        return np.minimum((ranks * 8) // len(x), 7)
    return bins(hex2) * 8 + bins(hex1)  # rank * 8 + file


def fair_squares(hex1: np.ndarray, hex2: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Cut the round eye into 64 patches with the same number of cells (or the same total `weight`):
    8 bands along hex1 + hex2, each band into 8 along hex1 - hex2. A column's cells stay together."""
    cols, inverse = np.unique(np.stack([hex1, hex2], 1), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse, weights=np.ones(len(hex1)) if weight is None else weight)
    u, v = cols[:, 0] - cols[:, 1], cols[:, 0] + cols[:, 1]

    def cut(order, w, parts=8):                                     # equal shares of the weight, in this order
        before = np.cumsum(w[order]) - w[order]                     # weight in front of each column
        out = np.empty(len(order), int)
        out[order] = np.minimum((before * parts / w.sum()).astype(int), parts - 1)
        return out

    rank = cut(np.lexsort((u, v)), counts)
    file = np.zeros(len(cols), int)
    for r in range(8):
        m = np.where(rank == r)[0]
        file[m] = cut(np.lexsort((v[m], u[m])), counts[m])
    return (rank * 8 + file)[inverse]


def eye_input(c: Connectome, layout: str = "ranks") -> tuple[np.ndarray, np.ndarray]:
    """Pairs (neuron index, feature index) where feature = plane * 64 + square.
    Feeding the 768 board features through these pairs gives the drive to each photoreceptor."""
    photo = c.where("photoreceptors")
    photo = photo[~np.isnan(c.hex1[photo])]
    if layout == "fair":                                            # the piece-type cells (R1-R6) set the cut: part of the
        r16 = np.char.startswith(c.type[photo].astype(str), "R1-R6")  # eye has none placed, and every square needs some
        sq = fair_squares(c.hex1[photo], c.hex2[photo], np.where(r16, 1.0, 0.1))
    else:
        sq = square_of(c.hex1[photo], c.hex2[photo])
    seen = np.zeros(64, int)                                        # fair: R1-R6 cells shared out per square
    neurons, features = [], []
    for i, (n, s) in enumerate(zip(photo, sq)):
        t = c.type[n]
        if t.startswith("R1-R6"):
            if layout == "fair":
                k, seen[s] = seen[s] % 6, seen[s] + 1
            else:
                k = i % 6  # this cell reports one piece type, either colour
            neurons += [n, n]
            features += [k * 64 + s, (k + 6) * 64 + s]
        elif t.startswith("R7"):  # own pieces
            neurons += [n] * 6
            features += [k * 64 + s for k in range(6)]
        elif t.startswith("R8"):  # opponent pieces
            neurons += [n] * 6
            features += [(k + 6) * 64 + s for k in range(6)]
    return np.array(neurons, np.int64), np.array(features, np.int64)


def motor_output(c: Connectome) -> np.ndarray:
    return c.where("descending")

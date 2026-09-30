"""Where the board goes in and where the moves come out.

In: the eye. Each photoreceptor sits in one column of the compound eye. The eye's
columns are split into an 8 x 8 grid and each grid cell is one chess square. Inside
a square:
  - R1-R6 cells (motion and brightness) are shared out between the 6 piece types
  - R7 cells light up for the fly's own pieces
  - R8 cells light up for the opponent's pieces
So a knight on e4 lights up the R1-R6 "knight" cells in the e4 patch of the eye,
plus R7 or R8 depending on whose knight it is.

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


def eye_input(c: Connectome) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (neuron index, feature index) where feature = plane * 64 + square.
    Feeding the 768 board features through these pairs gives the drive to each photoreceptor."""
    photo = c.where("photoreceptors")
    photo = photo[~np.isnan(c.hex1[photo])]
    sq = square_of(c.hex1[photo], c.hex2[photo])
    neurons, features = [], []
    for i, (n, s) in enumerate(zip(photo, sq)):
        t = c.type[n]
        if t.startswith("R1-R6"):
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

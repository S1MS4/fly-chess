"""Backprop through time, on the brain: where the error from one move goes, step by step back.

  python tools/bptt.py                                       # in WSL: the backprop fly after 1.e4
  python tools/brain_grid.py --backward runs/bptt/e4.npz     # then on Windows: the pictures

The brain runs forward for 12 steps as usual, the fly's move probabilities get the training loss
against Stockfish's opinion of this position, and TensorFlow works out d loss / d u for every neuron
at every step: which way each neuron's input should move so that the fly agrees with Stockfish more,
and how much. Training never changes the brain by default, so this is the blame that would flow
if it did (the full 12 steps back, not the 6 that --window uses).
"""
import json
import sys
from pathlib import Path

import chess
import chess.engine
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.connectome import REGIONS
from flychess.encode import legal_codes, planes
from flychess.model import SD_FLOOR
from flychess.play import DLFly
from flychess.train_dl import CANDS, MAX_MOVES, PAD, move_logits, target

OUT = Path("runs/bptt")
DECADES = 3            # each picture shows blame over this many powers of ten below its step's biggest, smaller is grey

# What happens at each step on the way back, read off the numbers per region (printed when this runs).
PHRASES = {12: "the error arrives at the descending neurons, the only ones the readout reads",
           11: "one synapse back: the neurons that feed them", 10: "the ascending neurons and the central brain share it",
           9: "it spreads one synapse further each step, without growing",
           8: "the ascending neurons now carry more than the descending ones",
           7: "it starts to grow, and this is as far back as training looks (--window 6)",
           6: "past the window it grows ten times a step", 5: "neurons the board hasn't reached yet amplify it",
           4: "700 times what arrived at step 12", 3: "8,000 times", 2: "100,000 times",
           1: "a million times, and the eye gets almost none of it"}
LEGEND = ('<span><i style="background:#f5614f"></i>the loss wants this neuron\'s input higher</span>'
          '<span><i style="background:#4d99ff"></i>lower</span><span><i style="background:#575c66"></i>no blame</span>'
          '<br>Each picture is scaled to its own step, from the biggest blame down to a thousandth of it, because from step 7 '
          'back the blame grows about ten times per step. Under each caption: the regions with the most blame, as the '
          'average size of d loss / d input over their neurons.')


def amount(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 1 else f"{v:.2g}"


def lichess_win(cp: float) -> float:
    """Centipawns -> chance to win in %, the formula ChessBench's win% comes from."""
    return 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)


def stockfish_target(board: chess.Board, depth: int = 18):
    """Stockfish's best CANDS moves with their win%, in the same slots as the training data."""
    moves, codes = legal_codes(board)
    with chess.engine.SimpleEngine.popen_uci("stockfish") as sf:
        infos = sf.analyse(board, chess.engine.Limit(depth=depth), multipv=CANDS)
    cand_idx, cand_win = np.full(CANDS, 255, np.uint8), np.full(CANDS, -1, np.float32)
    for k, info in enumerate(infos):
        cand_idx[k] = moves.index(info["pv"][0])
        cand_win[k] = lichess_win(info["score"].pov(board.turn).score(mate_score=10_000))
    legal = np.full(MAX_MOVES, PAD, np.uint16)
    legal[:len(codes)] = codes
    return legal[None], cand_idx[None], cand_win[None], [(board.san(moves[i]), w) for i, w in zip(cand_idx, cand_win) if i != 255]


def blame(model, x, legal, cand_idx, cand_win):
    """d loss / d u at every step, (steps, neurons), and the loss."""
    b = model.brain
    inp = b.inputs(x)
    W = tf.sparse.SparseTensor(b.indices, b.weights_now(), (b.n, b.n))
    with tf.GradientTape() as tape:
        rates, us = tf.zeros_like(inp), []
        for t in range(b.steps):
            u = tf.sparse.sparse_dense_matmul(W, rates) + inp
            tape.watch(u)
            us.append(u)
            rates = (1 - b.leak) * rates + b.leak * tf.nn.relu(tf.tanh(b.gain * (u - b.mu[t]) * b.inv_sd[t] + b.bias))
        logits = move_logits(model.heads(model.descending(rates)), legal)
        loss = tf.nn.softmax_cross_entropy_with_logits(target(cand_idx, cand_win), logits)[0]
    return np.stack([g[:, 0].numpy() for g in tape.gradient(loss, us)]), float(loss), tf.nn.softmax(logits)[0].numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = connectome.load()
    fly = DLFly("checkpoints/dl", c=c)
    board = chess.Board()
    board.push_uci("e2e4")
    legal, ci, cw, opinion = stockfish_target(board)
    print("Stockfish:", ", ".join(f"{s} {w:.1f}%" for s, w in opinion))
    g, loss, p = blame(fly.model, planes(board).reshape(1, 768), legal, ci, cw)
    print(f"loss {loss:.3f}")

    # Per region and step: mean |blame|, for the captions and the README.
    regions = {r: np.where(c.region == r)[0] for r in REGIONS if (c.region == r).any()}
    per_region = [{r: float(np.abs(g[s, idx]).mean()) for r, idx in regions.items()} for s in range(len(g))]
    for s in range(len(g) - 1, -1, -1):
        top = sorted(((v, r) for r, v in per_region[s].items()), reverse=True)[:5]
        print(f"step {s + 1:2d}: " + ", ".join(f"{r} {v:.2e}" for v, r in top))

    # Bytes for the brain view: 128 = no blame, up = this neuron's input should go up to lower the loss,
    # down = it should go down. Log scale, each step from its own biggest blame down DECADES powers of ten.
    has = np.where(~np.isnan(c.soma[:, 0]))[0]
    mag = np.abs(g)
    hi = np.array([np.log10(np.percentile(m[m > 0], 99.9)) for m in mag])[:, None]
    level = np.clip((np.log10(np.maximum(mag, 1e-30)) - (hi - DECADES)) / DECADES, 0, 1)
    view = np.clip(128 - np.sign(g) * 127 * level, 0, 255).astype(np.uint8)[:, has]
    captions = []
    for s in range(len(g) - 1, -1, -1):                                # the way back: step 12 first
        top = sorted(((v, r) for r, v in per_region[s].items() if r != "other" and v > 0), reverse=True)[:3]
        captions.append({"title": f"Step {s + 1} of 12", "phrase": PHRASES[s + 1],
                         "numbers": ", ".join(f"{r} {amount(v)}" for v, r in top) + (", nothing else" if len(top) == 1 else "")})
    # Why it grows: neurons the board hasn't reached yet had no spread at all during calibration, so
    # homeostasis gives them its biggest gain (1 / SD_FLOOR), and the blame gets multiplied by it.
    floor = fly.model.brain.inv_sd.numpy()[:, :, 0] >= 1 / SD_FLOOR - 1e-3
    at_floor = [{r: float(floor[s, idx].mean()) for r, idx in regions.items()} | {"all": float(floor[s].mean())}
                for s in range(len(floor))]
    np.savez(OUT / "e4.npz", bytes=view, per_region=json.dumps(per_region), at_floor=json.dumps(at_floor), loss=loss,
             span=1.0, captions=json.dumps(captions), legend=json.dumps(LEGEND))
    print(f"saved {OUT / 'e4.npz'}; share of shown neurons with any blame, per step:",
          [round(float((view[s] != 128).mean()), 3) for s in range(len(view))])


if __name__ == "__main__":
    main()

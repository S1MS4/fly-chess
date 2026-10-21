"""Which piece does the fly move, against the piece Stockfish moves? A confusion matrix over the six piece types,
for the fly's favourite move on the 4,096 held-out positions (the ones no fly ever trains on).

  python tools/confusion.py --fly day       # about a minute on the GPU; writes results/confusion.json
  python tools/readme_figures.py confusion  # then docs/img/confusion.png from it

A move has thousands of possible "classes" (every from and to square), too many for a matrix, so the classes
here are the piece that moves: pawn, knight, bishop, rook, queen or king. A row is the piece Stockfish moved
(its best move), a column the piece the fly moved. The diagonal is "the right piece", which isn't yet the right
move: top-1 also needs the right square. The positions are rebuilt exactly from the ChessBench shards
(runs/imagine/hold.parquet, made by tools/imagine_lab.py --rows-only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess.encode import planes, policy
from flychess.play import FLIES, make_fly

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default="day", choices=[k for k, v in FLIES.items() if "weights" not in v])
    ap.add_argument("--out", type=Path, default=Path("results/confusion.json"))
    args = ap.parse_args()
    rows = pd.read_parquet("runs/imagine/hold.parquet")
    fly = make_fly(args.fly)
    boards = [chess.Board(f) for f in rows.fen]
    matrix, exact = np.zeros((6, 6), int), 0
    for i in range(0, len(boards), 256):
        chunk = boards[i:i + 256]
        fl, tl, pl, _ = (h.numpy() for h in fly.model(np.stack([planes(b).ravel() for b in chunk])))
        for j, b in enumerate(chunk):
            mine = policy(b, fl[j], tl[j], pl[j])[0][0]                # the fly's favourite legal move
            best = chess.Move.from_uci(rows.move.iloc[i + j])          # Stockfish's best move
            matrix[PIECES.index(b.piece_type_at(best.from_square)), PIECES.index(b.piece_type_at(mine.from_square))] += 1
            exact += mine == best
    rec = {"fly": args.fly, "positions": len(boards), "time": time.strftime("%Y-%m-%d %H:%M"),
           "classes": NAMES, "rows": "the piece Stockfish's best move moves", "columns": "the piece the fly's favourite move moves",
           "matrix": matrix.tolist(), "right_piece": round(float(np.trace(matrix) / matrix.sum()), 4),
           "top1": round(exact / len(boards), 4),
           "right_piece_by_class": {n: round(float(matrix[k, k] / max(matrix[k].sum(), 1)), 4) for k, n in enumerate(NAMES)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("right_piece", "top1", "right_piece_by_class")}), flush=True)


if __name__ == "__main__":
    main()

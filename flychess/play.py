"""Players: the two flies, plus the opponents they practise and get measured against.

Every player has .move(board) -> chess.Move. The flies also have .think(board),
which returns the move plus how active every brain region was at every step,
which is what the brain viewer draws.
"""
import os
import random
from pathlib import Path

import chess
import chess.engine
import numpy as np
import tensorflow as tf

from . import connectome
from .bio import BioFly
from .connectome import REGIONS
from .encode import planes, score_moves
from .model import FlyChess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board, color: bool) -> int:
    return sum(VALUES[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def region_activity(c, trace) -> list[dict]:
    """Mean firing rate per region per step, for the first board in the batch."""
    ids = {r: tf.constant(c.where(r)) for r in REGIONS if len(c.where(r))}
    return [{r: float(tf.reduce_mean(tf.gather(t[:, 0], i))) for r, i in ids.items()} for t in trace]


class DLFly:
    def __init__(self, checkpoint: str = "checkpoints/dl", c=None):
        self.c = c or connectome.load()
        self.model = FlyChess(self.c)
        self.model(np.zeros((1, 768), np.float32))
        latest = tf.train.latest_checkpoint(checkpoint)
        if latest:
            tf.train.Checkpoint(model=self.model).restore(latest).expect_partial()
        self.trained = bool(latest)

    def think(self, board: chess.Board):
        (fl, tl, pl, v), trace = self.model(planes(board).reshape(1, 768), return_trace=True)
        scored = score_moves(board, fl[0].numpy(), tl[0].numpy(), pl[0].numpy())
        move = max(scored, key=lambda s: s[1])[0]
        return move, {"value": float(v[0]), "activity": region_activity(self.c, trace),
                      "top": sorted(((m.uci(), s) for m, s in scored), key=lambda x: -x[1])[:5]}

    def move(self, board):
        return self.think(board)[0]


class BioPlayer:
    def __init__(self, weights: str = "checkpoints/bio/kc_mbon.npy", c=None, learn: bool = False):
        self.c = c or connectome.load()
        self.fly = BioFly(self.c)
        self.trained = Path(weights).exists()
        if self.trained:
            self.fly.load(weights)
        self.learn = learn

    def options(self, board):
        moves = list(board.legal_moves)
        me = board.turn
        x = []
        for m in moves:
            board.push(m)
            x.append(planes(board, me).ravel())  # the position after the move, seen from the fly's side
            board.pop()
        return moves, np.stack(x)

    def think(self, board):
        moves, x = self.options(board)
        i, score, trace = self.fly.choose(x, greedy=not self.learn, return_trace=True)
        order = np.argsort(-score)[:5]
        # Trace columns are one per option; show the chosen one.
        trace = [t[:, i:i + 1] for t in trace]
        return moves[i], {"activity": region_activity(self.c, trace),
                          "top": [(moves[j].uci(), float(score[j])) for j in order]}

    def move(self, board):
        if self.learn:
            moves, x = self.options(board)
            return moves[self.fly.choose(x)[0]]
        return self.think(board)[0]


class RandomPlayer:
    name = "random"

    def move(self, board):
        return random.choice(list(board.legal_moves))


class GreedyPlayer:
    """Takes the most valuable piece it can, gives check if it can't, otherwise random."""
    name = "greedy"

    def move(self, board):
        moves = list(board.legal_moves)
        random.shuffle(moves)
        def gain(m):
            cap = board.piece_at(m.to_square)
            return (VALUES[cap.piece_type] if cap else 0) + (0.5 if board.gives_check(m) else 0)
        return max(moves, key=gain)


class StockfishPlayer:
    """Stockfish held to a set Elo. Needs the stockfish binary: set STOCKFISH=/path/to/stockfish
    or have it on PATH. UCI_Elo goes down to 1320; below that use skill level instead."""

    def __init__(self, elo: int | None = 1320, skill: int | None = None, movetime: float = 0.05):
        path = os.environ.get("STOCKFISH", "stockfish")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            self.name = f"stockfish-{elo}"
        else:
            self.engine.configure({"Skill Level": skill})
            self.name = f"stockfish-skill{skill}"
        self.limit = chess.engine.Limit(time=movetime)

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


def play_game(white, black, max_moves: int = 200, on_move=None) -> float:
    """Returns 1 for a white win, 0 for a black win, 0.5 for a draw."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        player = white if board.turn == chess.WHITE else black
        board.push(player.move(board))
        if on_move:
            on_move(board)
    r = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)

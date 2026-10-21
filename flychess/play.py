"""Players: the two flies, plus the opponents they practise and get measured against.

Every player has .move(board) -> chess.Move. The flies also have .think(board),
which returns the move plus everything the interface shows: a probability for
every legal move, and how active every neuron was at every step.
"""
import json
import os
import random
from pathlib import Path

import chess
import chess.engine
import numpy as np
import tensorflow as tf

from . import connectome
from .bio import BioFly
from .encode import planes, policy
from .model import FlyChess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board, color: bool) -> int:
    return sum(VALUES[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def position_key(board: chess.Board) -> str:
    """What makes two positions the same for the repetition rule: pieces, side to move, castling, en passant."""
    return " ".join(board.fen().split()[:4])


def fresh_move(board: chess.Board, ranked, seen=()) -> chess.Move:
    """The fly sees the board, not the game, so on its own it walks into threefold repetition
    even a queen up. It plays its favourite move that doesn't bring back an earlier position.
    ranked: [(move, p)] best first. seen: position keys from before this board's move stack."""
    for m, _ in ranked:
        board.push(m)
        again = board.is_repetition(2) or position_key(board) in seen
        board.pop()
        if not again:
            return m
    return ranked[0][0]                                                 # every move repeats: take the favourite anyway


def drawn(board: chess.Board, seen=()) -> bool:
    """Whether the game ends here as a draw, the way play_game and the app call it: stalemate, too little
    material, a position that has really been there three times (on the board's move stack, or in `seen`,
    the app's earlier positions of a board that came from a FEN), or 50 moves without a capture or pawn move."""
    if board.is_game_over():                                           # checkmate is looked at before this
        return not board.is_checkmate()
    return board.is_repetition(3) or list(seen).count(position_key(board)) >= 2 or board.is_fifty_moves()


def look_ahead(board: chess.Board, probs, value, weight: float, seen=()):
    """One move ahead and no further: the board after each of the fly's own moves, never the replies.
    A move that checkmates gets played. A move that ends the game in a draw by the rules is worth exactly a
    draw, 0, and it's out whenever another move leaves the fly better off than that. Every other move scores
    the log of its probability plus `weight` times what the fly's own value readout says about the position
    after it (from the opponent's side, who moves next, so with the sign turned).
    probs: [(move, p)] from the fly's move readout. value: (n, 768) boards -> (n,) values for the side to move.
    Returns the move, and per move what it imagined: {"uci", "p", "end" ("mate", "draw" or None), "value"}."""
    moves = [m for m, _ in probs]
    p = np.array([q for _, q in probs], np.float64)
    ends, again, x = [], [], []
    for m in moves:
        board.push(m)
        ends.append("mate" if board.is_checkmate() else "draw" if drawn(board, seen) else None)
        again.append(ends[-1] is None and (board.is_repetition(2) or position_key(board) in seen))   # as fresh_move
        if ends[-1] is None:
            x.append(planes(board).ravel())
        board.pop()
    live = np.array([e is None for e in ends])
    q = np.zeros(len(moves))
    if x:
        q[live] = -np.asarray(value(np.stack(x)), np.float64)
    q[[e == "mate" for e in ends]] = 1.0
    best = choose(p, q, ends, weight, again)
    seen_ahead = [{"uci": m.uci(), "p": float(p[i]), "end": ends[i], "value": round(float(q[i]), 4)} for i, m in enumerate(moves)]
    return moves[best], seen_ahead


def choose(p: np.ndarray, q: np.ndarray, ends: list, weight: float, again=None) -> int:
    """The rule look_ahead() plays by, given what it saw: each move's probability p, its value q for the fly,
    how the game ends after it, and whether it brings back an earlier position (tools/imagine_lab.py tunes
    `weight` with this same rule). While some move leaves the fly better off than a draw, drawing moves are out,
    and so are moves back to an earlier position: the opponent's reply can make that the third time (the first
    imagination games drew 21 of 120 that way, 2026-10-21)."""
    if "mate" in ends:
        return max((i for i, e in enumerate(ends) if e == "mate"), key=lambda i: p[i])
    logp = np.log(np.maximum(p, 1e-12))
    score = logp if weight == 0 else q + logp / weight                 # the same order as logp + weight * q, inf too
    live = np.array([e is None for e in ends])
    fresh = live & ~np.array(again if again is not None else [False] * len(ends), bool)
    if fresh.any() and q[fresh].max() > 0:                             # better off without a draw or going back
        score = np.where(fresh, score, -np.inf)
    elif live.any() and q[live].max() > 0:                             # only going back keeps it ahead: still no draws
        score = np.where(live, score, -np.inf)
    return int(np.argmax(score))


def imagine_weight(checkpoint: str | None) -> float:
    """How much the fly's imagination trusts its value readout, tuned on held-out positions by
    tools/imagine_lab.py and kept next to the checkpoint. 1 until it's been tuned."""
    try:
        return float(json.loads((Path(checkpoint) / "imagine.json").read_text())["weight"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0


def sample_boards(n: int = 256, seed: int = 0) -> np.ndarray:
    """Positions from random games, to calibrate an untrained brain's homeostasis."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(4, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        out.append(planes(board).ravel())
    return np.stack(out)


class DLFly:
    def __init__(self, checkpoint: str = "checkpoints/dl", c=None, eye: str = "ranks", read=(12,), imagine: bool = False):
        self.c = c or connectome.load()
        self.model = FlyChess(self.c, eye=eye, read=tuple(read))
        self.model(np.zeros((1, 768), np.float32))
        latest = tf.train.latest_checkpoint(checkpoint)
        if latest:
            tf.train.Checkpoint(model=self.model).restore(latest).expect_partial()
        else:
            self.model.brain.calibrate(sample_boards())
        self.trained = bool(latest)
        self.imagine, self.weight = imagine, imagine_weight(checkpoint)   # one move ahead, see look_ahead()

    def policy(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        fl, tl, pl, _ = (h[0].numpy() for h in self.model(planes(board).reshape(1, 768)))
        return policy(board, fl, tl, pl)

    def values(self, boards: np.ndarray) -> np.ndarray:
        """The value readout for a batch of (n, 768) boards, each for its side to move."""
        return self.model(boards)[3].numpy()

    def think(self, board: chess.Board, seen=(), imagine: bool | None = None):
        """The move, a probability for every legal move, and every neuron's rate at every step.
        imagine: look one move ahead this time (None: whatever the fly was built with)."""
        (fl, tl, pl, v), trace = self.model(planes(board).reshape(1, 768), return_trace=True)
        probs = policy(board, fl[0].numpy(), tl[0].numpy(), pl[0].numpy())
        info = {"value": float(v[0]), "candidates": probs, "rates": np.stack([t[:, 0].numpy() for t in trace])}
        if self.imagine if imagine is None else imagine:
            m, info["imagined"] = look_ahead(board, probs, self.values, self.weight, seen)
            return m, info
        return fresh_move(board, probs, seen), info

    def trace(self, boards: np.ndarray) -> np.ndarray:
        """(steps, neurons, boards) rates, for a batch of boards."""
        return np.stack([t.numpy() for t in self.model.brain(boards, return_trace=True)])

    def move(self, board):
        if self.imagine:
            return look_ahead(board, self.policy(board), self.values, self.weight)[0]
        return fresh_move(board, self.policy(board))


class BioPlayer:
    def __init__(self, weights: str = "checkpoints/bio/kc_mbon.npz", c=None, learn: bool = False, eye: str = "ranks", **bio):
        self.c = c or connectome.load()
        self.fly = BioFly(self.c, eye=eye, **bio)
        self.trained = Path(weights).exists()
        if self.trained:
            self.fly.load(weights)
        else:
            self.fly.calibrate(sample_boards())
        self.learn = learn

    def options(self, board):
        """Every position the fly could move into, seen from its own side."""
        moves = list(board.legal_moves)
        me = board.turn
        x = []
        for m in moves:
            board.push(m)
            x.append(planes(board, me).ravel())
            board.pop()
        return moves, np.stack(x)

    def probabilities(self, score: np.ndarray) -> np.ndarray:
        p = np.exp((score - score.max()) / self.fly.temp)
        return p / p.sum()

    def policy(self, board):
        moves, x = self.options(board)
        _, score = self.fly.choose(x, greedy=True, tag=False)
        p = self.probabilities(score)
        return sorted(zip(moves, p.tolist()), key=lambda mp: -mp[1])

    def think(self, board, seen=()):
        moves, x = self.options(board)
        _, score, trace = self.fly.choose(x, greedy=True, tag=False, return_trace=True)
        p = self.probabilities(score)
        probs = sorted(zip(moves, p.tolist()), key=lambda mp: -mp[1])
        m = fresh_move(board, probs, seen)
        i = moves.index(m)
        return m, {"candidates": probs, "rates": np.stack([t[:, i].numpy() for t in trace])}  # the chosen option's brain

    def trace(self, boards: np.ndarray) -> np.ndarray:
        return np.stack([t.numpy() for t in self.fly.run(boards, return_trace=True)])

    def move(self, board):
        if self.learn:                                                  # learning from games: sample, and tag the Kenyon cells
            moves, x = self.options(board)
            return moves[self.fly.choose(x, tag=True)[0]]
        return fresh_move(board, self.policy(board))


# Every fly you can play. Besides the two trained ones, the control brains from the experiments in the
# README: the same 3-minute readout probe on the real, the shuffled and the lesioned wiring, and a fly
# that never trained at all.
FLIES = {
    "dl": {"checkpoint": "checkpoints/dl"},
    "day": {"checkpoint": "checkpoints/dl-day/best", "eye": "fair", "read": (9, 10, 11, 12)},   # scripts/train_day.sh
    "day-imagine": {"checkpoint": "checkpoints/dl-day/best", "eye": "fair", "read": (9, 10, 11, 12), "imagine": True},
    "bio": {"weights": "checkpoints/bio/kc_mbon.npz"},
    "quick": {"checkpoint": "checkpoints/probe-real"},
    "shuffled": {"checkpoint": "checkpoints/probe-shuffled", "shuffle": True},
    "no-optic-lobe": {"checkpoint": "checkpoints/probe-no-optic-lobe", "lesion": ["optic lobe"]},
    "no-central-brain": {"checkpoint": "checkpoints/probe-no-central-brain", "lesion": ["central brain"]},
    "newborn": {"checkpoint": None},
}


def make_fly(kind: str, c=None):
    """The fly `kind` from FLIES, built on the real connectome `c` (loaded if not given)."""
    spec = FLIES[kind]
    c = c or connectome.load()
    if spec.get("shuffle"):
        c = c.shuffled()
    if spec.get("lesion"):
        c = c.lesioned(spec["lesion"])
    eye = spec.get("eye", "ranks")                                     # the first flies all learned with the first eye
    if "weights" in spec:
        return BioPlayer(spec["weights"], c=c, eye=eye, **spec.get("bio", {}))
    if spec["checkpoint"] is None:                                     # no checkpoint: calibrated, readout random
        tf.keras.utils.set_random_seed(0)                              # the same newborn every time
        return DLFly("checkpoints/newborn", c=c, eye=eye)
    return DLFly(spec["checkpoint"], c=c, eye=eye, read=spec.get("read", (12,)), imagine=spec.get("imagine", False))


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
        if elo is not None:
            self.options, self.name = {"UCI_LimitStrength": True, "UCI_Elo": elo}, f"stockfish-{elo}"
        else:
            self.options, self.name = {"Skill Level": skill}, f"stockfish-skill{skill}"
        self.limit = chess.engine.Limit(time=movetime)
        self.engine = self.start()

    def start(self):
        engine = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "stockfish"))
        engine.configure(self.options)
        return engine

    def move(self, board):
        for _ in range(5):                       # the heat guard can pause us for longer than python-chess waits (10 s)
            try:
                return self.engine.play(board, self.limit).move
            except (TimeoutError, chess.engine.EngineTerminatedError):
                self.engine.close()                                    # a fresh engine, then the same move again
                self.engine = self.start()
        return self.engine.play(board, self.limit).move

    def close(self):
        try:
            self.engine.quit()
        except (TimeoutError, chess.engine.EngineError):
            self.engine.close()                                        # stop it anyway: a live engine keeps Python from exiting


class MixedPlayer:
    """Stockfish at its weakest setting that plays a random legal move a share `p` of the time: the rungs
    of the rating ladder between Stockfish 1320 and a random mover."""

    def __init__(self, p: float, elo: int = 1320, seed: int = 0):
        self.p, self.rng = p, random.Random(seed)
        self.sf = StockfishPlayer(elo=elo)
        self.name = f"stockfish-{elo}-{round(p * 100)}%random"

    def move(self, board):
        if self.rng.random() < self.p:
            return self.rng.choice(list(board.legal_moves))
        return self.sf.move(board)

    def close(self):
        self.sf.close()


def over(board: chess.Board) -> bool:
    """Draws the way a referee calls them: a position that really happened three times, or 50
    moves without a capture or pawn move. claim_draw=True would also end the game when the side
    to move *could* repeat, and a random mover never would."""
    return board.is_game_over() or board.is_repetition(3) or board.is_fifty_moves()


def ending(board: chess.Board) -> str:
    """How a game that over() stopped (or the move cap) ended."""
    if o := board.outcome():                                           # checkmate, stalemate, insufficient material, ...
        return o.termination.name.lower().replace("_", " ")
    if board.is_repetition(3):
        return "threefold repetition"
    return "fifty moves" if board.is_fifty_moves() else "move cap"


def play_game(white, black, max_moves: int = 200, on_move=None, info: dict | None = None) -> float:
    """Returns 1 for a white win, 0 for a black win, 0.5 for a draw. Pass a dict as `info` and it also
    gets how the game ended, how many moves it took, and white's material lead at the end and at its
    biggest either way (pawn 1, knight and bishop 3, rook 5, queen 9)."""
    board = chess.Board()
    lead = [0]
    while not over(board) and board.fullmove_number <= max_moves:
        player = white if board.turn == chess.WHITE else black
        board.push(player.move(board))
        lead.append(material(board, chess.WHITE))
        if on_move:
            on_move(board)
    if info is not None:
        info.update({"end": ending(board), "moves": board.fullmove_number, "material": lead[-1],
                     "material_max": max(lead), "material_min": min(lead)})
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)

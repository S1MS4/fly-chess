"""Chess <-> numbers.

The fly always plays "white". When it is black to move the board is mirrored
(ranks flipped, colours swapped), so the same neurons handle both sides.

A position becomes 12 planes of 64 squares: one plane per piece type and colour,
1 where that piece stands. A move becomes (from square, to square, promotion).
"""
import chess
import numpy as np

PROMO = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]  # index 0 = no promotion
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def oriented(board: chess.Board, color: bool | None = None) -> tuple[chess.Board, bool]:
    """Return the board from one side's point of view (default: the side to move),
    and whether it was flipped."""
    color = board.turn if color is None else color
    return (board, False) if color == chess.WHITE else (board.mirror(), True)


def planes(board: chess.Board, color: bool | None = None) -> np.ndarray:
    """12 x 64 piece planes (own pieces first, then the opponent's), already oriented."""
    b, _ = oriented(board, color)
    out = np.zeros((12, 64), np.float32)
    for sq, p in b.piece_map().items():
        out[PIECES.index(p.piece_type) + (0 if p.color == chess.WHITE else 6), sq] = 1.0
    return out


def move_to_target(board: chess.Board, move: chess.Move) -> tuple[int, int, int]:
    """(from, to, promo index) in the oriented frame."""
    _, flipped = oriented(board)
    f, t = move.from_square, move.to_square
    if flipped:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f, t, PROMO.index(move.promotion)


def score_moves(board: chess.Board, from_logits, to_logits, promo_logits) -> list[tuple[chess.Move, float]]:
    """Score every legal move as from + to + promotion logit. Illegal moves never get a score."""
    scored = []
    for m in board.legal_moves:
        f, t, p = move_to_target(board, m)
        scored.append((m, float(from_logits[f] + to_logits[t] + promo_logits[p])))
    return scored

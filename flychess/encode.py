"""Chess <-> numbers.

The fly always plays "white". When it is black to move the board is mirrored
(ranks flipped, colours swapped), so the same neurons handle both sides.

A position becomes 12 planes of 64 squares: one plane per piece type and colour,
1 where that piece stands. A move becomes (from square, to square, promotion),
packed into one number: promotion * 4096 + from * 64 + to.
"""
import chess
import numpy as np

PROMO = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]  # index 0 = no promotion
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PAD = 0xFFFF  # an empty slot in a list of move codes


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
    f, t = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f, t, PROMO.index(move.promotion)


def move_code(board: chess.Board, move: chess.Move) -> int:
    f, t, p = move_to_target(board, move)
    return p * 4096 + f * 64 + t


def legal_codes(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    return moves, np.array([move_code(board, m) for m in moves], np.int64)


def policy(board: chess.Board, from_logits, to_logits, promo_logits) -> list[tuple[chess.Move, float]]:
    """Probability of every legal move, best first. Illegal moves never get a share.

    A move's score is its from + to + promotion logit, and softmax turns the scores
    into probabilities that add up to 1, the same way the training loss sees them."""
    moves, codes = legal_codes(board)
    s = from_logits[codes // 64 % 64] + to_logits[codes % 64] + promo_logits[codes // 4096]
    p = np.exp(s - s.max())                                             # softmax, shifted so exp() can't overflow
    p /= p.sum()
    return sorted(zip(moves, p.tolist()), key=lambda mp: -mp[1])

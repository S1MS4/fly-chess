"""The fly's one-move imagination (flychess.play.look_ahead): it plays a mate in one even when its move readout
barely likes it, skips a stalemate while it's winning, takes the draw while it's losing, and counts a position
the app saw twice before as a draw. A made-up value readout stands in for the brain, so this runs in seconds.

  python -m pytest tests     (or just: python tests/test_imagination.py)
"""
import chess
import numpy as np

from flychess.play import drawn, look_ahead, position_key

PAWN_ENDING = "k7/2K5/1P6/8/8/8/8/8 w - - 0 1"     # Kc8 stalemates the black king; no mate in one


def priors(board, favourite: str, share: float):
    """The move readout's probabilities: `share` for one move, the rest spread evenly."""
    moves = list(board.legal_moves)
    rest = (1 - share) / (len(moves) - 1)
    return sorted(((m, share if m.uci() == favourite else rest) for m in moves), key=lambda mp: -mp[1])


def opponent_feels(v: float):
    """A value readout that says the same thing about every position: v for the side to move (the opponent)."""
    return lambda x: np.full(len(x), v)


def test_mate_in_one_wins_over_the_favourite():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")    # Rd8 is mate
    probs = priors(board, "g1f1", 0.999)
    move, seen = look_ahead(board, probs, opponent_feels(0.0), weight=1.0)
    assert move.uci() == "d1d8"
    assert next(s for s in seen if s["uci"] == "d1d8")["end"] == "mate"


def test_stalemate_is_out_while_winning():
    board = chess.Board(PAWN_ENDING)
    assert any(m.uci() == "c7c8" for m in board.legal_moves)
    board.push(chess.Move.from_uci("c7c8"))
    assert board.is_stalemate()
    board.pop()
    probs = priors(board, "c7c8", 0.9)                                 # its favourite stalemates
    move, seen = look_ahead(board, probs, opponent_feels(-0.8), weight=1.0)   # every other move: black is lost
    assert move.uci() != "c7c8"
    assert next(s for s in seen if s["uci"] == "c7c8")["end"] == "draw"


def test_draw_is_taken_while_losing():
    board = chess.Board(PAWN_ENDING)
    probs = priors(board, "c7c8", 0.01)                                # it barely likes the stalemate
    move, _ = look_ahead(board, probs, opponent_feels(0.9), weight=8.0)   # every other move: black is winning
    assert move.uci() == "c7c8"


def test_no_ending_in_sight_follows_the_move_readout():
    board = chess.Board()
    probs = priors(board, "e2e4", 0.5)
    move, _ = look_ahead(board, probs, opponent_feels(0.0), weight=4.0)
    assert move.uci() == "e2e4"


def test_no_going_back_while_winning():
    board = chess.Board("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    for uci in ("a1a2", "h8g8", "a2a1", "g8h8"):                       # the rook and the king shuffle back
        board.push_uci(uci)
    probs = priors(board, "a1a2", 0.9)                                 # its favourite brings back a2 a second time
    move, _ = look_ahead(board, probs, opponent_feels(-0.8), weight=1.0)
    assert move.uci() != "a1a2"
    move, _ = look_ahead(board, probs, opponent_feels(0.8), weight=1.0)   # losing: going back is fine
    assert move.uci() == "a1a2"


def test_the_apps_earlier_positions_count_for_repetition():
    board = chess.Board()
    board.push_uci("g1f3")
    key = position_key(board)
    assert not drawn(board, seen=[key])                                # been there once before: twice in all
    assert drawn(board, seen=[key, key])                               # twice before: the third time


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")

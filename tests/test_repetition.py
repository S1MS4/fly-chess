"""fresh_move: the fly plays its favourite move, unless that brings back an earlier position.

  python -m pytest tests     (or just: python tests/test_repetition.py)
"""
import chess

from flychess.play import fresh_move, position_key

BACK, OTHER = chess.Move.from_uci("f6g8"), chess.Move.from_uci("e7e5")
RANKED = [(BACK, 0.9), (OTHER, 0.1)]                                 # the fly loves going back


def test_skips_a_move_that_repeats_the_game():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    assert fresh_move(board, RANKED) == OTHER                        # Ng8 would be the start position again


def test_uses_the_positions_the_browser_sends():
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 3 2")  # same position, no move history
    assert fresh_move(board, RANKED) == BACK                         # on its own it can't know
    assert fresh_move(board, RANKED, seen={position_key(chess.Board())}) == OTHER


def test_takes_the_favourite_when_every_move_repeats():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    assert fresh_move(board, [(BACK, 1.0)]) == BACK


if __name__ == "__main__":
    test_skips_a_move_that_repeats_the_game()
    test_uses_the_positions_the_browser_sends()
    test_takes_the_favourite_when_every_move_repeats()
    print("repetition rule works")

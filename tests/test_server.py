"""The web server's chess rules: draws only once they really happen, and opening names by move order.

  python -m pytest tests     (or just: python -m tests.test_server, from the project root)
"""
import chess

from app.server import opening, opening_name, outcome, position, position_key


def test_threefold_only_on_the_third_time():
    board, seen = chess.Board(), []                                  # seen: what the browser sends, every earlier position
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:                 # knights out and back, twice
        seen.append(position_key(board))
        board.push_uci(uci)
    now = chess.Board(board.fen())                                   # what the server gets: a FEN, no move history
    assert outcome(now, seen[:4]) is None                            # the start position once before: this is the second time
    assert outcome(now, seen) == "threefold repetition, draw"        # twice before: the third time


def test_fifty_moves_only_once_they_happened():
    assert outcome(chess.Board("8/8/4k3/8/8/4K3/8/R7 w - - 99 80")) is None    # python-chess's claim_draw would end it here
    assert outcome(chess.Board("8/8/4k3/8/8/4K3/8/R7 b - - 100 80")) == "fifty moves, draw"


def test_opening_named_by_move_order():
    assert opening_name("e2e4,g8f6,g1f3")["name"] == "Alekhine Defense"
    board = chess.Board()
    for uci in ("e2e4", "g8f6", "g1f3"):
        board.push_uci(uci)
    assert opening(board)["name"] == "Zukertort Opening: Lemberger Gambit"  # the same position, reached by 1.Nf3 Nf6 2.e4
    assert opening_name("") is None


def test_every_legal_move_knows_where_it_leads():
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3KB1R w KQkq - 4 8")
    for m in position(board)["legal"]:
        after = board.copy()
        after.push_uci(m["uci"])
        assert m["after"] == after.fen()


def test_imagination_switch_reaches_the_fly():
    """The app's Imagination switch: imagine=true gets to a backprop fly's think(), with the earlier positions as a
    list (a repetition needs their count), and the reply says what it imagined. A stand-in fly, no brain needed."""
    import app.server as server
    asked = []

    class Stub:                                                        # plays its first legal move, like a very dull fly
        trained, imagine = True, False

        def think(self, board, seen=(), imagine=None):
            asked.append((imagine, seen))
            m = next(iter(board.legal_moves))
            info = {"value": 0.0, "candidates": [(m, 1.0)], "rates": None}
            if imagine:
                info["imagined"] = [{"uci": m.uci(), "p": 1.0, "end": None, "value": 0.25}]
            return m, info

        def policy(self, board):
            return [(next(iter(board.legal_moves)), 1.0)]

    fly, view = server.fly, server.brain_view
    server.fly, server.brain_view = (lambda kind: Stub()), (lambda kind, rates: (b"", []))
    try:
        plain = server.play(server.Play(fen=chess.STARTING_FEN, fly="day"))
        ahead = server.play(server.Play(fen=chess.STARTING_FEN, fly="day", imagine=True, seen=["x", "x"]))
    finally:
        server.fly, server.brain_view = fly, view
    assert asked[0][0] is None and plain["fly"]["imagined"] is None
    assert asked[1] == (True, ["x", "x"]) and list(ahead["fly"]["imagined"].values())[0]["value"] == 0.25


if __name__ == "__main__":
    test_threefold_only_on_the_third_time()
    test_fifty_moves_only_once_they_happened()
    test_opening_named_by_move_order()
    test_every_legal_move_knows_where_it_leads()
    test_imagination_switch_reaches_the_fly()
    print("server rules work")

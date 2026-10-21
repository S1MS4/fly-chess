"""Lessons in finishing a game: won positions, labelled so that winning faster scores higher.

ChessBench gives every move the chance to win after it, and once a game is won that's 100% for almost
any move. In a queen-up endgame all 8 of Stockfish's top moves read 100.0, mate in one and a pointless
king walk alike, so the fly learns that every move is as good as mate. Here Stockfish ranks them again
by how much they win by: centipawns, and for a forced mate, how many moves it takes.

  python scripts/make_curriculum.py           # about 40 minutes on 4 cores, and it can be stopped and restarted
  python -m flychess.train_dl --prepare       # packs these too
  python -m flychess.train_dl --curriculum 0.1 ...   # a tenth of every batch from here

Two kinds of positions, both with the winning side to move:
  decided    from the downloaded ChessBench files, where the best move wins 99.5% or more
  endgames   random placements of the basic mates: KQ vs K, KR vs K, two rooks, queen against a minor
             piece, two bishops, and a pawn or two to promote
  stalemates (--stalemates N) a lot of material against a bare king, with at least one move that stalemates it

    python scripts/make_curriculum.py --stalemates 20000 --only-stalemates   # just those, about 10 minutes

Stockfish (depth 12, its 8 best moves) scores each move. A mate in N counts as 10,000 - 100 N
centipawns, and every 100 centipawns behind the best move costs 2.5 points of "win%", so the training
target (TAU = 2.5) gives a move 100 centipawns or one mate move slower 1/e of the best move's share:

    win%(move) = 100 - (score(best) - score(move)) / 40

Mate in 5 against mate in 6: 100 against 97.5. A stalemate (0 centipawns) next to a mate: 0.
Positions from the held-out rows of the main data, and a random 5% of the rest, are kept for testing.
Writes data/curriculum/{decided,endgames}.parquet in the ChessBench columns plus "split".
"""
import argparse
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data/positions")
OUT = Path("data/curriculum")
DEPTH, LINES, CHUNK = 12, 8, 2000
SETS = {"KQ": 3, "KR": 3, "KRR": 1, "KQ.N": 1, "KQ.B": 1, "KQ.P": 1, "KBB": 1, "KP": 1, "KPP": 1, "KQR": 1}   # weight of each
engine = None


def score(s: chess.engine.PovScore, turn: bool) -> float:
    """Centipawns for the side to move, with a mate as 10,000 minus 100 per move it takes."""
    s = s.pov(turn)
    if s.is_mate():
        n = s.mate()
        return 10_000 - 100 * n if n > 0 else -10_000 - 100 * n
    return max(-5_000, min(5_000, s.score()))


def label(fen: str):
    """(fen, best move, value, moves, win%s) like a ChessBench row, or None if it isn't won after all."""
    global engine
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "stockfish"))
        engine.configure({"Threads": 1, "Hash": 32})
    board = chess.Board(fen)
    infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=min(LINES, board.legal_moves.count()))
    scored = sorted(((score(i["score"], board.turn), i["pv"][0].uci()) for i in infos if i.get("pv")), reverse=True)
    if not scored or scored[0][0] < 500:                              # not a won position for the side to move
        return None
    wins = [max(0.0, 100 - (scored[0][0] - s) / 40) for s, _ in scored]
    return fen, scored[0][1], 1.0, " ".join(m for _, m in scored), " ".join(f"{w:.1f}" for w in wins)


TRAPS = {"KQ": 4, "KR": 2, "KQR": 2, "KRR": 1, "KQQ": 1, "KQB": 1, "KQN": 1, "KRB": 1, "KQP": 1}   # weight of each


def stalemates_after(board: chess.Board) -> list[chess.Move]:
    """The legal moves that would stalemate the other side."""
    out = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_stalemate():
            out.append(m)
        board.pop()
    return out


def stalemate_trap(rng: random.Random) -> str:
    """White to move, far ahead against a bare king that's mostly on the edge, with at least one legal move
    that stalemates it: the positions the day-long fly drew 39 of 120 rated games from (2026-10-21)."""
    edge = [s for s in chess.SQUARES if chess.square_file(s) in (0, 7) or chess.square_rank(s) in (0, 7)]
    while True:
        name = rng.choices(list(TRAPS), weights=list(TRAPS.values()))[0]
        board = chess.Board(None)
        lone = rng.choice(edge) if rng.random() < 0.8 else rng.choice(chess.SQUARES)
        others = rng.sample([s for s in chess.SQUARES if s != lone], len(name))
        board.set_piece_at(lone, chess.Piece(chess.KING, chess.BLACK))
        for sq, symbol in zip(others, name):
            board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
        board.turn = chess.WHITE
        pawns_ok = all(chess.square_rank(sq) not in (0, 7) for sq in board.pieces(chess.PAWN, chess.WHITE))
        if pawns_ok and board.is_valid() and not board.is_game_over() and stalemates_after(board):
            return board.fen()


def random_endgame(rng: random.Random) -> str:
    """White to move with the extra material, black's king not in check, not over yet."""
    name = rng.choices(list(SETS), weights=list(SETS.values()))[0]
    strong, _, weak = name.partition(".")
    pieces = [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK)]
    pieces += [(chess.Piece.from_symbol(p).piece_type, chess.WHITE) for p in strong[1:]]
    pieces += [(chess.Piece.from_symbol(p).piece_type, chess.BLACK) for p in weak]
    while True:
        board = chess.Board(None)
        for sq, (kind, color) in zip(rng.sample(chess.SQUARES, len(pieces)), pieces):
            board.set_piece_at(sq, chess.Piece(kind, color))
        board.turn = chess.WHITE
        pawns_ok = all(chess.square_rank(sq) not in (0, 7) for sq in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK))
        if pawns_ok and board.is_valid() and not board.is_game_over():
            return board.fen()


def held_out_fens() -> set[str]:
    """The positions train_dl never trains on (its HOLD and VAL rows), found the way --prepare orders them."""
    import numpy as np
    from flychess.train_dl import TRAIN0, pack                         # imports TensorFlow, so only after the workers exist
    rows = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob("shard_*.parquet"))])
    order = np.random.RandomState(0).choice(len(rows), size=len(rows), replace=False)   # what .sample(frac=1, random_state=0) does
    out = set()
    for i in order:
        row = tuple(rows.iloc[i])
        if pack(row) is not None:
            out.add(row[0])
            if len(out) == TRAIN0:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endgames", type=int, default=12_000)
    ap.add_argument("--workers", type=int, default=4, help="one Stockfish each; more cores means more heat")
    ap.add_argument("--limit", type=int, default=None, help="only this many decided positions (a quick test)")
    ap.add_argument("--stalemates", type=int, default=0, help="also this many won endgames with a stalemate trap")
    ap.add_argument("--only-stalemates", action="store_true", help="just those (the other two are made already)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers)                                           # forked now, while this process is still small
    held = set() if args.only_stalemates else held_out_fens()
    sets = []
    if not args.only_stalemates:
        rows = pd.concat([pd.read_parquet(f, columns=["fen", "wins"]) for f in sorted(DATA.glob("shard_*.parquet"))])
        decided = rows[rows.wins.str.split().str[0].astype(float) >= 99.5].fen.drop_duplicates().tolist()[:args.limit]
        rng = random.Random(0)
        endgames = list(dict.fromkeys(random_endgame(rng) for _ in range(args.endgames)))
        print(f"{len(decided)} decided positions ({sum(f in held for f in decided)} held out), {len(endgames)} endgames", flush=True)
        sets += [("endgames", endgames), ("decided", decided)]
    if args.stalemates:
        rng = random.Random(2)
        traps = list(dict.fromkeys(stalemate_trap(rng) for _ in range(args.stalemates)))
        print(f"{len(traps)} stalemate traps", flush=True)
        sets.append(("stalemates", traps))
    columns = ["fen", "move", "value", "cands", "wins"]
    for name, fens in sets:
        parts = OUT / f"{name}.parts"                                   # saved as it goes: a stopped run picks up here
        parts.mkdir(exist_ok=True)
        for k in range(0, len(fens), CHUNK):
            part = parts / f"{k // CHUNK:04d}.parquet"
            if not part.exists():
                pd.DataFrame([r for r in pool.imap(label, fens[k:k + CHUNK], chunksize=8) if r], columns=columns).to_parquet(part)
                print(f"{name}: {min(k + CHUNK, len(fens))} of {len(fens)} labelled", flush=True)
        df = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)
        split_rng = random.Random(1)
        df["split"] = ["test" if f in held or split_rng.random() < 0.05 else "train" for f in df.fen]
        df.to_parquet(OUT / f"{name}.parquet")
        print(f"saved {OUT / name}.parquet: {len(df)} positions, {(df.split == 'test').sum()} for testing", flush=True)
    pool.terminate()                                                    # close() + join() would wait forever: each worker's engine
                                                                        # has a thread that never ends; Stockfish quits when its pipe closes


if __name__ == "__main__":
    main()

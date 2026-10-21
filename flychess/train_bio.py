"""Train the biological fly: no backprop, only dopamine.

  python -m flychess.train_bio --minutes 20       # lessons from a tutor (fast, the default)
  python -m flychess.train_bio --games 2000       # or learn by playing games, paid in dopamine

Tutor: like a fly in a lab experiment. It is shown the position after Stockfish's
move and the position after some other legal move, and gets sugar for the first and
a shock for the second. 64 lessons go through the brain at once, so this is
hundreds of times faster than learning from whole games.

Games: after every one of the fly's moves the opponent replies, and the fly gets a
small dose of dopamine for what happened in between (material won or lost, check),
and a big one for the result at the end. Opponents get harder as it improves:
random, then greedy, then Stockfish.
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from . import connectome
from .encode import planes
from .play import BioPlayer, GreedyPlayer, RandomPlayer, StockfishPlayer, material

DATA = Path("data/positions")


def lesson(row):
    """(position after Stockfish's move, position after a random other move), both from the mover's side."""
    fen, move = row
    board = chess.Board(fen)
    others = [m for m in board.legal_moves if m.uci() != move]
    if not others:
        return None
    me, out = board.turn, []
    for m in (chess.Move.from_uci(move), random.choice(others)):
        board.push(m)
        out.append(np.packbits(planes(board, me).astype(bool).ravel()))
        board.pop()
    return out


def lessons(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_parquet(sorted(DATA.glob("shard_*.parquet"))[0], columns=["fen", "move"]).head(n)
    with Pool(os.cpu_count()) as pool:
        pairs = [p for p in pool.imap(lesson, rows.itertuples(index=False, name=None), chunksize=1000) if p]
    good, bad = zip(*pairs)
    return np.stack(good), np.stack(bad)


def unpack(bits):
    return np.unpackbits(bits, axis=1).astype(np.float32)


def top1(player: BioPlayer, fens: list[str], moves: list[str]) -> float:
    """How often the fly's favourite of all legal moves is Stockfish's move."""
    return float(np.mean([player.policy(chess.Board(f))[0][0].uci() == m for f, m in zip(fens, moves)]))


def train_tutor(fly: BioPlayer, args, log, weights: Path) -> None:
    good, bad = lessons(args.lessons)
    hold = 1024
    val = (unpack(good[:hold]), unpack(bad[:hold]))
    if not fly.trained:
        fly.fly.calibrate(unpack(good[hold:hold + 256]))                   # homeostasis, measured on real positions
    check = pd.read_parquet(sorted(DATA.glob("shard_*.parquet"))[0], columns=["fen", "move"]).head(300)
    print(f"{len(good)} lessons ready")

    def measure():
        """Held-out pairs: how often the position after Stockfish's move scores higher (a tie counts half)."""
        s = np.concatenate([fly.fly.score(fly.fly.run(v)) for v in (val[0][i:i + 256] for i in range(0, hold, 256))])
        t = np.concatenate([fly.fly.score(fly.fly.run(v)) for v in (val[1][i:i + 256] for i in range(0, hold, 256))])
        return float((s > t).mean() + 0.5 * (s == t).mean())

    log({"lessons": 0, "minutes": 0, "val_pair_acc": measure(), "top1": top1(fly, check.fen, check.move)})
    active, last, done, recent = 0.0, time.time(), 0, []
    while active < args.minutes * 60:
        i = np.random.randint(hold, len(good) - args.batch)
        recent.append(fly.fly.tutor(unpack(good[i:i + args.batch]), unpack(bad[i:i + args.batch])))
        done += args.batch
        now = time.time()
        active += min(now - last, 30.0)                                 # a pause by tools/heat_guard.py isn't training time
        last = now
        if done % (args.batch * 100) == 0:
            rec = {"lessons": done, "minutes": round(active / 60, 2), "pair_acc": round(np.mean(recent), 4),
                   "val_pair_acc": measure()}
            if done % (args.batch * 500) == 0:
                rec["top1"] = top1(fly, check.fen, check.move)
            recent = []
            log(rec)
            fly.fly.save(weights)
            last = time.time()                                          # measuring isn't training time either
    log({"lessons": done, "minutes": round(active / 60, 2), "val_pair_acc": measure(),
         "top1": top1(fly, check.fen, check.move)})
    fly.fly.save(weights)


def train_games(fly: BioPlayer, args, log, weights: Path) -> None:
    ladder = [RandomPlayer(), GreedyPlayer()]
    if shutil.which(os.environ.get("STOCKFISH", "stockfish")):
        ladder.append(StockfishPlayer(elo=None, skill=0))
    level, recent = 0, deque(maxlen=50)
    for game in range(1, args.games + 1):
        opp = ladder[level]
        me = chess.WHITE if game % 2 else chess.BLACK
        board = chess.Board()
        fly.fly.reset_trace()
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= 150:
            if board.turn != me:
                board.push(opp.move(board))
                continue
            before = material(board, me)
            board.push(fly.move(board))
            if board.is_game_over():
                break
            board.push(opp.move(board))
            reward = 0.1 * (material(board, me) - before)
            reward -= 0.05 if board.is_check() else 0
            fly.fly.dopamine(reward)
        score = {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)
        score = score if me == chess.WHITE else 1 - score
        fly.fly.dopamine({1.0: 1.0, 0.0: -1.0}.get(score, 0.0))
        recent.append(score)
        log({"game": game, "opponent": opp.name, "score": score, "moves": board.fullmove_number,
             "recent_score": sum(recent) / len(recent), "level": level})
        if game % 10 == 0:
            fly.fly.save(weights)
        if len(recent) == recent.maxlen and sum(recent) / len(recent) > 0.6 and level + 1 < len(ladder):
            level += 1
            recent.clear()
            print(f"moving up to {ladder[level].name}")
    fly.fly.save(weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20, help="tutor mode: how long to train")
    ap.add_argument("--lessons", type=int, default=400_000, help="tutor mode: positions to prepare")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--games", type=int, default=0, help="learn from this many games instead of a tutor")
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--eye", choices=["ranks", "fair"], default="ranks", help="how the eye is cut into squares (io_maps.py)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or ("bio-shuffled" if args.shuffle_wiring else "bio")
    run, ckpt = Path("runs") / name, Path("checkpoints") / name
    run.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    weights = ckpt / "kc_mbon.npz"
    fly = BioPlayer(str(weights), c=connectome.load(shuffle=args.shuffle_wiring), learn=True, eye=args.eye)
    out = open(run / "metrics.jsonl", "a")

    def log(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec)

    (train_games if args.games else train_tutor)(fly, args, log, weights)


if __name__ == "__main__":
    main()

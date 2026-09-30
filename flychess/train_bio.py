"""Train the biological fly: play games, pay it in dopamine.

  python -m flychess.train_bio --games 5000

After every one of the fly's moves the opponent replies, and the fly gets a small
dose of dopamine for what happened in between: material won or lost, giving or
taking check. At the end of the game it gets a big dose for the result. A win or
a loss alone would arrive once every 80 moves or so, far too rarely to learn from.

Opponents get harder as it improves: random, then greedy, then Stockfish.
"""
import argparse
import json
import os
import shutil
from collections import deque
from pathlib import Path

import chess

from . import connectome
from .play import BioPlayer, GreedyPlayer, RandomPlayer, StockfishPlayer, material


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or ("bio-shuffled" if args.shuffle_wiring else "bio")
    run, ckpt = Path("runs") / name, Path("checkpoints") / name
    run.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    weights = ckpt / "kc_mbon.npy"

    fly = BioPlayer(str(weights), c=connectome.load(shuffle=args.shuffle_wiring), learn=True)
    ladder = [RandomPlayer(), GreedyPlayer()]
    if shutil.which(os.environ.get("STOCKFISH", "stockfish")):
        ladder.append(StockfishPlayer(elo=None, skill=0))
    level, recent = 0, deque(maxlen=50)
    log = open(run / "metrics.jsonl", "a")

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
        result = board.result(claim_draw=True)
        score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        score = score if me == chess.WHITE else 1 - score
        fly.fly.dopamine({1.0: 1.0, 0.0: -1.0}.get(score, 0.0))
        recent.append(score)

        rec = {"game": game, "opponent": opp.name, "score": score, "moves": board.fullmove_number,
               "recent_score": sum(recent) / len(recent), "level": level}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if game % 10 == 0:
            print(rec)
            fly.fly.save(weights)
        if len(recent) == recent.maxlen and rec["recent_score"] > 0.6 and level + 1 < len(ladder):
            level += 1
            recent.clear()
            print(f"moving up to {ladder[level].name}")
    fly.fly.save(weights)


if __name__ == "__main__":
    main()

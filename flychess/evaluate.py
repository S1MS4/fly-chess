"""How strong is the fly, really?

  python -m flychess.evaluate --fly dl --games 40
  python -m flychess.evaluate --fly bio --games 40

Plays both colours against random, greedy and Stockfish at fixed Elo levels,
then turns the score against Stockfish into a performance rating:

    rating = opponent_elo + 400 * log10(score / (1 - score))

Scoring 50% against Stockfish-1320 means about 1320. Scoring 10% means about 940.
Below Stockfish's lowest setting the number gets shaky, so it is reported with
the raw scores next to it. Run the same thing for the shuffled-wiring brain
(--checkpoint checkpoints/dl-shuffled) and for lesions (--lesion "optic lobe")
to see what the real wiring is worth.
"""
import argparse
import json
import math
import os
import shutil

from . import connectome
from .play import BioPlayer, DLFly, GreedyPlayer, RandomPlayer, StockfishPlayer, play_game


def match(fly, opp, games: int) -> dict:
    """The fly's score over `games` games, half of them with each colour."""
    res = [play_game(fly, opp) if g % 2 == 0 else 1 - play_game(opp, fly) for g in range(games)]
    return {"score": sum(res) / games, "won": res.count(1.0), "drawn": res.count(0.5), "lost": res.count(0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", choices=["dl", "bio"], default="dl")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--lesion", nargs="*", default=[])
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--elos", nargs="*", type=int, default=[1320, 1500])
    args = ap.parse_args()

    c = connectome.load(shuffle=args.shuffle_wiring, lesion=args.lesion)
    if args.fly == "dl":
        fly = DLFly(args.checkpoint or "checkpoints/dl", c=c)
    else:
        fly = BioPlayer(args.checkpoint or "checkpoints/bio/kc_mbon.npz", c=c)

    results = {"random": match(fly, RandomPlayer(), args.games),
               "greedy": match(fly, GreedyPlayer(), args.games)}
    ratings = []
    if shutil.which(os.environ.get("STOCKFISH", "stockfish")):
        for elo in args.elos:
            sf = StockfishPlayer(elo=elo)
            results[f"stockfish-{elo}"] = match(fly, sf, args.games)
            sf.close()
            s = results[f"stockfish-{elo}"]["score"]
            s = min(max(s, 0.5 / args.games), 1 - 0.5 / args.games)  # avoid log(0)
            ratings.append(elo + 400 * math.log10(s / (1 - s)))
    else:
        print("stockfish not found, skipping the Elo part (set STOCKFISH=/path/to/stockfish)")
    if ratings:
        results["estimated_elo"] = round(sum(ratings) / len(ratings))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

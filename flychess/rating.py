"""An Elo for every fly, anchored at Stockfish's weakest setting.

Stockfish's UCI_Elo 1320 is the only calibrated rating this low (the Stockfish team anchors it to
the CCRL rating list), and every fly loses every game to it. So the ladder steps down from there:
Stockfish 1320 that plays a random legal move 25%, 50% or 75% of the time, a greedy mover and a
random mover. The rungs play their neighbours, every fly plays every rung, and one maximum-likelihood
fit places everybody at once, Stockfish 1320 held at 1320:

    P(a beats b) = 1 / (1 + 10^((Rb - Ra) / 400)),    a draw counts as half a win

  python -m flychess.rating --games 20                  # every fly, about an hour on my laptop
  python -m flychess.rating --games 20 --flies dl bio   # just these two
  python -m flychess.rating --fit-only                  # refit the games already played
  python -m flychess.rating --flies quick --out runs/part.json   # another process, then:
  python -m flychess.rating --merge runs/part.json --fit-only    # fold its games back in
  python -m flychess.rating --flies day --fresh                  # play a fly again after more training

Writes app/static/ratings.json: every game's result, and each player's rating with a 95% interval
from resampling the games (a bootstrap). The interface shows the flies' ratings from it.
"""
import argparse
import gc
import json
import math
import os
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import connectome
from .play import GreedyPlayer, MixedPlayer, RandomPlayer, StockfishPlayer, make_fly, play_game

OUT = Path("app/static/ratings.json")
ANCHOR, ANCHOR_ELO = "stockfish-1320", 1320
RUNGS = {"stockfish-1320-25": 0.25, "stockfish-1320-50": 0.5, "stockfish-1320-75": 0.75}
LADDER = [(ANCHOR, "stockfish-1320-25"), ("stockfish-1320-25", "stockfish-1320-50"),
          ("stockfish-1320-50", "stockfish-1320-75"), ("stockfish-1320-75", "random"),
          ("greedy", "random"), ("greedy", "stockfish-1320-75"), ("greedy", "stockfish-1320-50")]
FLY_OPPONENTS = ["random", "greedy", "stockfish-1320-75", "stockfish-1320-50", "stockfish-1320-25", ANCHOR]


def opponent(name: str, seed: int):
    if name == "random":
        return RandomPlayer()
    if name == "greedy":
        return GreedyPlayer()
    if name == ANCHOR:
        return StockfishPlayer(elo=1320)
    return MixedPlayer(RUNGS[name], seed=seed)


def play_pair(a, b, name_a: str, name_b: str, games: int, seed: int) -> list[dict]:
    """`games` games between a and b, colours alternating. Each record: white, black, white's score, how it
    ended and the material (white's lead in pawns at the end, and at its biggest either way)."""
    out = []
    for g in range(games):
        random.seed(seed * 1000 + g)                                   # the random and greedy movers draw from here
        white, black = (a, b) if g % 2 == 0 else (b, a)
        names = (name_a, name_b) if g % 2 == 0 else (name_b, name_a)
        info = {}
        out.append({"white": names[0], "black": names[1], "score": play_game(white, black, info=info), **info})
    return out


def fit(games: list[dict], virtual: float = 1.0) -> dict[str, float]:
    """Maximum-likelihood Elo for everybody in `games`, with `virtual` drawn games added to every
    pairing so a 20-0 doesn't run off to infinity (the same trick as BayesElo's prior)."""
    players = sorted({g["white"] for g in games} | {g["black"] for g in games})
    if not players:
        return {}
    met = defaultdict(set)                                             # everybody needs a path of games to the anchor
    for g in games:
        met[g["white"]].add(g["black"])
        met[g["black"]].add(g["white"])
    linked, frontier = {ANCHOR}, [ANCHOR]
    while frontier:
        for q in met[frontier.pop()] - linked:
            linked.add(q)
            frontier.append(q)
    if missing := [p for p in players if p not in linked]:
        raise ValueError(f"no games linking {', '.join(missing)} to {ANCHOR}")
    idx = {p: i for i, p in enumerate(players)}
    score, count = defaultdict(float), defaultdict(float)              # per pairing (i < j): i's points, games
    for g in games:
        i, j, s = idx[g["white"]], idx[g["black"]], g["score"]
        if i > j:
            i, j, s = j, i, 1 - s
        score[i, j] += s
        count[i, j] += 1
    for key in list(count):
        score[key] += virtual / 2
        count[key] += virtual
    k = math.log(10) / 400
    r = np.zeros(len(players))
    free = [i for i in range(len(players)) if players[i] != ANCHOR]
    for _ in range(100):                                               # Newton's method on the log-likelihood
        grad, hess = np.zeros(len(players)), np.zeros((len(players), len(players)))
        for (i, j), n in count.items():
            e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            grad[i] += k * (score[i, j] - n * e)
            grad[j] -= k * (score[i, j] - n * e)
            w = k * k * n * e * (1 - e)
            hess[i, i] -= w
            hess[j, j] -= w
            hess[i, j] += w
            hess[j, i] += w
        step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        r[free] -= step
        if np.abs(step).max() < 1e-6:
            break
    r += ANCHOR_ELO - r[idx[ANCHOR]]
    return {p: float(r[i]) for p, i in idx.items()}


def bootstrap(games: list[dict], rounds: int = 300, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% interval for every rating: refit on the games resampled within each pairing."""
    rng = np.random.default_rng(seed)
    by_pair = defaultdict(list)
    for g in games:
        by_pair[frozenset((g["white"], g["black"]))].append(g)
    samples = defaultdict(list)
    for _ in range(rounds):
        resampled = [gs[i] for gs in by_pair.values() for i in rng.integers(0, len(gs), len(gs))]
        for p, elo in fit(resampled).items():
            samples[p].append(elo)
    return {p: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for p, v in samples.items()}


def summary(games: list[dict]) -> list[dict]:
    """Won-drawn-lost for every pairing, from the first player's side."""
    out = defaultdict(lambda: [0, 0, 0])
    for g in games:
        a, b, s = g["white"], g["black"], g["score"]
        key = (a, b) if (b, a) not in out else (b, a)
        if key != (a, b):
            s = 1 - s
        out[key][0 if s == 1 else 1 if s == 0.5 else 2] += 1
    return [{"a": a, "b": b, "won": w, "drawn": d, "lost": l} for (a, b), (w, d, l) in out.items()]


def save(games: list[dict], rounds: int, out: Path = OUT) -> dict:
    """Write every game and the fit. The games go to disk even when there's no fit yet, and the file is
    swapped in whole, so a page reading it (or a kill) never sees half of one."""
    try:
        ratings, ci = fit(games), bootstrap(games, rounds)
    except (ValueError, np.linalg.LinAlgError) as err:
        print(f"no ratings yet: {err}")
        ratings, ci = {}, {}
    data = {"anchor": {"player": ANCHOR, "elo": ANCHOR_ELO}, "date": time.strftime("%Y-%m-%d"),
            "ratings": {p: {"elo": round(e), "low": round(ci[p][0]), "high": round(ci[p][1])} for p, e in ratings.items()},
            "pairs": summary(games), "games": games}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, out)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games per pairing")
    ap.add_argument("--flies", nargs="*", default=["dl", "bio", "shuffled", "quick", "no-optic-lobe", "no-central-brain", "newborn"])
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--rounds", type=int, default=300, help="bootstrap resamples")
    ap.add_argument("--out", type=Path, default=OUT, help="where the games and ratings go (and where a rerun resumes from)")
    ap.add_argument("--merge", type=Path, nargs="*", default=[], help="add these files' games, for pairings not played here yet")
    ap.add_argument("--opponents", nargs="*", default=FLY_OPPONENTS, help="only these rungs for the flies")
    ap.add_argument("--no-ladder", action="store_true", help="skip the games between the rungs (no ratings without them)")
    ap.add_argument("--fresh", action="store_true", help="forget these flies' old games first (a fly that trained more)")
    args = ap.parse_args()
    games = json.loads(args.out.read_text())["games"] if args.out.exists() else []
    if args.fresh:
        games = [g for g in games if g["white"] not in args.flies and g["black"] not in args.flies]
    done = {frozenset((g["white"], g["black"])) for g in games}
    for f in args.merge:
        new = [g for g in json.loads(f.read_text())["games"] if frozenset((g["white"], g["black"])) not in done]
        games.extend(new)
        done = {frozenset((g["white"], g["black"])) for g in games}  # so a pairing in two files counts once
        print(f"{len(new)} games from {f}")

    def run(a, b, name_a, name_b, seed):
        t = time.time()
        new = play_pair(a, b, name_a, name_b, args.games, seed)
        games.extend(new)
        data = save(games, args.rounds, args.out)
        s = sum(g["score"] if g["white"] == name_a else 1 - g["score"] for g in new) / len(new)
        now = data["ratings"].get(name_a, {}).get("elo", "?")
        print(f"{name_a} vs {name_b}: {s:.0%} in {time.time() - t:.0f}s | {name_a} {now}", flush=True)

    if not args.fit_only:
        for seed, (a, b) in enumerate([] if args.no_ladder else LADDER):   # the rungs, without any fly
            if frozenset((a, b)) in done:
                continue
            pa, pb = opponent(a, seed), opponent(b, seed + 100)
            try:
                run(pa, pb, a, b, seed)
            finally:                                                   # an engine left open keeps a crashed run from exiting
                for p in (pa, pb):
                    getattr(p, "close", lambda: None)()
        c = connectome.load()
        for kind in args.flies:
            if all(frozenset((kind, o)) in done for o in args.opponents):
                continue
            fly = make_fly(kind, c)
            for o, name in enumerate(FLY_OPPONENTS):
                if name not in args.opponents or frozenset((kind, name)) in done:
                    continue
                seed = 1000 + zlib.crc32(kind.encode()) % 9000 * 10 + o   # from the fly's name: the same games on every rerun
                opp = opponent(name, seed)
                try:
                    run(fly, opp, kind, name, seed)
                finally:
                    getattr(opp, "close", lambda: None)()
            del fly
            tf.keras.backend.clear_session()                               # free the GPU for the next fly
            gc.collect()
    data = save(games, args.rounds, args.out)
    for p, r in sorted(data["ratings"].items(), key=lambda kv: -kv[1]["elo"]):
        print(f"{p:22s} {r['elo']:5d}  ({r['low']} to {r['high']})")


if __name__ == "__main__":
    main()

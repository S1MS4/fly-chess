"""Stream the Lichess evaluation database and keep N positions.

The file (database.lichess.org/lichess_db_eval.jsonl.zst) is over 20 GB, so it is
never saved. It is streamed, decompressed on the fly, and only the first N usable
positions are kept. It is not sorted, so the first N are a fair sample, and it
streams fast: about a million positions every 20 seconds. Each line looks like:

  {"fen": "...", "evals": [{"pvs": [{"cp": 31, "line": "e2e4 e7e5 ..."},
                                    {"cp": 25, "line": "d2d4 d7d5 ..."}], "depth": 36}]}

Every position keeps:
  move    Stockfish's best move (first move of the deepest analysis)
  value   how good the position is for the side to move, in [-1, 1]
  cands   up to 5 candidate moves Stockfish looked at (multi-PV), best first
  wins    the chance to win after each of them, 0..100, for the side to move

The candidates are what makes training fast: the fly learns how good several moves
are, not just which one won. A catch: the start of the file has about twice as many
endgames as the rest, so download_chessbench.py is the default source now.

Usage: python scripts/download_lichess.py --positions 3000000
"""
import argparse
import io
import json
import math
from pathlib import Path

import pandas as pd
import requests
import zstandard
from tqdm import tqdm

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUT = Path("data/positions")
SHARD = 1_000_000
MATE = 10_000  # a forced mate counts as a 100 pawn advantage


def centipawns(pv: dict, white_to_move: bool) -> int:
    """Score of one line for the side to move. Faster mates score higher."""
    cp = (MATE - abs(pv["mate"])) * (1 if pv["mate"] > 0 else -1) if "mate" in pv else pv["cp"]
    return cp if white_to_move else -cp


def win_percent(cp: float) -> float:
    """Lichess's chance-to-win curve: centipawns -> 0..100."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=3_000_000)
    ap.add_argument("--min-depth", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, shard, kept = [], 0, 0
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r.raw), encoding="utf-8")
        bar = tqdm(total=args.positions, unit="pos")
        for line in lines:
            d = json.loads(line)
            evals = [e for e in d["evals"] if e.get("depth", 0) >= args.min_depth and e["pvs"][0].get("line")]
            if not evals:
                continue
            fen = d["fen"] + " 0 1"  # the eval db leaves out the move counters
            white = fen.split()[1] == "w"
            deepest = max(evals, key=lambda e: e["depth"])
            widest = max(evals, key=lambda e: (len(e["pvs"]), e["depth"]))
            best = deepest["pvs"][0]
            move = best["line"].split()[0]
            cands = {pv["line"].split()[0]: centipawns(pv, white) for pv in widest["pvs"][:5] if pv.get("line")}
            cands[move] = max(max(cands.values()), centipawns(best, white))  # the deepest best move stays on top
            order = sorted(cands, key=lambda m: (m != move, -cands[m]))
            value = math.tanh(centipawns(best, white) / 400)
            rows.append((fen, move, value, " ".join(order), " ".join(f"{win_percent(cands[m]):.1f}" for m in order)))
            kept += 1
            bar.update(1)
            if len(rows) == SHARD or kept == args.positions:
                pd.DataFrame(rows, columns=["fen", "move", "value", "cands", "wins"]).to_parquet(OUT / f"shard_{shard:03d}.parquet")
                rows, shard = [], shard + 1
            if kept == args.positions:
                break
    print(f"kept {kept} positions in {shard} shards under {OUT}")


if __name__ == "__main__":
    main()

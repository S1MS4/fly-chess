"""Stream the Lichess evaluation database and keep N positions.

The file (database.lichess.org/lichess_db_eval.jsonl.zst) is tens of GB, so it is
never saved. It is streamed, decompressed on the fly, and only the first N usable
positions are kept. Each line looks like:

  {"fen": "...", "evals": [{"pvs": [{"cp": 31, "line": "e2e4 e7e5 ..."}], "depth": 36}]}

For every position this keeps the FEN, the best move (first move of the deepest
principal variation) and a value in [-1, 1] from the side to move's point of view.

Usage: python scripts/download_lichess.py --positions 10000000
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


def to_value(pv: dict, white_to_move: bool) -> float:
    """Centipawns or mate from white's view -> [-1, 1] from the side to move."""
    if "mate" in pv:
        v = 1.0 if pv["mate"] > 0 else -1.0
    else:
        v = math.tanh(pv["cp"] / 400.0)
    return v if white_to_move else -v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=10_000_000)
    ap.add_argument("--min-depth", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, shard, kept = [], 0, 0
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        reader = zstandard.ZstdDecompressor().stream_reader(r.raw)
        lines = io.TextIOWrapper(reader, encoding="utf-8")
        bar = tqdm(total=args.positions, unit="pos")
        for line in lines:
            d = json.loads(line)
            best = max(d["evals"], key=lambda e: e.get("depth", 0))
            if best.get("depth", 0) < args.min_depth or not best["pvs"][0].get("line"):
                continue
            pv = best["pvs"][0]
            fen = d["fen"] + " 0 1"  # the eval db leaves out the move counters
            rows.append((fen, pv["line"].split()[0], to_value(pv, fen.split()[1] == "w")))
            kept += 1
            bar.update(1)
            if len(rows) == SHARD or kept == args.positions:
                pd.DataFrame(rows, columns=["fen", "move", "value"]).to_parquet(OUT / f"shard_{shard:03d}.parquet")
                rows, shard = [], shard + 1
            if kept == args.positions:
                break
    print(f"kept {kept} positions in {shard} shards under {OUT}")


if __name__ == "__main__":
    main()

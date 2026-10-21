"""Download ChessBench positions: Stockfish's chance to win after every legal move.

ChessBench (Ruoss et al. 2024, "Grandmaster-level chess without search", DeepMind) ran
Stockfish 16 on positions from real Lichess games and wrote down, for every legal
move, the probability of winning after it. This copy groups them per position:
huggingface.co/datasets/prdev/chessbench-full-policy-value, about 514,000 positions
per 72 MB file.

It is the best free teacher I found. The soft target covers every legal move, not
just the 1 to 5 the Lichess database happens to have, and the positions come from
real games, so openings and middlegames show up as often as they do over a board.
Writes the same shards as download_lichess.py, so train_dl.py --prepare reads either.

Usage: python scripts/download_chessbench.py --files 6        # about 3 million positions
       python scripts/download_chessbench.py --start 6 --files 10 --out data/positions_more
                                                              # 10 more files somewhere else, keeping what's there
"""
import argparse
import time
from pathlib import Path

import msgpack
import pandas as pd
import requests
import urllib3
import zstandard
from tqdm import tqdm

URL = "https://huggingface.co/datasets/prdev/chessbench-full-policy-value/resolve/main/train-{:05d}-of-01024.msgpack.zst"
OUT = Path("data/positions")
TOP = 8  # moves kept per position; with TAU = 5 win% the rest would get almost no share anyway
WAIT = 30  # seconds before trying a broken download again, doubling every time: 30 s, 1, 2, 4, 8 min


def fetch(i: int) -> list[tuple]:
    rows = []
    with requests.get(URL.format(i), stream=True, timeout=60) as r:
        r.raise_for_status()
        for rec in msgpack.Unpacker(zstandard.ZstdDecompressor().stream_reader(r.raw), raw=False):
            moves = sorted(rec["moves"].items(), key=lambda kv: -kv[1]["win_prob"])[:TOP]
            wins = [100 * v["win_prob"] for _, v in moves]
            rows.append((rec["fen"], moves[0][0], 2 * wins[0] / 100 - 1,
                         " ".join(m for m, _ in moves), " ".join(f"{w:.1f}" for w in wins)))
    return rows


def download(i: int, out: Path, tries: int = 6) -> Path:
    """ChessBench file i as a shard of rows (fen, best move, value, Stockfish's best moves, their win%).
    A connection that breaks halfway (it happened 7 files into the day run) gets five more tries."""
    for t in range(tries):
        try:
            rows = fetch(i)
            break
        except (requests.RequestException, urllib3.exceptions.HTTPError, zstandard.ZstdError, OSError) as err:
            if t == tries - 1:
                raise
            print(f"file {i}: {type(err).__name__}, trying again in {WAIT * 2 ** t} s", flush=True)
            time.sleep(WAIT * 2 ** t)
    path = out / f"shard_{i:03d}.parquet"
    pd.DataFrame(rows, columns=["fen", "move", "value", "cands", "wins"]).to_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)                             # a stopped download never leaves half a shard
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--start", type=int, default=0, help="the first file to get")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.start == 0 and args.out == OUT:                            # a fresh download replaces what was there
        for old in OUT.glob("shard_*.parquet"):
            old.unlink()
    for i in tqdm(range(args.start, args.start + args.files), unit="file"):
        if not (args.out / f"shard_{i:03d}.parquet").exists():
            download(i, args.out)
    print("done, now run: python -m flychess.train_dl --prepare")


if __name__ == "__main__":
    main()

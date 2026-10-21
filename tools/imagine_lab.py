"""How much should the fly's one-move imagination trust its value readout over its move readout?

  python tools/imagine_lab.py --fly day          # about 6 minutes on the GPU; writes checkpoints/dl-day/best/imagine.json
  python tools/imagine_lab.py --rows-only        # just find the held-out positions (CPU, a minute, once)

With imagination (flychess.play.look_ahead) a move scores log(p) + weight x value: p from the move readout,
the value from the value readout looking at the board after the move. This tries a range of weights on the
4,096 HOLD positions, the ones no fly ever trains on, rebuilt exactly from the ChessBench shards (castling
rights and all), with every legal move's board run through the brain once. For each weight it counts how
often the chosen move is Stockfish's best (top-1) and how often it's within 1 win% of it (a good move), and
keeps the weight with the most good moves. Weight 0 is the fly without imagination, except that it still
takes a mate in one and skips a stalemate; infinite is the value readout alone.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess.encode import planes, policy
from flychess.play import FLIES, choose, drawn, make_fly
from flychess.train_dl import DATA, HOLD, PACKED, pack

OUT = Path("runs/imagine")
GRID = (0, 0.25, 0.5, 1, 2, 4, 8, 16, float("inf"))
BATCH = 256                         # boards per brain call (the GPU's sparse multiply allows 344)


def hold_rows() -> pd.DataFrame:
    """The HOLD positions with their FENs, in the order train_dl --prepare packed them (the same search as
    scripts/make_curriculum.held_out_fens). Found once, then kept in runs/imagine/hold.parquet."""
    cache = OUT / "hold.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob("shard_*.parquet"))])
    order = np.random.RandomState(0).choice(len(rows), size=len(rows), replace=False)   # .sample(frac=1, random_state=0)
    keep = []
    for i in order:
        row = tuple(rows.iloc[i])
        if pack(row) is not None:
            keep.append(row)
            if len(keep) == HOLD:
                break
    out = pd.DataFrame(keep, columns=rows.columns)
    packed = np.load(PACKED)["boards"][:HOLD]
    same = [np.array_equal(np.packbits(planes(chess.Board(f)).astype(bool).ravel()), packed[i]) for i, f in enumerate(out.fen)]
    if not all(same):                                                  # the order must match train_dl's, or HOLD isn't HOLD
        raise SystemExit(f"only {sum(same)} of {HOLD} boards match data/positions/packed.npz, not saving")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def brain_values(fly, boards: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([fly.values(np.stack(boards[i:i + BATCH])) for i in range(0, len(boards), BATCH)]) if boards else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default="day", choices=[k for k, v in FLIES.items() if "weights" not in v])
    ap.add_argument("--limit", type=int, default=HOLD, help="only the first N positions (a quick test)")
    ap.add_argument("--dry", action="store_true", help="print the table, don't save the weight")
    ap.add_argument("--rows-only", action="store_true")
    args = ap.parse_args()
    rows = hold_rows().head(args.limit)
    if args.rows_only:
        print(f"{len(rows)} held-out positions in {OUT / 'hold.parquet'}")
        return
    t0, fly = time.time(), make_fly(args.fly)

    # the move readout on every position, then the value readout on the board after every legal move
    boards = [chess.Board(f) for f in rows.fen]
    heads = [fly.model(np.stack([planes(b).ravel() for b in boards[i:i + BATCH]])) for i in range(0, len(boards), BATCH)]
    fl, tl, pl = (np.concatenate([h[k].numpy() for h in heads]) for k in range(3))
    seen, after, where = [], [], []
    for n, b in enumerate(boards):
        probs = policy(b, fl[n], tl[n], pl[n])
        ends = []
        for m, _ in probs:
            b.push(m)
            ends.append("mate" if b.is_checkmate() else "draw" if drawn(b) else None)
            if ends[-1] is None:
                after.append(planes(b).ravel())
                where.append((n, len(ends) - 1))
            b.pop()
        seen.append((probs, ends))
    v = brain_values(fly, after)
    q = [np.zeros(len(probs)) for probs, _ in seen]
    for (n, i), value in zip(where, v):
        q[n][i] = -value                                               # the opponent moves next: its value, turned
    for n, (_, ends) in enumerate(seen):
        q[n][[e == "mate" for e in ends]] = 1.0

    # how good the chosen move is, by Stockfish's win% for each of its best moves
    table = []
    for w in GRID:
        top1 = good = lost = 0.0
        for n, row in enumerate(rows.itertuples(index=False)):
            probs, ends = seen[n]
            m = probs[choose(np.array([p for _, p in probs]), q[n], ends, w)][0].uci()
            wins = dict(zip(row.cands.split(), map(float, row.wins.split())))
            best = max(wins.values())
            mine = wins.get(m, min(wins.values()))                     # not in Stockfish's list: worse than its worst listed
            top1 += m == row.move
            good += m in wins and mine >= best - 1
            lost += best - mine
        table.append({"weight": w, "top1": round(top1 / len(rows), 4), "good": round(good / len(rows), 4),
                      "win_pct_lost": round(lost / len(rows), 2)})
        print(f"weight {w:>5}: top-1 {100 * table[-1]['top1']:.1f}%, good moves {100 * table[-1]['good']:.1f}%, "
              f"win% lost {table[-1]['win_pct_lost']:.2f}", flush=True)
    pick = max(table, key=lambda r: (r["good"], r["top1"]))
    rec = {"weight": pick["weight"], "positions": len(rows), "boards_imagined": len(after),
           "minutes": round((time.time() - t0) / 60, 1), "time": time.strftime("%Y-%m-%d %H:%M"), "grid": table}
    print(f"best weight {pick['weight']} ({len(after):,} boards imagined, {rec['minutes']} min)", flush=True)
    if not args.dry:
        target = Path(FLIES[args.fly]["checkpoint"]) / "imagine.json"
        target.write_text(json.dumps(rec, indent=1))
        with open(OUT / f"{args.fly}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"saved {target}")


if __name__ == "__main__":
    main()

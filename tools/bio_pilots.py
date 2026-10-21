"""Short pilots for a better dopamine fly, all on the same footing as tools/bio_rules.py.

  python tools/bio_pilots.py           # about 30 minutes on the RTX 4060

Every pilot starts from the same untrained mushroom body, gets 32,000 tutor lessons (500 batches of 64)
and is scored every 100 batches on the same 512 held-out pairs, plus at the end on 200 positions:
how often its favourite of all the legal moves is Stockfish's (top-1), which is what decides games.

  baseline       what bio.py does: averaged doses, compartment valence, 5% of Kenyon cells
  sparse 2/3/10% how many Kenyon cells may fire (APL's inhibition, set stronger or weaker)
  two-way        dopamine also strengthens the synapses of Kenyon cells that were silent at that moment
                 (Berry et al. 2018, Cohn et al. 2015), up to twice their measured strength
  harder         after 16,000 lessons, the bad move is Stockfish's second best instead of a random one
  expectation    the dose comes from the fly's own dopamine neurons: the reward plus how differently they
                 fire for this position than usual, which includes what the MBONs tell them through the
                 real MBON -> DAN connections (Felsenberg et al. 2017, Li et al. 2020)
Writes runs/bio-pilots/results.json.
"""
import json
import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.bio import BioFly
from flychess.encode import planes
from flychess.play import BioPlayer, sample_boards
from flychess.train_bio import DATA

BATCH, VAL, BATCHES = 64, 512, 500
PILOTS = [{"name": "baseline"}, {"name": "sparse 2%", "sparse": 0.02}, {"name": "sparse 3%", "sparse": 0.03},
          {"name": "sparse 10%", "sparse": 0.10}, {"name": "two-way", "two_way": 0.02},
          {"name": "harder", "harder": 250}, {"name": "expectation", "expectation": 1.0}]


def pairs(rows: pd.DataFrame, harder: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(position after Stockfish's move, position after a random other move, or its second best)."""
    rng = np.random.default_rng(0)
    good, bad = [], []
    for fen, move, cands in rows.itertuples(index=False, name=None):
        board = chess.Board(fen)
        me = board.turn
        others = [m.uci() for m in board.legal_moves if m.uci() != move]
        second = [m for m in cands.split()[1:2] if m in others]
        if not others or (harder and not second):
            continue
        for m, out in ((move, good), (second[0] if harder else others[rng.integers(len(others))], bad)):
            board.push_uci(m)
            out.append(planes(board, me).ravel())
            board.pop()
    return np.stack(good).astype(np.float32), np.stack(bad).astype(np.float32)


def lesson(fly: BioFly, good, bad, pilot, dan_usual):
    """One batch of tutor lessons: bio.py's rule, or one of the pilots' changes to it."""
    n = len(good)
    rates = fly.run(np.concatenate([good, bad]))
    s = fly.score(rates)
    wrong = 1 / (1 + np.exp(np.clip((s[:n] - s[n:]) / fly.temp, -30, 30)))
    kc = fly.sparse_kc(rates).numpy()
    if "expectation" in pilot:                                          # the fly's own DANs, plus the reward
        dans = rates.numpy()[fly.dan_idx] - dan_usual[:, None]
        pam = np.maximum(0, 1 + pilot["expectation"] * dans[fly.dan_is_pam].mean(0))      # sugar after the good move
        ppl1 = np.maximum(0, 1 + pilot["expectation"] * dans[~fly.dan_is_pam].mean(0))    # a shock after the bad one
        sugar, shock = kc[:, :n] @ pam[:n] / n, kc[:, n:] @ ppl1[n:] / n
    else:
        sugar, shock = kc[:, :n] @ wrong / n, kc[:, n:] @ wrong / n
    dose = np.outer(sugar, fly.pam_reach) + np.outer(shock, fly.ppl1_reach)
    fly.weaken(dose)
    if "two_way" in pilot:                                               # silent cells' synapses get stronger instead
        silent = np.outer(1 - kc[:, :n].mean(1) / max(kc.max(), 1e-6), fly.pam_reach * float(wrong.mean())) \
            + np.outer(1 - kc[:, n:].mean(1) / max(kc.max(), 1e-6), fly.ppl1_reach * float(wrong.mean()))
        fly.w += pilot["two_way"] * fly.lr * fly.w0 * silent
        np.clip(fly.w, 0, 2 * fly.w0, out=fly.w)


def main() -> None:
    c = connectome.load()
    rows = pd.read_parquet(sorted(DATA.glob("shard_*.parquet"))[0], columns=["fen", "move", "cands"])
    val = pairs(rows.iloc[:VAL])
    train = pairs(rows.iloc[VAL + 300:VAL + 300 + BATCHES * BATCH + 4000])
    hard = pairs(rows.iloc[VAL + 300 + BATCHES * BATCH + 4000:VAL + 300 + 2 * BATCHES * BATCH + 8000], harder=True)
    hard_val = pairs(rows.iloc[:2 * VAL], harder=True)
    check = rows.iloc[VAL:VAL + 200]
    out = {"pilots": []}
    player = BioPlayer("untrained", c=c)
    fly = player.fly
    dan = np.concatenate([c.where("dopamine (PAM)"), c.where("dopamine (PPL1)")])
    fly.dan_idx, fly.dan_is_pam = dan, np.isin(dan, c.where("dopamine (PAM)"))
    for pilot in PILOTS:
        fly.w, fly.k = fly.w0.copy(), int(pilot.get("sparse", 0.05) * len(fly.kc))   # the same untrained fly every time
        fly.calibrate(train[0][:256])                                    # the cut changes what the MBONs usually get
        dan_usual = fly.run(sample_boards(64, seed=1)).numpy()[dan].mean(1)

        def pair_accuracy(v):
            s = np.concatenate([fly.score(fly.run(v[0][i:i + 128])) for i in range(0, len(v[0]), 128)])
            t = np.concatenate([fly.score(fly.run(v[1][i:i + 128])) for i in range(0, len(v[1]), 128)])
            return float((s > t).mean() + 0.5 * (s == t).mean())

        curve = [(0, pair_accuracy(val))]
        rng = np.random.default_rng(1)
        for k in range(1, BATCHES + 1):
            src = hard if pilot.get("harder") and k > pilot["harder"] else train
            i = rng.integers(0, len(src[0]) - BATCH)
            lesson(fly, src[0][i:i + BATCH], src[1][i:i + BATCH], pilot, dan_usual)
            if k % 100 == 0:
                curve.append((k, pair_accuracy(val)))
        top1 = float(np.mean([player.policy(chess.Board(f))[0][0].uci() == m for f, m in zip(check.fen, check.move)]))
        result = {**pilot, "curve": curve, "hard_pairs": pair_accuracy(hard_val), "top1": top1}
        print(pilot["name"], " ".join(f"{a:.3f}" for _, a in curve), f"| hard pairs {result['hard_pairs']:.3f} | top-1 {top1:.3f}",
              flush=True)
        out["pilots"].append(result)
        Path("runs/bio-pilots").mkdir(parents=True, exist_ok=True)
        Path("runs/bio-pilots/results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

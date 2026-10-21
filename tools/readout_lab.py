"""Readout experiments on cached brain activity: how much data the readout can use, which steps to read,
and what happens after step 12.

  python tools/readout_lab.py cache --eye fair --positions 1000000   # the brain once over 1M positions, about 45 min
  python tools/readout_lab.py curve --eye fair                       # readouts fitted on 25k ... 1M of them
  python tools/readout_lab.py steps --eye fair                       # read step 12, the mean of 9 to 12, or all four
  python tools/readout_lab.py beyond --eye fair                      # 20 steps: does the board keep growing after 12?

With the brain frozen, a position's descending neurons do the same thing every time it's shown, so the
brain only has to run once per position. Their rates (0 to 1) are kept as bytes, 1,314 per position and
step: 1.3 GB per million positions. After that, a readout trains in seconds per pass instead of hours,
and fits until the loss on the VAL positions stops falling, so two readouts can be compared without the
noise of a short run. Results go to runs/readout-lab/<eye>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.model import FlyChess
from flychess.train_dl import HOLD, PACKED, TRAIN0, judge, losses, unpack

OUT = Path("runs/readout-lab")
EARLY = 300_000                     # positions with steps 9 to 11 kept too (the rest only step 12)


def brain(eye: str, steps: int = 12) -> FlyChess:
    model = FlyChess(connectome.load(), steps=steps, eye=eye)
    model(np.zeros((1, 768), np.float32))
    return model


def run_steps(model, x, keep):
    """The brain for all its steps; the descending neurons' rates at the steps in `keep` (1-based), as bytes."""
    b = model.brain
    inp = b.inputs(x)
    W = tf.sparse.SparseTensor(b.indices, b.weights_now(), (b.n, b.n))
    rates, out = tf.zeros_like(inp), {}
    for t in range(b.steps):
        z = b.gain * (tf.sparse.sparse_dense_matmul(W, rates) + inp - b.mu[t]) * b.inv_sd[t] + b.bias
        rates = (1 - b.leak) * rates + b.leak * tf.nn.relu(tf.tanh(z))
        if t + 1 in keep:
            out[t + 1] = tf.cast(tf.round(tf.transpose(tf.gather(rates, model.motor)) * 255), tf.uint8)
    return out


def cache(args) -> None:
    d = np.load(PACKED)
    boards = d["boards"]
    rows = np.concatenate([np.arange(TRAIN0), np.arange(TRAIN0, TRAIN0 + args.positions)])   # held out first, then training
    folder = OUT / f"cache-{args.eye}"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "rows.npy", rows)
    model = brain(args.eye)
    model.brain.calibrate(unpack(boards[HOLD:HOLD + 256]))            # the same calibration train_dl makes
    n_dn = len(model.motor)
    s12 = np.lib.format.open_memmap(folder / "s12.npy", "w+", np.uint8, (len(rows), n_dn))
    early = np.lib.format.open_memmap(folder / "s9_11.npy", "w+", np.uint8, (3, TRAIN0 + EARLY, n_dn))
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        r = rows[i:i + args.batch]
        keep = (9, 10, 11, 12) if i < TRAIN0 + EARLY else (12,)
        out = run_steps(model, tf.constant(unpack(boards[r])), keep)
        s12[i:i + len(r)] = out[12].numpy()
        if 9 in out:
            m = min(len(r), TRAIN0 + EARLY - i)
            for k, s in enumerate((9, 10, 11)):
                early[k, i:i + m] = out[s].numpy()[:m]
        if (i // args.batch) % 500 == 0:
            done = i + len(r)
            print(f"{done:,} of {len(rows):,} positions, {done / (time.time() - t0):.0f} a second", flush=True)
    s12.flush()
    early.flush()
    print(f"cached in {folder}, {(time.time() - t0) / 60:.1f} minutes")


def fit(feats, targets, train_rows, val_rows, hold_rows, max_epochs: int = 40, patience: int = 3, batch: int = 1024,
        lr: float = 1e-3, seed: int = 0) -> dict:
    """A fresh readout (the same four Dense heads as FlyChess) fitted until the VAL loss stops falling.
    feats: rows (an int tensor) -> float32 (rows, features), on the GPU. targets: the packed arrays.
    Returns its numbers on the HOLD positions, at the epoch with the lowest VAL loss."""
    tf.keras.utils.set_random_seed(seed)
    heads = [tf.keras.layers.Dense(k, name=n) for k, n in ((64, "from"), (64, "to"), (5, "promo"))] + \
            [tf.keras.layers.Dense(1, activation="tanh", name="value")]
    out = lambda x: (heads[0](x), heads[1](x), heads[2](x), heads[3](x)[:, 0])
    out(feats(tf.constant([0])))
    variables = [v for h in heads for v in h.trainable_weights]
    opt = tf.keras.optimizers.Adam(lr)
    gpu = [tf.constant(a) for a in targets[:4]]                         # legal, cand_idx, cand_win, values
    pick = lambda r: [tf.gather(a, r) for a in gpu]

    @tf.function
    def step(r):
        with tf.GradientTape() as tape:
            policy, value, _ = losses(out(feats(r)), *pick(r))
            loss = policy + value
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))

    @tf.function
    def score(r):
        policy, _, logits = losses(out(feats(r)), *pick(r))
        return policy, tf.argmax(logits, 1)

    def evaluate(rows):
        picks, total = [], 0.0
        for i in range(0, len(rows), 2048):
            r = rows[i:i + 2048]
            policy, p = score(tf.constant(r))
            picks.append(p.numpy())
            total += float(policy) * len(r)
        return total / len(rows), np.concatenate(picks)

    rng = np.random.default_rng(seed)
    best, best_weights, since, epochs = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        order = rng.permutation(train_rows)
        for i in range(0, len(order) - batch + 1, batch):
            step(tf.constant(order[i:i + batch]))
        loss, _ = evaluate(val_rows)
        epochs = epoch + 1
        if loss < best - 1e-4:
            best, best_weights, since = loss, [w.numpy() for w in variables], 0
        else:
            since += 1
            if since >= patience:
                break
    for w, b in zip(variables, best_weights):
        w.assign(b)
    loss, picks = evaluate(hold_rows)
    exact, good = judge(picks, {"cand_idx": targets[1], "cand_win": targets[2], "best": targets[4]}, hold_rows)
    return {"top1": round(exact, 4), "good_moves": round(good, 4), "held_out_loss": round(loss, 4),
            "val_loss": round(float(best), 4), "epochs": epochs, "train_positions": int(len(train_rows))}


def load_cache(eye: str):
    folder = OUT / f"cache-{eye}"
    rows = np.load(folder / "rows.npy")
    d = np.load(PACKED)
    targets = tuple(d[k][:rows.max() + 1] for k in ("legal", "cand_idx", "cand_win", "values", "best"))
    return folder, rows, targets


def on_gpu(path: Path, rows=None) -> tf.Tensor:
    """A cached array, read into memory once and kept on the GPU as bytes."""
    a = np.load(path)
    return tf.constant(a if rows is None else a[rows])


def record(eye: str, key: str, value) -> None:
    f = OUT / f"{eye}.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[key] = value
    f.write_text(json.dumps(data, indent=1))


def curve(args) -> None:
    """The same readout fitted on more and more positions: is more data still worth it?"""
    folder, rows, targets = load_cache(args.eye)
    assert np.array_equal(rows, np.arange(len(rows)))                   # the cache's row i is packed row i
    s12 = on_gpu(folder / "s12.npy")
    feats = lambda r: tf.cast(tf.gather(s12, r), tf.float32) / 255
    results = []
    for n in args.sizes:
        if TRAIN0 + n > len(rows):
            break
        res = fit(feats, targets, np.arange(TRAIN0, TRAIN0 + n), np.arange(HOLD, TRAIN0), np.arange(HOLD))
        results.append(res)
        print(f"{n:>9,} positions: {res}", flush=True)
        record(args.eye, "curve", results)


def steps(args) -> None:
    """Read step 12 only (what the flies do), the mean of steps 9 to 12, or all four side by side."""
    folder, rows, targets = load_cache(args.eye)
    n = min(EARLY, len(rows) - TRAIN0)
    s12 = on_gpu(folder / "s12.npy", slice(0, TRAIN0 + n))
    early = on_gpu(folder / "s9_11.npy")
    f = lambda a, r: tf.cast(tf.gather(a, r), tf.float32) / 255
    variants = {"step 12": lambda r: f(s12, r),
                "mean of steps 9-12": lambda r: (f(early[0], r) + f(early[1], r) + f(early[2], r) + f(s12, r)) / 4,
                "steps 9-12 side by side": lambda r: tf.concat([f(early[0], r), f(early[1], r), f(early[2], r), f(s12, r)], 1)}
    out = {}
    for name, feats in variants.items():
        out[name] = fit(feats, targets, np.arange(TRAIN0, TRAIN0 + n), np.arange(HOLD, TRAIN0), np.arange(HOLD))
        print(f"{name}: {out[name]}", flush=True)
        record(args.eye, "steps", out)


def beyond(args) -> None:
    """Run the brain for 20 steps (homeostasis calibrated for all 20) and look at the board's signal per
    region and step, and at readouts of steps 12, 16 and 20 fitted on the same 100,000 positions."""
    from flychess.connectome import REGIONS
    d = np.load(PACKED)
    model = brain(args.eye, steps=20)
    model.brain.calibrate(unpack(d["boards"][HOLD:HOLD + 256]))
    c = connectome.load()
    b = model.brain
    x = tf.constant(unpack(d["boards"][5000:5256]))
    inp = b.inputs(x)
    W = tf.sparse.SparseTensor(b.indices, b.weights_now(), (b.n, b.n))
    rates, spread = tf.zeros_like(inp), []
    for t in range(b.steps):
        z = b.gain * (tf.sparse.sparse_dense_matmul(W, rates) + inp - b.mu[t]) * b.inv_sd[t] + b.bias
        rates = (1 - b.leak) * rates + b.leak * tf.nn.relu(tf.tanh(z))
        r = rates.numpy()
        spread.append({reg: float(r[c.region == reg].std(1).mean()) for reg in REGIONS if (c.region == reg).any()})
    record(args.eye, "beyond_spread", spread)
    print("descending spread per step:", [round(s["descending"], 3) for s in spread], flush=True)
    n = args.beyond_positions
    rows = np.concatenate([np.arange(TRAIN0), np.arange(TRAIN0, TRAIN0 + n)])
    feats = {s: np.zeros((len(rows), len(model.motor)), np.uint8) for s in (12, 16, 20)}
    boards = d["boards"]                                               # once: an .npz rereads the whole array on every d[...]
    for i in range(0, len(rows), 32):
        r = rows[i:i + 32]
        out = run_steps(model, tf.constant(unpack(boards[r])), (12, 16, 20))
        for s in feats:
            feats[s][i:i + len(r)] = out[s].numpy()
    targets = tuple(d[k][:rows.max() + 1] for k in ("legal", "cand_idx", "cand_win", "values", "best"))
    res = {}
    for s, a in feats.items():
        a = tf.constant(a)
        res[f"step {s}"] = fit(lambda r, a=a: tf.cast(tf.gather(a, r), tf.float32) / 255, targets, np.arange(TRAIN0, TRAIN0 + n),
                               np.arange(HOLD, TRAIN0), np.arange(HOLD))
        print(f"read at step {s}: {res[f'step {s}']}", flush=True)
    record(args.eye, "beyond_readout", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cache", "curve", "steps", "beyond"])
    ap.add_argument("--eye", default="fair", choices=["ranks", "fair"])
    ap.add_argument("--positions", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32, help="32 is the fastest per position on my GPU (tools/bench.py)")
    ap.add_argument("--sizes", type=int, nargs="*", default=[25_000, 50_000, 100_000, 200_000, 400_000, 1_000_000])
    ap.add_argument("--beyond-positions", type=int, default=100_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"cache": cache, "curve": curve, "steps": steps, "beyond": beyond}[args.what](args)


if __name__ == "__main__":
    main()

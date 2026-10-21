"""Does the readout get better if it sees the brain's output at each of steps 9 to 12 instead of their mean?

  python tools/steps_lab.py extract --positions 3000000     # the brain once over them, about 3 hours on the GPU
  python tools/steps_lab.py fit                             # both readouts on the same positions, about 2 hours

The day-long fly reads the mean of steps 9 to 12 of its 1,314 descending neurons, one readout layer, no hidden
ones. On 300,000 positions the four steps side by side did worse than their mean (tools/readout_lab.py steps:
2.932 against 2.906): four times the weights, not enough positions. With millions of positions that excuse is
gone, and the mean throws away how the output moves from step to step. So: run the day fly's brain (its own
calibration from runs/dl-day/cache/brain.npz) over the HOLD and VAL positions and the first N training
positions, keep every step's rates as bytes (4 x 1,314 per position, on D: by default), then fit a readout on
the mean and one on the four side by side, on exactly the same positions, the way train_cached fits the fly:
streamed blocks, the step size halved after a pass that doesn't help, the starting point counting as the first
best. Results in runs/steps-lab/result.json.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.model import FlyChess
from flychess.train_dl import HOLD, PACKED, TRAIN0, judge, losses, unpack

STEPS = (9, 10, 11, 12)
FIELDS = ("legal", "cand_idx", "cand_win", "values")
OUT = Path("runs/steps-lab")
BLOCK = 65_536                                                       # rows per block on the GPU: 344 MB of bytes


def extract(args) -> None:
    folder = Path(args.cache)
    folder.mkdir(parents=True, exist_ok=True)
    boards = np.load(PACKED)["boards"]
    n = TRAIN0 + args.positions
    model = FlyChess(connectome.load(), eye="fair", read=STEPS)
    model(np.zeros((1, 768), np.float32))
    b = np.load("runs/dl-day/cache/brain.npz")                         # the day fly's own calibration
    model.brain.mu.assign(b["mu"])
    model.brain.inv_sd.assign(b["inv_sd"])
    dn = len(model.motor)
    done_file = folder / "rates_done.json"
    done = json.loads(done_file.read_text())["rows"] if done_file.exists() else 0
    rates = np.lib.format.open_memmap(folder / "rates.npy", "r+" if done else "w+", np.uint8, (n, len(STEPS), dn))

    @tf.function(input_signature=[tf.TensorSpec((None, 768), tf.float32)])   # as train_cached.extract
    def forward(x):
        trace = model.brain(x, return_trace=True)
        return tf.cast(tf.round(tf.stack([model.descending(trace[s - 1]) for s in STEPS], 1) * 255), tf.uint8)

    t0, start = time.time(), done
    for i in range(done, n, args.batch):
        rates[i:i + args.batch] = forward(unpack(boards[i:i + args.batch])).numpy()
        if (i // args.batch) % 2000 == 0 or i + args.batch >= n:
            rates.flush()
            done_file.write_text(json.dumps({"rows": min(i + args.batch, n)}))
            speed = (min(i + args.batch, n) - start) / max(time.time() - t0, 1e-9)
            print(f"steps-lab rates: {min(i + args.batch, n):,} of {n:,} positions, {speed:.0f} a second", flush=True)


class Readout:
    """The fly's four readout heads on `width` inputs, as in flychess.model.FlyChess."""

    def __init__(self, width: int, lr: float):
        self.heads = [tf.keras.layers.Dense(64), tf.keras.layers.Dense(64), tf.keras.layers.Dense(5),
                      tf.keras.layers.Dense(1, activation="tanh")]
        for h in self.heads:
            h.build((None, width))
        self.variables = [w for h in self.heads for w in h.trainable_weights]
        self.opt = tf.keras.optimizers.Adam(lr)

    def outputs(self, x):
        return self.heads[0](x), self.heads[1](x), self.heads[2](x), self.heads[3](x)[:, 0]


def fit(args) -> None:
    rates = np.load(Path(args.cache) / "rates.npy", mmap_mode="r")
    done = json.loads((Path(args.cache) / "rates_done.json").read_text())["rows"]
    if done < len(rates):
        raise SystemExit(f"only {done:,} of {len(rates):,} positions extracted; run extract to the end first")
    d = {k: v[:len(rates)] for k, v in np.load(PACKED).items() if k in (*FIELDS, "best", "cand_idx", "cand_win")}
    flat = d["cand_win"][TRAIN0:].astype(np.float32).max(1) >= 99.5      # as train_cached: those come from the curriculum
    train = TRAIN0 + np.where(~flat)[0]
    targets = {k: tf.constant(d[k]) for k in FIELDS}
    hold, val = np.arange(HOLD), np.arange(HOLD, TRAIN0)
    held = tf.constant(np.asarray(rates[:TRAIN0]))
    results = {}
    for name, side in (("mean of steps 9-12", False), ("steps 9-12 side by side", True)):
        tf.keras.utils.set_random_seed(0)
        width = rates.shape[2] * (len(STEPS) if side else 1)
        r = Readout(width, args.lr)

        def features(block):                                           # bytes -> what this readout reads
            x = tf.cast(block, tf.float32) / 255
            return tf.reshape(x, (-1, width)) if side else tf.reduce_mean(x, 1)

        @tf.function(reduce_retracing=True, autograph=False)
        def step(block, rows, first):
            tt = [tf.gather(targets[k], rows + first) for k in FIELDS]
            with tf.GradientTape() as tape:
                policy, value, _ = losses(r.outputs(features(tf.gather(block, rows))), *tt)
                loss = policy + value
            r.opt.apply_gradients(zip(tape.gradient(loss, r.variables), r.variables))

        @tf.function(reduce_retracing=True, autograph=False)
        def score(rows):
            policy, _, logits = losses(r.outputs(features(tf.gather(held, rows))), *[tf.gather(targets[k], rows) for k in FIELDS])
            return policy, tf.argmax(logits, 1)

        def check(rows):
            p, k = score(tf.constant(rows))
            top, good = judge(k.numpy(), d, rows)
            return {"top1": round(top, 4), "good_moves": round(good, 4), "loss": round(float(p), 4)}

        rng = np.random.default_rng(0)
        starts = list(range(TRAIN0, len(rates), BLOCK))
        load = lambda s: tf.constant(np.asarray(rates[s:s + BLOCK]))
        best, best_w, since, t0 = check(val)["loss"], [w.numpy() for w in r.variables], 0, time.time()
        r.opt.learning_rate.assign(args.lr)
        with ThreadPoolExecutor(1) as pool:
            for epoch in range(1, args.max_epochs + 1):
                order = rng.permutation(starts)
                nxt = pool.submit(load, order[0])
                for j, s in enumerate(order):
                    block = nxt.result()
                    if j + 1 < len(order):
                        nxt = pool.submit(load, order[j + 1])          # the next block loads while this one trains
                    mine = rng.permutation(train[(train >= s) & (train < s + BLOCK)] - s)   # this block's rows, shuffled
                    for k in range(0, len(mine), args.batch):
                        step(block, tf.constant(mine[k:k + args.batch]), tf.constant(s))
                rec = check(val)
                lr = float(r.opt.learning_rate.numpy())
                print(f"{name}: pass {epoch}, lr {lr:.6f}, VAL {rec}, {(time.time() - t0) / 60:.1f} min", flush=True)
                if rec["loss"] < best - 1e-4:
                    best, best_w, since = rec["loss"], [w.numpy() for w in r.variables], 0
                else:
                    since += 1
                    if since >= args.patience:
                        break
                    for w, b in zip(r.variables, best_w):
                        w.assign(b)
                    r.opt.learning_rate.assign(lr / 2)
        for w, b in zip(r.variables, best_w):
            w.assign(b)
        results[name] = {"hold": check(hold), "best_val_loss": best, "passes": epoch, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{name}: HOLD {results[name]['hold']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "train_positions": int(len(train)), "results": results}
    (OUT / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["extract", "fit"])
    ap.add_argument("--positions", type=int, default=3_000_000, help="training positions, after the HOLD and VAL ones")
    ap.add_argument("--cache", default="/mnt/d/fly-chess/steps-lab", help="where the rates go (16 GB for 3M positions)")
    ap.add_argument("--batch", type=int, default=None, help="32 for extract (fastest per position), 8192 for fit")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    args.batch = args.batch or (32 if args.what == "extract" else 8192)
    {"extract": extract, "fit": fit}[args.what](args)


if __name__ == "__main__":
    main()

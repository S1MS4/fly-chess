"""The backprop fly the fast way: the brain runs once per position, then the readout learns from that.

  python -m flychess.train_cached --name dl-day --curriculum 0.1                    # 3 million positions, about 2.5 hours
  python -m flychess.train_cached --name dl-day --curriculum 0.1 --more-hours 18    # then more ChessBench files for 18 hours

The same fly, readout and loss as train_dl.py with the brain frozen (its default). The difference: with
the brain frozen, a position's descending neurons do exactly the same thing every time it's shown, so
running the brain again on every pass is wasted work. Here every position goes through the brain once,
the descending neurons' rates are kept as bytes (1,314 per position, 4 GB for the 3 million downloaded
positions), and the readout then goes over them as many times as it keeps getting better, in seconds per
pass instead of hours. On my laptop the readout fitted that way on 200,000 positions already beat the one
that streamed 527,000 positions in 25 minutes (tools/readout_lab.py, README cheat 9).

  A. calibrate the brain the way train_dl.py does, run it once over every downloaded position and the
     curriculum, fit the readout until the loss on the VAL positions stops falling, save the fly
  B. (--more-hours) download the next ChessBench files one at a time, run the brain over each, fit again on everything
     every --refit-hours and at the end (or at runs/<name>/stop); when the disk is full the oldest file makes room

Everything picks up where it was if you run it again. The fly is saved where flychess.play.DLFly can play
it: checkpoints/<name> and checkpoints/<name>/best. One line per pass goes to runs/<name>/metrics.jsonl.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from . import connectome
from .model import FlyChess
from .train_dl import CURRICULUM, HOLD, PACKED, TRAIN0, judge, losses, pack, remember, top1, unpack

FIELDS = ("legal", "cand_idx", "cand_win", "values")
CHUNK = 262_144                        # positions per block the fit streams through the GPU (344 MB of rates)
MORE = Path("data/positions_more")
MIN_FREE_GB = 25                       # when the disk gets this full, the oldest cached file makes room


def extract(model, boards, out: Path, batch: int = 32, every: int = 2000) -> np.ndarray:
    """The descending neurons at the last step for every board, as bytes, into out (a .npy). Picks up where a
    stopped run left off: out_done.json says how many rows are finished."""
    done_file = out.with_name(out.stem + "_done.json")
    n, dn = len(boards), len(model.motor)
    done = json.loads(done_file.read_text())["rows"] if done_file.exists() and out.exists() else 0
    if done >= n:
        return np.load(out, mmap_mode="r")
    feats = np.lib.format.open_memmap(out, "r+" if done else "w+", np.uint8, (n, dn))

    @tf.function(input_signature=[tf.TensorSpec((None, 768), tf.float32)])
    def forward(x):
        if model.read == (model.brain.steps,):
            return tf.cast(tf.round(model.descending(model.brain(x)) * 255), tf.uint8)
        return tf.cast(tf.round(model.read_out(model.brain(x, return_trace=True)) * 255), tf.uint8)

    t0, start = time.time(), done
    for i in range(done, n, batch):
        feats[i:i + batch] = forward(unpack(boards[i:i + batch])).numpy()
        if (i // batch) % every == 0 or i + batch >= n:
            feats.flush()
            done_file.write_text(json.dumps({"rows": min(i + batch, n)}))
            speed = (min(i + batch, n) - start) / max(time.time() - t0, 1e-9)
            print(f"{out.name}: {min(i + batch, n):,} of {n:,} positions, {speed:.0f} a second", flush=True)
    feats.flush()
    done_file.write_text(json.dumps({"rows": n}))
    return np.load(out, mmap_mode="r")


def prepare_more(i: int, cache: Path) -> Path:
    """ChessBench file i (downloaded if needed) packed like train_dl --prepare packs the first ones, shuffled.
    Returns the .npz with boards and targets; a separate process, so it runs while the GPU works."""
    out = cache / f"more_{i:03d}.npz"
    if not out.exists():
        shard = MORE / f"shard_{i:03d}.parquet"
        if not shard.exists():
            subprocess.run([sys.executable, "scripts/download_chessbench.py", "--start", str(i), "--files", "1",
                            "--out", str(MORE)], check=True)
        rows = pd.read_parquet(shard).sample(frac=1, random_state=i)
        os.nice(19)                                                    # 8 packers next to the GPU job took the CPU to 98.9 C
        with Pool(4) as pool:                                          # 4 at the lowest priority: done long before the GPU needs it
            packed = [p for p in pool.imap(pack, rows.itertuples(index=False, name=None), chunksize=2000) if p]
        boards, legal, cand_idx, cand_win, best, values = zip(*packed)
        np.savez(out.with_suffix(".tmp.npz"), boards=np.stack(boards), legal=np.stack(legal), cand_idx=np.stack(cand_idx),
                 cand_win=np.stack(cand_win), best=np.array(best, np.uint8), values=np.array(values, np.float16))
        out.with_suffix(".tmp.npz").replace(out)
        shard.unlink()                                                 # packed, the parquet isn't needed any more
    return out


class Fitter:
    """The readout, fitted on cached rates streamed through the GPU a block at a time (CHUNK positions, 344 MB):
    the next block loads in a thread while the GPU trains on this one, and a pass visits the blocks in a random
    order. The rows were shuffled before packing, so every block is a fair sample. Only the VAL positions and the
    curriculum stay on the GPU. (All 4 GB of rates on the GPU at once ran WSL out of memory on the way there.)"""

    def __init__(self, model, d, feats, cur, cur_feats, args, write):
        self.model, self.d, self.cur, self.args, self.write = model, d, cur, args, write
        keep = np.arange(len(d["best"])) >= TRAIN0                     # HOLD and VAL never train
        if args.curriculum and cur is not None:                        # flat 100% labels would argue with the curriculum's
            keep[TRAIN0:] = d["cand_win"][TRAIN0:].astype(np.float32).max(1) < 99.5
        self.feats, self.keep = feats, keep
        self.val = np.arange(HOLD, TRAIN0)
        self.val_f = tf.constant(np.asarray(feats[HOLD:TRAIN0]))
        self.val_t = {k: tf.constant(d[k][HOLD:TRAIN0]) for k in FIELDS}
        self.extra = int(round(args.batch * args.curriculum)) if cur is not None else 0
        if cur is not None:
            self.cur_feats, self.cur_t = tf.constant(np.asarray(cur_feats)), {k: tf.constant(cur[k]) for k in FIELDS}
            self.cur_train, self.cur_test = np.where(~cur["test"])[0], np.where(cur["test"])[0]
        self.heads = (model.from_head, model.to_head, model.promo_head, model.value_head)
        self.variables = model.readout_weights()
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.rng = np.random.default_rng(args.seed)
        self.passes = 0

    def outputs(self, x):
        return self.heads[0](x), self.heads[1](x), self.heads[2](x), self.heads[3](x)[:, 0]

    @staticmethod
    def read(f, r):
        return tf.cast(tf.gather(f, r), tf.float32) / 255

    @tf.function(reduce_retracing=True, autograph=False)              # no AutoGraph: plain ops, and it rereads files
    def step(self, f, t, r, rc):
        x, tt = self.read(f, r), [tf.gather(t[k], r) for k in FIELDS]
        if self.extra:
            x = tf.concat([x, self.read(self.cur_feats, rc)], 0)
            tt = [tf.concat([a, tf.gather(self.cur_t[k], rc)], 0) for a, k in zip(tt, FIELDS)]
        with tf.GradientTape() as tape:
            policy, value, _ = losses(self.outputs(x), *tt)
            loss = policy + value
        self.opt.apply_gradients(zip(tape.gradient(loss, self.variables), self.variables))

    @tf.function(reduce_retracing=True, autograph=False)
    def score(self, f, t, r):
        policy, _, logits = losses(self.outputs(self.read(f, r)), *[tf.gather(t[k], r) for k in FIELDS])
        return policy, tf.argmax(logits, 1)

    def evaluate(self, f, t, rows, data, judged):
        """Loss and top-1 on rows of f and t; judged: the same positions' row numbers in data."""
        total, picks = 0.0, []
        for i in range(0, len(rows), 4096):
            r = tf.constant(rows[i:i + 4096])
            p, k = self.score(f, t, r)
            total += float(p) * len(r)
            picks.append(k.numpy())
        exact, good = judge(np.concatenate(picks), data, judged)
        return {"top1": round(exact, 4), "good_moves": round(good, 4), "held_out_loss": round(total / len(rows), 4)}

    def checks(self):
        rec = self.evaluate(self.val_f, self.val_t, np.arange(len(self.val)), self.d, self.val)
        if self.cur is not None:
            rec.update({f"finish_{k}": v for k, v in
                        self.evaluate(self.cur_feats, self.cur_t, self.cur_test, self.cur, self.cur_test).items()})
        return rec

    def blocks(self, more: list[Path]) -> list:
        """Every block of a pass, in a random order: (where, first row, the rows in it that train)."""
        out = [("local", s, np.where(self.keep[s:s + CHUNK])[0]) for s in range(0, len(self.keep), CHUNK)]
        for p in more:
            n = len(np.load(p, mmap_mode="r"))
            out += [(p, s, np.arange(min(CHUNK, n - s))) for s in range(0, n, CHUNK)]
        return [out[i] for i in self.rng.permutation(len(out)) if len(out[i][2])]

    def load(self, block):
        """A block's rates and targets, from disk into memory (in a thread, while the GPU trains on the last one)."""
        where, s, rows = block
        if where == "local":
            return np.asarray(self.feats[s:s + CHUNK]), {k: self.d[k][s:s + CHUNK] for k in FIELDS}, rows
        t = np.load(where.with_name(where.stem + "_t.npz"))
        return np.asarray(np.load(where, mmap_mode="r")[s:s + CHUNK]), {k: t[k][s:s + CHUNK] for k in FIELDS}, rows

    def batches(self, f, t, rows):
        size = self.args.batch - self.extra
        for i in range(0, len(rows) - size + 1, size):
            rc = self.rng.choice(self.cur_train, self.extra) if self.extra else np.zeros(0, np.int64)
            self.step(f, t, tf.constant(rows[i:i + size]), tf.constant(rc))

    def epoch(self, more: list[Path]):
        """One pass over everything, a block at a time."""
        blocks = self.blocks(more)
        with ThreadPoolExecutor(1) as pool:
            nxt = pool.submit(self.load, blocks[0])
            for j in range(len(blocks)):
                f_np, t_np, rows = nxt.result()
                if j + 1 < len(blocks):
                    nxt = pool.submit(self.load, blocks[j + 1])
                f, t = tf.constant(f_np), {k: tf.constant(v) for k, v in t_np.items()}
                del f_np, t_np
                self.batches(f, t, self.rng.permutation(rows))
                del f, t

    def fit(self, more: list[Path], label: str) -> dict:
        """Passes until the VAL loss stops falling; keeps the best readout. A pass that doesn't beat the best goes
        back to the best readout and halves the step size: on 17.4M positions, starting from the last fit, the
        first pass helped and the next four at the same step size only bounced around it (2026-10-21). The readout
        it starts from counts as the first best, so a fit that finds nothing better keeps it instead of saving a
        worse one (the next cycle's first pass, on 26.2M positions, came out worse than its start)."""
        start = self.checks()
        print(f"the readout it starts from: VAL loss {start['held_out_loss']:.4f}", flush=True)
        best, best_w, since, t0 = start["held_out_loss"], [w.numpy() for w in self.variables], 0, time.time()
        if self.args.lessons_changed:                                  # it never saw these lessons: its VAL loss
            best = np.inf                                              # isn't the bar, the best pass is
        self.opt.learning_rate.assign(self.args.lr)
        positions = int(self.keep.sum()) + sum(len(np.load(p, mmap_mode="r")) for p in more)
        for _ in range(self.args.max_epochs):
            self.epoch(more)
            self.passes += 1
            rec = self.checks()
            lr = float(self.opt.learning_rate.numpy())
            self.write({"epoch": self.passes, "fit": label, "positions": int(positions), "minutes": round((time.time() - t0) / 60, 2),
                        "lr": lr, **rec})
            if rec["held_out_loss"] < best - 1e-4:
                best, best_w, since = rec["held_out_loss"], [w.numpy() for w in self.variables], 0
            else:
                since += 1
                if since >= self.args.patience:
                    break
                for w, b in zip(self.variables, best_w):
                    w.assign(b)
                self.opt.learning_rate.assign(lr / 2)
        for w, b in zip(self.variables, best_w):
            w.assign(b)
        return {"best_val_loss": round(float(best), 4), "positions": int(positions)}


def save(model, d, cur, ckpt_dir: Path, write, info: dict) -> None:
    """The fitted readout checked through the real model on the HOLD positions, then saved like any other fly."""
    held = dict(d, boards=np.load(PACKED)["boards"][:HOLD])            # only the held-out boards, for the real model
    final = top1(model, held, rows=slice(0, HOLD), batch=64)           # small batches: the curriculum's rates share the GPU
    if cur is not None:
        final.update({f"finish_{k}": v for k, v in top1(model, cur, np.where(cur["test"])[0], batch=64).items()})
    write({"epoch": "final", **info, **final})
    ckpt = tf.train.Checkpoint(model=model)
    for folder in (ckpt_dir, ckpt_dir / "best"):
        tf.train.CheckpointManager(ckpt, folder, max_to_keep=1).save()
    print(f"saved {ckpt_dir}", flush=True)


def free_gb(folder: Path) -> float:
    """Free space on the drive `folder` is really on (runs/<name>/cache can be a junction to another drive)."""
    return shutil.disk_usage(folder).free / 1e9


def size_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.iterdir() if f.is_file()) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dl-day")
    ap.add_argument("--eye", choices=["ranks", "fair"], default="fair")
    ap.add_argument("--calm-unreached", action="store_true", help="as in train_dl; no effect on a frozen brain")
    ap.add_argument("--read", type=int, nargs="+", default=[12], help="steps whose descending neurons the readout averages")
    ap.add_argument("--curriculum", type=float, default=0.0, help="share of every batch from data/curriculum")
    ap.add_argument("--batch", type=int, default=8192, help="big: a step costs about the same at 1,024 and at 8,192")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=4, help="passes without a better VAL loss before a fit stops")
    ap.add_argument("--more-hours", type=float, default=0.0, help="then download and add more ChessBench files for this long")
    ap.add_argument("--more-start", type=int, default=6, help="the first ChessBench file not yet downloaded")
    ap.add_argument("--refit-hours", type=float, default=6.0, help="fit and save again this often while adding files")
    ap.add_argument("--skip-first-fit", action="store_true", help="the saved readout already learned from every file here")
    ap.add_argument("--max-cache-gb", type=float, default=float("inf"), help="past this, the oldest file makes room too")
    ap.add_argument("--lessons-changed", action="store_true",
                    help="data/curriculum has new lessons: refit even if VAL alone doesn't get better (delete the curriculum cache first)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions and N curriculum ones (a quick test)")
    ap.add_argument("--prepare-more", type=int, default=None, help=argparse.SUPPRESS)   # used internally, in its own process
    # train_dl's fixed settings, so remember() can check a resumed run is the same fly
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--lesion", nargs="*", default=[])
    args = ap.parse_args()
    run, ckpt_dir, cache = Path("runs") / args.name, Path("checkpoints") / args.name, Path("runs") / args.name / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    if args.prepare_more is not None:
        prepare_more(args.prepare_more, cache)
        return
    tf.keras.utils.set_random_seed(args.seed)
    remember(run, ckpt_dir, args, fixed=("eye", "shuffle_wiring", "lesion", "read"))   # the cache depends on all four
    log = open(run / "metrics.jsonl", "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)

    # A. the brain, calibrated once like train_dl.py (a rerun loads the same calibration back), over everything
    c = connectome.load(shuffle=args.shuffle_wiring, lesion=args.lesion)
    d = {k: v[:args.limit] for k, v in np.load(PACKED).items()}
    model = FlyChess(c, eye=args.eye, read=tuple(args.read))
    model(np.zeros((1, 768), np.float32))
    brain_file = cache / "brain.npz"
    if brain_file.exists():
        b = np.load(brain_file)
        model.brain.mu.assign(b["mu"])
        model.brain.inv_sd.assign(b["inv_sd"])
    else:
        model.brain.calibrate(unpack(d["boards"][HOLD:HOLD + 256]), calm=args.calm_unreached)
        np.savez(brain_file, mu=model.brain.mu.numpy(), inv_sd=model.brain.inv_sd.numpy())
    t0 = time.time()
    feats = extract(model, d["boards"], cache / "positions.npy")
    cur = None
    if (CURRICULUM / "packed.npz").exists():
        cur = {k: v[:args.limit] for k, v in np.load(CURRICULUM / "packed.npz").items()}
    cur_feats = extract(model, cur["boards"], cache / "curriculum.npy") if cur is not None else None
    print(f"the brain went over everything downloaded in {(time.time() - t0) / 3600:.2f} hours", flush=True)
    del d["boards"]                                                    # 300 MB nobody needs now

    if (ckpt_dir / "best" / "checkpoint").exists():                    # a rerun starts from the readout it had
        tf.train.Checkpoint(model=model).restore(tf.train.latest_checkpoint(str(ckpt_dir / "best"))).expect_partial()
    fitter = Fitter(model, d, feats, cur, cur_feats, args, write)
    done_more = sorted(cache.glob("more_*_t.npz"), key=lambda p: int(p.stem.split("_")[1]))   # oldest first, past 999 too
    more = [p.with_name(p.name.replace("_t.npz", ".npy")) for p in done_more]
    if not (ckpt_dir / "best" / "checkpoint").exists() or (more and not args.skip_first_fit):
        save(model, d, cur, ckpt_dir, write, fitter.fit(more, "everything so far"))

    # B. more ChessBench files while there's time and disk; the next one downloads and packs while the GPU works
    if args.more_hours > 0:
        end, last_fit = time.time() + args.more_hours * 3600, time.time()
        i = max([args.more_start] + [int(p.stem.split("_")[1]) + 1 for p in more])
        helper = lambda k: subprocess.Popen([sys.executable, "-m", "flychess.train_cached", "--name", args.name,
                                             "--prepare-more", str(k)], start_new_session=True)   # its own group, see below
        nxt, failed = helper(i), 0
        while time.time() < end and not (run / "stop").exists():      # touch runs/<name>/stop: the last fit, then out
            while (free_gb(cache) < MIN_FREE_GB or size_gb(cache) > args.max_cache_gb) and len(more) > 1:
                old = more.pop(0)                                      # full: the oldest file makes room
                for f in (old, old.with_name(old.stem + "_t.npz"), old.with_name(old.stem + "_done.json")):
                    f.unlink(missing_ok=True)
                print(f"cache {size_gb(cache):.0f} GB, {free_gb(cache):.0f} GB free: dropped {old.stem}, the oldest file "
                      f"(fitted on already)", flush=True)
            if free_gb(cache) < MIN_FREE_GB:
                print(f"only {free_gb(cache):.0f} GB free and nothing left to drop, stopping here", flush=True)
                break
            if nxt.wait() != 0:                                        # after its own tries (see download_chessbench.py)
                failed += 1
                if failed == 5:
                    print("couldn't get 5 ChessBench files in a row, stopping here", flush=True)
                    break
                print(f"couldn't get ChessBench file {i}, going on with the next one", flush=True)
                i += 1
                nxt = helper(i)
                continue
            failed = 0
            packed = cache / f"more_{i:03d}.npz"
            nxt = helper(i + 1)
            data = np.load(packed)
            extract(model, data["boards"], cache / f"more_{i:03d}.npy")
            np.savez(cache / f"more_{i:03d}_t.npz", **{k: data[k] for k in (*FIELDS, "best")})
            packed.unlink()                                            # the rates and the targets are all a fit needs
            more.append(cache / f"more_{i:03d}.npy")
            print(f"{len(more)} more files, cache {size_gb(cache):.0f} GB, {free_gb(cache):.0f} GB free, "
                  f"{(end - time.time()) / 3600:.1f} hours left", flush=True)
            i += 1
            if time.time() - last_fit > args.refit_hours * 3600:
                save(model, d, cur, ckpt_dir, write, fitter.fit(more, f"{len(more)} more files"))
                last_fit = time.time()
        if nxt.poll() is None:                                         # a file still downloading: stop it and its packers
            os.killpg(nxt.pid, 9)
        save(model, d, cur, ckpt_dir, write, fitter.fit(more, f"{len(more)} more files, the last fit"))


if __name__ == "__main__":
    main()

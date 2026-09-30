"""Deep learning version: teach the connectome with backprop.

  python -m flychess.train_dl --prepare          # parquet shards -> packed arrays, once
  python -m flychess.train_dl                    # train
  python -m flychess.train_dl --shuffle-wiring   # same thing on a scrambled brain (the control)

Each training example is one Lichess position, the move Stockfish liked best, and
Stockfish's evaluation. The fly guesses, the loss says how wrong it was, and the
gradient says how to nudge every one of the millions of connection strengths.
"""
import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from . import connectome
from .encode import move_to_target, planes, score_moves
from .model import FlyChess

DATA = Path("data/positions")
PACKED = DATA / "packed.npz"


def prepare() -> None:
    """Turn FEN strings into bit-packed planes once, so training never touches python-chess."""
    fens, boards, targets, values = [], [], [], []
    for f in sorted(DATA.glob("shard_*.parquet")):
        df = pd.read_parquet(f)
        for fen, mv, v in tqdm(zip(df.fen, df.move, df.value), total=len(df), desc=f.name):
            b = chess.Board(fen)
            m = chess.Move.from_uci(mv)
            if m not in b.legal_moves:
                continue
            boards.append(np.packbits(planes(b).astype(bool).ravel()))
            targets.append(move_to_target(b, m))
            values.append(v)
            fens.append(fen)
    np.savez(PACKED, boards=np.stack(boards), targets=np.array(targets, np.int16),
             values=np.array(values, np.float32), fens=np.array(fens))
    print(f"packed {len(boards)} positions into {PACKED}")


def dataset(boards, targets, values, batch: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((boards, targets, values)).shuffle(200_000).batch(batch, drop_remainder=True)

    def unpack(b, t, v):
        bits = tf.bitwise.right_shift(tf.expand_dims(b, -1), tf.constant([7, 6, 5, 4, 3, 2, 1, 0], tf.uint8))
        x = tf.cast(tf.reshape(tf.bitwise.bitwise_and(bits, 1), (-1, 768)), tf.float32)
        return x, tf.cast(t, tf.int32), v

    return ds.map(unpack, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)


def legal_accuracy(model: FlyChess, fens, targets, n: int = 1000) -> float:
    """Top-1: how often the fly's favourite legal move is Stockfish's move."""
    hits = 0
    for i in range(0, n, 100):
        bs = [chess.Board(f) for f in fens[i:i + 100]]
        x = np.stack([planes(b).ravel() for b in bs])
        fl, tl, pl, _ = [h.numpy() for h in model(x)]
        for j, b in enumerate(bs):
            best = max(score_moves(b, fl[j], tl[j], pl[j]), key=lambda s: s[1])[0]
            hits += move_to_target(b, best) == tuple(targets[i + j])
    return hits / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=200_000, help="optimiser steps")
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--lesion", nargs="*", default=[], help='e.g. --lesion "optic lobe"')
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.prepare:
        prepare()
        return

    name = args.name or ("dl-shuffled" if args.shuffle_wiring else "dl")
    run, ckpt_dir = Path("runs") / name, Path("checkpoints") / name
    run.mkdir(parents=True, exist_ok=True)

    c = connectome.load(shuffle=args.shuffle_wiring, lesion=args.lesion)
    print(f"brain: {c.n} neurons, {len(c.pre)} connections")
    d = np.load(PACKED)
    hold = 5000
    ds = dataset(d["boards"][hold:], d["targets"][hold:], d["values"][hold:], args.batch).repeat()
    val_fens, val_targets = d["fens"][:hold], d["targets"][:hold]

    model = FlyChess(c)
    opt = tf.keras.optimizers.Adam(args.lr, clipnorm=1.0)
    ckpt = tf.train.Checkpoint(model=model, opt=opt, step=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"resumed from {manager.latest_checkpoint}")
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    tb = tf.summary.create_file_writer(str(run))

    @tf.function
    def train_step(x, t, v):
        with tf.GradientTape() as tape:
            fl, tl, pl, val = model(x)
            policy = ce(t[:, 0], fl) + ce(t[:, 1], tl) + 0.1 * ce(t[:, 2], pl)
            value = tf.reduce_mean(tf.square(val - v))
            loss = policy + value
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return policy, value

    log = open(run / "metrics.jsonl", "a")
    t0 = time.time()
    for x, t, v in ds:
        step = int(ckpt.step.assign_add(1))
        policy, value = train_step(x, t, v)
        if step % 50 == 0:
            rec = {"step": step, "policy_loss": float(policy), "value_loss": float(value),
                   "positions_per_s": 50 * args.batch / (time.time() - t0)}
            t0 = time.time()
            if step % 2000 == 0:
                rec["top1"] = legal_accuracy(model, val_fens, val_targets)
                manager.save()
            with tb.as_default():
                for k, val in rec.items():
                    tf.summary.scalar(k, val, step=step)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)
        if step >= args.steps:
            break
    manager.save()


if __name__ == "__main__":
    main()

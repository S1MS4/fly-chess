"""Deep learning version: teach the connectome with backprop.

  python -m flychess.train_dl --prepare           # parquet shards -> packed arrays, once (about a minute)
  python -m flychess.train_dl --minutes 60        # train for an hour in total, then stop
  python -m flychess.train_dl --shuffle-wiring    # the same on a scrambled brain (the control)

Each example is one position plus what Stockfish thought of it: the chance to win
after each of its best moves, and an evaluation. The fly gives every legal move a
probability, the loss says how far that is from Stockfish's opinion, and the
gradient says how to nudge every one of the 6 million connection strengths.

What makes it fit into half an hour on a laptop (all explained in the README):
  - homeostasis: every neuron's input is centred and scaled, so the board reaches the legs
  - the softmax only runs over legal moves, so nothing is spent learning the rules
  - the target is Stockfish's whole opinion (soft labels), not only its favourite move
  - the readout is fitted first, on the untouched brain (a linear probe), then keeps learning
    from new positions while the brain stays as measured (see LR below)
  - if you do train the brain: backprop walks back through the last 6 of the 12 steps,
    and the weight gradient is one XLA kernel
  - every part has its own learning rate, warmed up then cosine down to zero at the time limit
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import tensorflow as tf

from . import connectome
from .encode import PAD, legal_codes, planes
from .model import FlyChess

DATA = Path("data/positions")
PACKED = DATA / "packed.npz"
CURRICULUM = Path("data/curriculum")   # won positions labelled by how fast they win (scripts/make_curriculum.py)
FIELDS = ("boards", "legal", "cand_idx", "cand_win", "values")
MAX_MOVES = 96   # positions with more legal moves than this are skipped (about 1 in 1000)
CANDS = 8        # Stockfish's best moves kept per position
TAU = 2.5        # a move this many win% worse than the best gets 1/e of the best move's share
GOOD = 1.0       # a move within this many win% of Stockfish's best counts as a good move
HOLD = 4096      # the first positions are never trained on, they measure top-1 at the end
VAL = 2048       # the next ones aren't trained on either: the checks during a run, and which checkpoint was best
TRAIN0 = HOLD + VAL
# Adam moves every weight about this much per step, however small its gradient. On a laptop-sized
# run, nudging the 6.2 million connections made the fly worse at every learning rate I tried
# (1e-3 down to 3e-5), while the readout alone kept getting better and ran 2.3x faster. So by
# default the brain stays as measured and only the readout learns; set --lr-synapses and
# --lr-neurons (around 1e-5) to train the brain itself on a long cloud run.
LR = {"readout": 3e-4, "synapses": 0.0, "neurons": 0.0}


def pack(row):
    """One position -> board bits, legal move codes, and the chance to win after Stockfish's best moves."""
    fen, move, value, cands, wins = row
    board = chess.Board(fen)
    moves, codes = legal_codes(board)
    slot = {m.uci(): i for i, m in enumerate(moves)}
    cands = [(slot[m], float(w)) for m, w in zip(cands.split(), wins.split()) if m in slot][:CANDS]
    if len(codes) > MAX_MOVES or move not in slot or not cands:
        return None
    idx, win = np.full(CANDS, 255, np.uint8), np.full(CANDS, -1, np.float16)  # -1: no candidate in this slot
    idx[:len(cands)], win[:len(cands)] = zip(*cands)
    legal = np.full(MAX_MOVES, PAD, np.uint16)
    legal[:len(codes)] = codes
    return np.packbits(planes(board).astype(bool).ravel()), legal, idx, win, slot[move], value


def pack_with_split(row):
    """pack() for a curriculum row, which also says whether it's for training or testing and where it's from."""
    p = pack(row[:5])
    return None if p is None else (*p, row[5] == "test", row[6] == "endgames", row[6] == "stalemates")


def prepare() -> None:
    """Turn FEN strings into packed arrays once, on every CPU core, so training never touches python-chess.
    The rows get shuffled first: the source files come sorted, and the held-out positions must be a fair sample.
    Packs data/curriculum too, if scripts/make_curriculum.py made it."""
    shards = sorted(DATA.glob("shard_*.parquet"))
    if PACKED.exists() and all(PACKED.stat().st_mtime > f.stat().st_mtime for f in shards):
        print(f"{PACKED} is newer than every shard, keeping it")
    else:
        rows = pd.concat([pd.read_parquet(f) for f in shards]).sample(frac=1, random_state=0)
        with Pool(os.cpu_count()) as pool:
            packed = [p for p in pool.imap(pack, rows.itertuples(index=False, name=None), chunksize=2000) if p]
        boards, legal, cand_idx, cand_win, best, values = zip(*packed)
        np.savez(PACKED, boards=np.stack(boards), legal=np.stack(legal), cand_idx=np.stack(cand_idx),
                 cand_win=np.stack(cand_win), best=np.array(best, np.uint8), values=np.array(values, np.float16))
        print(f"packed {len(boards)} of {len(rows)} positions into {PACKED}")
    lessons = sorted(CURRICULUM.glob("*.parquet"))
    if lessons:
        rows = pd.concat([pd.read_parquet(f).assign(source=f.stem) for f in lessons]).sample(frac=1, random_state=0)
        cols = ["fen", "move", "value", "cands", "wins", "split", "source"]
        with Pool(os.cpu_count()) as pool:
            packed = [p for p in pool.imap(pack_with_split, rows[cols].itertuples(index=False, name=None), chunksize=500) if p]
        boards, legal, cand_idx, cand_win, best, values, test, endgame, stalemate = zip(*packed)
        np.savez(CURRICULUM / "packed.npz", boards=np.stack(boards), legal=np.stack(legal), cand_idx=np.stack(cand_idx),
                 cand_win=np.stack(cand_win), best=np.array(best, np.uint8), values=np.array(values, np.float16),
                 test=np.array(test), endgame=np.array(endgame), stalemate=np.array(stalemate))
        print(f"packed {len(boards)} curriculum positions ({sum(test)} for testing) into {CURRICULUM / 'packed.npz'}")


def unpack(bits: np.ndarray) -> np.ndarray:
    return np.unpackbits(bits, axis=1).astype(np.float32)               # 96 bytes -> 768 zeros and ones


@tf.autograph.experimental.do_not_convert                              # plain TensorFlow ops: AutoGraph has nothing to do here,
def move_logits(heads, legal):                                         # and it reads source from disk, which an edit mid-run breaks
    """Score of every legal move: from + to + promotion. Empty slots get -1e9, so softmax ignores them."""
    fl, tl, pl, _ = heads
    code = tf.cast(legal, tf.int32)
    empty = code == PAD
    code = tf.where(empty, 0, code)
    s = (tf.gather(fl, code // 64 % 64, batch_dims=1) + tf.gather(tl, code % 64, batch_dims=1)
         + tf.gather(pl, code // 4096, batch_dims=1))
    return tf.where(empty, -1e9, s)


@tf.autograph.experimental.do_not_convert
def target(cand_idx, cand_win):
    """Stockfish's opinion as probabilities over the legal moves: the best move gets the
    biggest share, a move TAU win% worse 1/e of it, moves it didn't list nothing."""
    win = tf.cast(cand_win, tf.float32)
    share = tf.where(win >= 0, tf.exp((win - tf.reduce_max(win, 1, keepdims=True)) / TAU), 0.0)
    share /= tf.reduce_sum(share, 1, keepdims=True)
    return tf.reduce_sum(tf.one_hot(cand_idx, MAX_MOVES) * share[..., None], 1)


@tf.autograph.experimental.do_not_convert
def losses(heads, legal, cand_idx, cand_win, value):
    logits = move_logits(heads, legal)
    policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target(cand_idx, cand_win), logits))  # -sum(target * log softmax)
    value_loss = tf.reduce_mean(tf.square(heads[3] - tf.cast(value, tf.float32)))
    return policy, value_loss, logits


def schedule(progress: float) -> float:
    """Warm up over the first 3% of the run, then a cosine from 1 down to zero."""
    if progress < 0.03:
        return progress / 0.03
    return 0.5 * (1 + math.cos(math.pi * min((progress - 0.03) / 0.97, 1.0)))


def judge(pick, d, rows) -> tuple[float, float]:
    """For each chosen legal-move slot: is it Stockfish's move (top-1), and is it within
    GOOD win% of it (good move)? Chess has lots of ties, so the second number is fairer."""
    pick = np.asarray(pick)[:, None]
    win = d["cand_win"][rows].astype(np.float32)
    picked = np.where(d["cand_idx"][rows] == pick, win, -1).max(1)    # win% of the pick, -1 if Stockfish didn't list it
    return float(np.mean(pick[:, 0] == d["best"][rows])), float(np.mean(picked >= win.max(1) - GOOD))


def top1(model, d, rows=slice(HOLD, TRAIN0), batch: int = 256) -> dict:
    """On positions it never trained on: how often the fly's favourite legal move is Stockfish's
    (or as good), and the policy loss there. During a run that's the VAL positions, at the end all HOLD ones.
    rows: a slice, or an array of row numbers."""
    idx = np.arange(rows.start, rows.stop) if isinstance(rows, slice) else np.asarray(rows)
    picks, loss = [], []
    for i in range(0, len(idx), batch):
        s = idx[i:i + batch]
        policy, _, logits = losses(model(unpack(d["boards"][s])), d["legal"][s], d["cand_idx"][s], d["cand_win"][s], d["values"][s])
        picks.append(tf.argmax(logits, 1).numpy())
        loss.append(float(policy) * len(s))
    exact, good = judge(np.concatenate(picks), d, idx)
    return {"top1": round(exact, 4), "good_moves": round(good, 4), "held_out_loss": round(float(np.sum(loss)) / len(idx), 4)}


def snapshot(model, path: Path, **info) -> None:
    """Only the readout's weights (176k numbers, 0.7 MB): enough to replay what the fly would have
    answered at this point of training, for the pictures of gradient descent (tools/descent.py)."""
    heads = {"from": model.from_head, "to": model.to_head, "promo": model.promo_head, "value": model.value_head}
    np.savez(path, **info, **{f"{h}_{k}": v.numpy() for h, layer in heads.items() for k, v in zip(("kernel", "bias"), layer.weights)})


def blind(d, n: int = 200_000, batch: int = 1024) -> dict:
    """Scores of a 'fly' that never looks at the board: one fixed score per from-square,
    to-square and promotion, fitted to the same targets. Legal moves alone already give
    it a head start, so only what a fly scores above this comes from reading the board."""
    scores = [tf.Variable(tf.zeros(k)) for k in (64, 64, 5)]
    opt = tf.keras.optimizers.Adam(0.05)
    tile = lambda b: [tf.tile(s[None], (b, 1)) for s in scores] + [tf.zeros(b)]

    @tf.function
    def step(legal, ci, cw):
        with tf.GradientTape() as tape:
            policy, _, _ = losses(tile(tf.shape(legal)[0]), legal, ci, cw, tf.zeros(tf.shape(legal)[0]))
        opt.apply_gradients(zip(tape.gradient(policy, scores), scores))

    for i in range(HOLD, HOLD + n - batch, batch):
        step(d["legal"][i:i + batch], d["cand_idx"][i:i + batch], d["cand_win"][i:i + batch])
    exact, good = judge(tf.argmax(move_logits(tile(HOLD), d["legal"][:HOLD]), 1).numpy(), d, slice(0, HOLD))
    return {"blind_top1": round(exact, 4), "blind_good_moves": round(good, 4)}


def fit_readout(model, d, n: int, batch: int = 256, epochs: int = 8, snapshots: Path | None = None) -> None:
    """Linear probe: run the untouched brain once over n positions, keep what the descending
    neurons did, and fit only the readout on that. The full training then starts from a
    readout that already makes sense, instead of from random numbers."""
    rows = slice(TRAIN0, TRAIN0 + n)
    dn = np.concatenate([model.descending(model.brain(unpack(d["boards"][rows][i:i + batch]))).numpy().astype(np.float16)
                         for i in range(0, n, batch)])                  # float16: 300k positions fit in 790 MB
    legal, ci, cw, v = (d[k][rows] for k in ("legal", "cand_idx", "cand_win", "values"))
    head_vars = model.readout_weights()
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(dn, legal, ci, cw, v):
        with tf.GradientTape() as tape:
            policy, value, _ = losses(model.heads(tf.cast(dn, tf.float32)), legal, ci, cw, v)
            loss = policy + value
        opt.apply_gradients(zip(tape.gradient(loss, head_vars), head_vars))
        return policy

    k = 0
    for epoch in range(epochs):
        order = np.random.permutation(n - n % batch).reshape(-1, batch)
        loss = []
        for b in order:
            if snapshots and (k < 100 and k % 5 == 0 or k % 50 == 0):  # densely while it changes most
                snapshot(model, snapshots / f"probe_{k:05d}.npz", probe_step=k, positions=k * batch)
            loss.append(float(step(dn[b], legal[b], ci[b], cw[b], v[b])))
            k += 1
        print(f"readout epoch {epoch + 1}: policy loss {np.mean(loss):.3f}")
    if snapshots:
        snapshot(model, snapshots / f"probe_{k:05d}.npz", probe_step=k, positions=k * batch)


def remember(run: Path, ckpt_dir: Path, args, fixed=("eye", "shuffle_wiring", "lesion")) -> None:
    """Write the run's settings next to its metrics. A checkpoint only makes sense with the eye and the
    wiring it learned with, so resuming it with different ones stops here instead of quietly going wrong."""
    f = run / "config.json"
    now = {k: v for k, v in vars(args).items() if k != "prepare"}
    if f.exists() and (ckpt_dir / "checkpoint").exists():
        before = json.loads(f.read_text())
        if changed := [k for k in fixed if before.get(k, "ranks" if k == "eye" else None) != now.get(k)]:
            raise SystemExit(f"{ckpt_dir} was trained with other {', '.join(changed)}: {before}. Use another --name.")
    f.write_text(json.dumps(now, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--minutes", type=float, default=60, help="total training time for this run")
    ap.add_argument("--batch", type=int, default=128)
    for group, lr in LR.items():
        ap.add_argument(f"--lr-{group}", type=float, default=lr, help=f"learning rate of the {group} (0 freezes them)")
    ap.add_argument("--eval-every", type=int, default=500, help="steps between top-1 checks")
    ap.add_argument("--window", type=int, default=6, help="backprop through the last N of the 12 steps")
    ap.add_argument("--probe", type=int, default=50_000, help="positions for the readout warm start, 0 to skip")
    ap.add_argument("--shuffle-wiring", action="store_true")
    ap.add_argument("--lesion", nargs="*", default=[], help='e.g. --lesion "optic lobe"')
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="save the readout every N steps (and every 10 for the first 500) for tools/descent.py, 0 = never")
    ap.add_argument("--eye", choices=["ranks", "fair"], default="ranks", help="how the eye is cut into squares (io_maps.py)")
    ap.add_argument("--curriculum", type=float, default=0.0,
                    help="share of every batch from data/curriculum, the won positions ranked by how fast they win")
    ap.add_argument("--calm-unreached", action="store_true",
                    help="homeostasis gives no extra gain to neurons the board hasn't reached yet (model.calibrate)")
    ap.add_argument("--brain-opt", choices=["adam", "sgd"], default="adam",
                    help="optimiser for the synapses and neurons; sgd (momentum 0.9) moves each one by its gradient")
    ap.add_argument("--max-change", type=float, default=0.0,
                    help="keep every synapse within this share of its measured strength, e.g. 0.2; 0 = no leash")
    ap.add_argument("--seed", type=int, default=0, help="the same seed gives the same readout start and probe order")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    if args.prepare:
        prepare()
        return

    name = args.name or ("dl-shuffled" if args.shuffle_wiring else "dl")
    run, ckpt_dir = Path("runs") / name, Path("checkpoints") / name
    run.mkdir(parents=True, exist_ok=True)
    remember(run, ckpt_dir, args)

    c = connectome.load(shuffle=args.shuffle_wiring, lesion=args.lesion)
    print(f"brain: {c.n} neurons, {len(c.pre)} connections")
    d = dict(np.load(PACKED))
    n = len(d["best"])
    print(f"data: {n} positions")

    model = FlyChess(c, window=args.window, eye=args.eye)
    model(np.zeros((1, 768), np.float32))                              # build the layers
    lrs = {k: getattr(args, f"lr_{k}") for k in LR}
    groups = {k: vs for k, vs in {"readout": model.readout_weights(), "synapses": [model.brain.theta],
                                  "neurons": [model.brain.gain, model.brain.bias]}.items() if lrs[k] > 0}
    brain_opt = lambda lr: (tf.keras.optimizers.SGD(lr, momentum=0.9, global_clipnorm=1.0) if args.brain_opt == "sgd"
                            else tf.keras.optimizers.Adam(lr, global_clipnorm=1.0))
    opts = {k: tf.keras.optimizers.Adam(lrs[k], global_clipnorm=1.0) if k == "readout" else brain_opt(lrs[k]) for k in groups}
    ckpt = tf.train.Checkpoint(model=model, step=tf.Variable(0, dtype=tf.int64), seconds=tf.Variable(0.0, dtype=tf.float64),
                               best=tf.Variable(np.inf, dtype=tf.float64), **opts)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)
    best = tf.train.CheckpointManager(ckpt, ckpt_dir / "best", max_to_keep=1)   # the lowest loss on the VAL positions so far
    snaps = run / "snapshots"
    if args.snapshot_every:
        snaps.mkdir(exist_ok=True)
    leash = None
    if args.max_change and "synapses" in groups:                       # every synapse stays within +-max_change of its
        w0 = tf.nn.softplus(model.brain.theta).numpy()                 # measured strength (theta is still as measured here)
        inv = lambda w: np.log(np.expm1(np.maximum(w, 1e-6)))           # inverse softplus
        leash = (tf.constant(inv(w0 * (1 - args.max_change)), tf.float32), tf.constant(inv(w0 * (1 + args.max_change)), tf.float32))
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"resumed from {manager.latest_checkpoint} after {float(ckpt.seconds) / 60:.1f} minutes")
    else:
        model.brain.calibrate(unpack(d["boards"][HOLD:HOLD + 256]), calm=args.calm_unreached)   # once: input statistics, no targets
        if args.probe:
            fit_readout(model, d, args.probe, snapshots=snaps if args.snapshot_every else None)
    cur = None                                                         # if it exists, finishing gets measured either way
    if args.curriculum or (CURRICULUM / "packed.npz").exists():
        cur = dict(np.load(CURRICULUM / "packed.npz"))
        cur_train, cur_test = np.where(~cur["test"])[0], np.where(cur["test"])[0]
        print(f"curriculum: {len(cur_train)} positions to learn from, {len(cur_test)} to test finishing on")
    log = open(run / "metrics.jsonl", "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec)

    variables = [v for vs in groups.values() for v in vs]
    brain_learns = "synapses" in groups or "neurons" in groups

    @tf.function
    def train_step(x, legal, ci, cw, v):
        if not brain_learns:                                           # frozen brain: run it outside the tape,
            dn = model.descending(model.brain(x))                      # so nothing is kept for a backward pass
        with tf.GradientTape() as tape:
            heads = model(x, training=True) if brain_learns else model.heads(dn)
            policy, value, _ = losses(heads, legal, ci, cw, v)
            loss = policy + value
        grads = iter(tape.gradient(loss, variables))
        norms = []
        for k, vs in groups.items():
            g = [next(grads) for _ in vs]
            norms.append(tf.linalg.global_norm(g))                     # before clipping: how hard each group gets pushed
            opts[k].apply_gradients(zip(g, vs))
        if leash is not None:
            model.brain.theta.assign(tf.clip_by_value(model.brain.theta, *leash))
        return policy, value, tf.stack(norms)

    def checks():
        """The numbers logged every --eval-every steps: VAL positions, and how well it finishes won games."""
        out = top1(model, d)
        if cur is not None:
            out.update({f"finish_{k}": v for k, v in top1(model, cur, cur_test).items()})
        return out

    if int(ckpt.step) == 0:
        write({"step": 0, "positions": 0, "minutes": 0.0, **checks(), **blind(d)})
        if args.snapshot_every:
            snapshot(model, snaps / f"step_{0:07d}.npz", step=0, positions=0, minutes=0.0)
    budget, rng = args.minutes * 60, np.random.default_rng(int(ckpt.step))
    rows = np.arange(TRAIN0, n)
    if args.curriculum:                                                # the won positions' flat 100% labels would argue with
        rows = rows[d["cand_win"][TRAIN0:].astype(np.float32).max(1) < 99.5]   # the curriculum's, so it teaches those instead
    extra = int(round(args.batch * args.curriculum))
    order, cursor, window = rng.permutation(rows), 0, []
    t_last = t_log = t_saved = time.time()
    while float(ckpt.seconds) < budget:
        if cursor + args.batch - extra > len(order):
            order, cursor = rng.permutation(order), 0
        b = np.sort(order[cursor:cursor + args.batch - extra])         # sorted rows read faster
        cursor += args.batch - extra
        batch = [d[k][b] for k in FIELDS]
        if extra:
            cb = np.sort(rng.choice(cur_train, extra))
            batch = [np.concatenate([a, cur[k][cb]]) for a, k in zip(batch, FIELDS)]
        for k, opt in opts.items():
            opt.learning_rate.assign(lrs[k] * schedule(float(ckpt.seconds) / budget))
        window.append(train_step(unpack(batch[0]), *batch[1:]))
        step = int(ckpt.step.assign_add(1))
        now = time.time()
        ckpt.seconds.assign_add(min(now - t_last, 30.0))               # a step takes 0.4 s; longer means tools/heat_guard.py
        t_last = now                                                   # paused us, and a pause isn't training time
        if args.snapshot_every and (step <= 500 and step % 10 == 0 or step % args.snapshot_every == 0):
            snapshot(model, snaps / f"step_{step:07d}.npz", step=step, positions=step * args.batch,
                     minutes=float(ckpt.seconds) / 60)
        if step % 50 == 0:
            policy, value = np.mean([[float(p), float(v)] for p, v, _ in window], 0)
            norms = np.mean([g.numpy() for _, _, g in window], 0)
            rec = {"step": step, "positions": step * args.batch, "minutes": round(float(ckpt.seconds) / 60, 2),
                   "policy_loss": round(float(policy), 4), "value_loss": round(float(value), 4),
                   "positions_per_s": round(len(window) * args.batch / (now - t_log), 1),
                   **{f"grad_{k}": round(float(g), 5) for k, g in zip(groups, norms)}}
            window = []
            if step % args.eval_every == 0:
                rec.update(checks())
                if rec["held_out_loss"] < float(ckpt.best):
                    ckpt.best.assign(rec["held_out_loss"])
                    best.save()
            write(rec)
            t_last = t_log = time.time()                               # measuring and logging isn't training time
        if time.time() - t_saved > 600:                                # every 10 minutes
            manager.save()
            t_saved = time.time()
    final = top1(model, d, rows=slice(0, HOLD))
    if cur is not None:
        final.update({f"finish_{k}": v for k, v in top1(model, cur, cur_test).items()})
    write({"step": int(ckpt.step), "positions": int(ckpt.step) * args.batch, "minutes": round(float(ckpt.seconds) / 60, 2),
           **final, "best_val_loss": round(float(ckpt.best), 4)})
    manager.save()


if __name__ == "__main__":
    main()

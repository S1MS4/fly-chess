"""Gradient descent you can see, from the readout snapshots of a run with --snapshot-every.

  python -m flychess.train_dl --minutes 10 --name dl-pictures --snapshot-every 250   # in WSL, once
  python tools/descent.py --run dl-pictures                                         # then the pictures

  docs/img/learning_e4.gif   the fly's answer to 1.e4 at every snapshot, from the newborn's random readout
                             to the trained one: arrows as thick as its probability, its top 6 moves as bars
  docs/img/landscape.png     the loss on held-out positions around the path gradient descent took, in the
                             plane of the path's two biggest directions (Li et al. 2018), and a zoom on
                             the training after the probe
  docs/img/landscape_3d.gif  the same surface from the side, turning

The brain doesn't learn in these runs, so what the descending neurons do for a position never changes.
Run the brain once per position, and the fly at any snapshot is one small matrix multiply away. The move
scores are even linear in the readout's weights, so the loss anywhere in the plane is a sum of three
precomputed arrays and a softmax.
"""
import argparse
import io
import sys
from pathlib import Path

import chess
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess.encode import PAD, legal_codes, planes
from flychess.train_dl import HOLD, PACKED, TAU, unpack
from readme_figures import BLUE, INK, MUTED, ORANGE, OUT, SURFACE, draw_board   # same colours, same rcParams

PARAMS = ("from_kernel", "from_bias", "to_kernel", "to_bias", "promo_kernel", "promo_bias")   # the policy part


def snapshots(run: Path) -> list[dict]:
    """All snapshots in training order: the probe's (from the newborn's random readout), then training's."""
    files = sorted((run / "snapshots").glob("probe_*.npz")) + sorted((run / "snapshots").glob("step_*.npz"))[1:]
    out, probe_end = [], 0
    for f in files:
        d = dict(np.load(f))
        if "probe_step" in d:
            seen = int(d["positions"])
            probe_end = seen
            phase = "newborn, random readout" if seen == 0 else "readout probe: 50,000 positions, 8 times over"
        else:
            seen = probe_end + int(d["positions"])
            phase = "training on new positions, the brain as measured"
        out.append({"seen": seen, "phase": phase, **{k: d[k] for k in PARAMS}})
    return out


def features(run_name: str, boards: np.ndarray, cache: Path) -> np.ndarray:
    """What the descending neurons do at step 12 for each board, through the run's calibrated brain."""
    if cache.exists():
        return np.load(cache)
    import json
    import tensorflow as tf
    from flychess.play import DLFly
    config = Path("runs") / run_name / "config.json"
    fly = DLFly(f"checkpoints/{run_name}", eye=json.loads(config.read_text()).get("eye", "ranks") if config.exists() else "ranks")
    m = fly.model
    dn = np.concatenate([m.descending(m.brain(tf.constant(boards[i:i + 256]))).numpy() for i in range(0, len(boards), 256)])
    np.save(cache, dn)
    return dn


def heads(snap: dict, dn: np.ndarray) -> np.ndarray:
    """(positions, 133): 64 from-square scores, 64 to-square scores, 5 promotion scores."""
    k = np.concatenate([snap["from_kernel"], snap["to_kernel"], snap["promo_kernel"]], 1)
    b = np.concatenate([snap["from_bias"], snap["to_bias"], snap["promo_bias"]])
    return dn @ k + b


def move_scores(z: np.ndarray, legal: np.ndarray, empty_value: float = -1e9) -> np.ndarray:
    code = legal.astype(np.int64)
    empty = code == PAD
    code[empty] = 0
    s = (np.take_along_axis(z[:, :64], code // 64 % 64, 1) + np.take_along_axis(z[:, 64:128], code % 64, 1)
         + np.take_along_axis(z[:, 128:], code // 4096, 1))
    return np.where(empty, empty_value, s)


def targets(cand_idx: np.ndarray, cand_win: np.ndarray, width: int) -> np.ndarray:
    """Stockfish's opinion as shares of the legal moves, the same as train_dl.target()."""
    win = cand_win.astype(np.float64)
    share = np.where(win >= 0, np.exp((win - win.max(1, keepdims=True)) / TAU), 0.0)
    share /= share.sum(1, keepdims=True)
    t = np.zeros((len(win), width))
    for j in range(cand_idx.shape[1]):
        ok = cand_idx[:, j] < width
        t[np.where(ok)[0], cand_idx[ok, j]] += share[ok, j]
    return t


def policy_loss(scores: np.ndarray, target: np.ndarray) -> float:
    m = scores.max(1, keepdims=True)
    log_p = scores - m - np.log(np.exp(scores - m).sum(1, keepdims=True))
    return float(-(target * np.where(target > 0, log_p, 0)).sum(1).mean())


def learning_gif(snaps: list[dict], dn_e4: np.ndarray, out: Path) -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    moves, codes = legal_codes(board)
    legal = np.full((1, 96), PAD, np.uint16)
    legal[0, :len(codes)] = codes
    probs = []
    for s in snaps:
        sc = move_scores(heads(s, dn_e4), legal)[0, :len(moves)]
        p = np.exp(sc - sc.max())
        probs.append(p / p.sum())
    top_ever = max(float(p.max()) for p in probs)
    frames = []
    for s, p in zip(snaps, probs):
        order = np.argsort(-p)
        fig = plt.figure(figsize=(8.6, 4.6))
        ax = fig.add_axes((0.02, 0.06, 0.46, 0.8))
        draw_board(ax, board, [(moves[i], float(p[i])) for i in order[:8]])
        ax = fig.add_axes((0.62, 0.14, 0.34, 0.66))
        top = order[:6]
        ax.barh(range(6), [100 * p[i] for i in top], color=BLUE, height=0.62)
        for j, i in enumerate(top):
            ax.text(100 * p[i] + 0.4, j, f"{100 * p[i]:.1f}%", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(6), [board.san(moves[i]) for i in top], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 100 * top_ever * 1.25)
        ax.set_xlabel("probability")
        ax.grid(axis="y", visible=False)
        fig.text(0.02, 0.94, "The fly's answer to 1.e4 while it learns", fontsize=13, weight="bold", color=INK)
        fig.text(0.02, 0.885, f"{s['phase']}, {s['seen']:,} positions so far", fontsize=9.5, color=MUTED)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=90, facecolor=SURFACE, bbox_inches=None)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))
    durations = [700] + [90] * (len(frames) - 2) + [2500]                # hold the newborn and the end
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"saved {out}: {len(frames)} frames, newborn's favourite {board.san(moves[int(np.argmax(probs[0]))])} "
          f"{probs[0].max():.1%}, at the end {board.san(moves[int(np.argmax(probs[-1]))])} {probs[-1].max():.1%}")


def plane(snaps: list[dict], dn: np.ndarray, legal, target, steps: int = 61, margin: float = 0.25):
    """Li et al. 2018: PCA of the path's weights minus the final ones gives two directions; the loss on a grid
    in that plane, and the path projected onto it."""
    theta = np.stack([np.concatenate([s[k].ravel() for k in PARAMS]) for s in snaps])
    diff = theta - theta[-1]
    _, sv, vt = np.linalg.svd(diff, full_matrices=False)
    d1, d2 = vt[0], vt[1]
    explained = sv[:2] ** 2 / (sv ** 2).sum()
    path = np.stack([diff @ d1, diff @ d2], 1)

    def as_snap(vec):
        out, i = {}, 0
        for k in PARAMS:
            n = snaps[-1][k].size
            out[k] = vec[i:i + n].reshape(snaps[-1][k].shape)
            i += n
        return out

    base = move_scores(heads(snaps[-1], dn), legal)
    lin = [move_scores(heads(as_snap(d), dn), legal, empty_value=0.0) for d in (d1, d2)]   # per unit of each direction
    span = path.max(0) - path.min(0)
    a = np.linspace(path[:, 0].min() - margin * span[0], path[:, 0].max() + margin * span[0], steps)
    b = np.linspace(path[:, 1].min() - margin * span[1], path[:, 1].max() + margin * span[1], steps)
    loss = np.array([[policy_loss(base + x * lin[0] + y * lin[1], target) for x in a] for y in b])
    true = [policy_loss(move_scores(heads(s, dn), legal), target) for s in snaps]
    return a, b, loss, path, explained, true


def landscape_png(full, zoom, snaps, zoom_from: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, (a, b, loss, path, explained, true), title, first in (
            (axes[0], full, "From the newborn to 10 minutes of training", zoom_from),
            (axes[1], zoom, "Only the training after the probe", 0)):
        levels = np.quantile(loss, np.linspace(0, 1, 16))
        cs = ax.contourf(a, b, loss, levels=np.unique(levels), cmap="Blues_r")
        ax.contour(a, b, loss, levels=np.unique(levels), colors="white", linewidths=0.4, alpha=0.6)
        split = first                                                   # where training on new positions starts in this path
        if split:
            ax.plot(path[:split, 0], path[:split, 1], "-o", color=ORANGE, lw=1.4, ms=2.2, zorder=3,
                    label="readout probe: 50,000 positions, 8 times over")
        ax.plot(path[max(split - 1, 0):, 0], path[max(split - 1, 0):, 1], "-o", color=INK, lw=1.4, ms=2.2, zorder=3,
                label="training on new positions")
        ax.plot(*path[0], "o", ms=9, color=ORANGE, mec="white", zorder=4,
                label=f"start: loss {true[0]:.3f}" + (" (newborn)" if split else " (after the probe)"))
        ax.plot(*path[-1], "D", ms=8, color=INK, mec="white", zorder=4, label=f"end: loss {true[-1]:.3f}")
        ax.set_xlabel(f"direction 1 ({explained[0]:.0%} of how the weights moved)")
        ax.set_ylabel(f"direction 2 ({explained[1]:.0%})")
        ax.set_title(title)
        ax.grid(False)
        ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(cs, ax=ax, shrink=0.85, label="loss on 4,096 held-out positions", format="%.3f")
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def landscape_gif(full, out: Path, frames: int = 48) -> None:
    a, b, loss, path, _, true = full
    A, B = np.meshgrid(a, b)
    on_plane = [loss[np.abs(b - y).argmin(), np.abs(a - x).argmin()] for x, y in path]   # the path drawn on the surface
    top = np.quantile(loss, 0.97)
    images = []
    for k in range(frames):
        fig = plt.figure(figsize=(7, 5.6))
        ax = fig.add_axes((0.0, 0.0, 0.92, 0.93), projection="3d")
        ax.plot_surface(A, B, np.minimum(loss, top), cmap="Blues_r", linewidth=0, antialiased=True, alpha=0.9, rstride=1, cstride=1)
        ax.plot(path[:, 0], path[:, 1], np.minimum(on_plane, top) + 0.003, color=ORANGE, lw=2.2, zorder=10)
        ax.scatter(*path[-1], min(on_plane[-1], top), color=INK, s=30, zorder=11)
        ax.view_init(elev=32, azim=-60 + 360 * k / frames)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_zlabel("held-out loss")
        fig.text(0.03, 0.95, "The held-out loss around the path gradient descent took (orange)", fontsize=11, weight="bold",
                 color=INK)
        fig.patch.set_facecolor(SURFACE)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=90, facecolor=SURFACE, bbox_inches=None)
        plt.close(fig)
        images.append(Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    images[0].save(out, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=True)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dl-pictures")
    args = ap.parse_args()
    run = Path("runs") / args.run
    config = run / "config.json"
    if config.exists():
        import json
        cfg = json.loads(config.read_text())
        if cfg.get("lr_synapses", 0) > 0 or cfg.get("lr_neurons", 0) > 0:
            raise SystemExit(f"{args.run} trained the brain too, so one pass of the brain per position isn't enough here")
    snaps = snapshots(run)
    print(f"{len(snaps)} snapshots, {snaps[-1]['seen']:,} positions at the end")

    d = np.load(PACKED)
    legal, ci, cw = d["legal"][:HOLD], d["cand_idx"][:HOLD], d["cand_win"][:HOLD]
    dn = features(args.run, unpack(d["boards"][:HOLD]), run / "features_hold.npy")
    board = chess.Board()
    board.push_uci("e2e4")
    dn_e4 = features(args.run, planes(board).reshape(1, 768), run / "features_e4.npy")
    target = targets(ci, cw, legal.shape[1])

    learning_gif(snaps, dn_e4, OUT / "learning_e4.gif")
    full = plane(snaps, dn, legal, target)
    zoom_from = next(i for i, s in enumerate(snaps) if s["phase"].startswith("training"))
    zoom = plane(snaps[zoom_from - 1:], dn, legal, target)
    landscape_png(full, zoom, snaps, zoom_from, OUT / "landscape.png")
    landscape_gif(full, OUT / "landscape_3d.gif")
    print("loss along the path:", [round(x, 3) for x in full[5][::10]])


if __name__ == "__main__":
    main()

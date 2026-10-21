"""Where a training step's time goes, and what batch size, neuron order and the input pipeline change.

  python tools/bench.py              # about 10 minutes on my RTX 4060, nothing else on the GPU please
  python tools/bench.py --quick      # fewer repeats

Measures, per batch size: the frozen brain's forward pass (all a readout-only run needs), the full
plastic step (forward, the hand-written backward through 6 steps, Adam), the CPU's share (unpacking
boards), and the GPU's utilisation while it runs. Then the same forward with the neurons renumbered by
reverse Cuthill-McKee, so neurons that talk to each other sit close in memory.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from scipy.sparse.csgraph import reverse_cuthill_mckee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.connectome import Connectome
from flychess.model import FlyChess
from flychess.train_dl import PACKED, losses, unpack

NEURON_FIELDS = ("body_id", "type", "cls", "superclass", "side", "region", "nt", "hex1", "hex2", "soma")


def renumbered(c: Connectome, order: np.ndarray) -> Connectome:
    """The same brain with neuron order[i] renamed to i."""
    new = np.empty_like(order)
    new[order] = np.arange(len(order))
    return Connectome(**{**c.__dict__, **{k: getattr(c, k)[order] for k in NEURON_FIELDS},
                         "pre": new[c.pre].astype(np.int32), "post": new[c.post].astype(np.int32)})


class Utilisation:
    """nvidia-smi's SM utilisation, sampled once a second in the background."""

    def __enter__(self):
        self.samples, self.stop = [], False
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                   "-lms", "500"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self.read, daemon=True)
        self.t.start()
        return self

    def read(self):
        for line in self.p.stdout:
            if self.stop:
                break
            u, m = line.split(",")
            self.samples.append((float(u), float(m)))

    def __exit__(self, *exc):
        self.stop = True
        self.p.terminate()
        self.mean = float(np.mean([u for u, _ in self.samples])) if self.samples else float("nan")
        self.memory = float(np.max([m for _, m in self.samples])) if self.samples else float("nan")


def timed(fn, x, reps: int) -> float:
    """Seconds per call, after a warm-up, waiting for the GPU every time."""
    def once():
        out = fn(*x)
        return float(tf.reduce_sum(out[0] if isinstance(out, (list, tuple)) else out))
    once()
    t = time.perf_counter()
    for _ in range(reps):
        once()
    return (time.perf_counter() - t) / reps


def forward_fn(model):
    @tf.function
    def f(x):
        return model.descending(model.brain(x))
    return f


def plastic_fn(model):
    opt = {k: tf.keras.optimizers.Adam(1e-5) for k in ("readout", "synapses", "neurons")}
    groups = {"readout": model.readout_weights(), "synapses": [model.brain.theta], "neurons": [model.brain.gain, model.brain.bias]}
    variables = [v for vs in groups.values() for v in vs]

    @tf.function
    def f(x, legal, ci, cw, v):
        with tf.GradientTape() as tape:
            policy, value, _ = losses(model(x, training=True), legal, ci, cw, v)
            loss = policy + value
        grads = iter(tape.gradient(loss, variables))
        for k, vs in groups.items():
            opt[k].apply_gradients(zip([next(grads) for _ in vs], vs))
        return policy
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-measured", action="store_true", help="only the plastic step and the reordered brain")
    ap.add_argument("--out", type=Path, default=Path("runs/bench.json"))
    args = ap.parse_args()
    reps = 5 if args.quick else 20
    d = np.load(PACKED)
    rows = slice(10_000, 10_512)
    data = {k: d[k][rows] for k in ("boards", "legal", "cand_idx", "cand_win", "values")}
    x_all = unpack(data["boards"])
    c = connectome.load()
    results = {"gpu": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name")}

    # The CPU's part of a step: picking 128 random rows out of the packed arrays and unpacking the boards.
    big = {k: d[k] for k in ("boards", "legal", "cand_idx", "cand_win", "values")}
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(50):
        b = np.sort(rng.integers(0, len(big["boards"]), 128))
        _ = unpack(big["boards"][b]), big["legal"][b], big["cand_idx"][b], big["cand_win"][b], big["values"][b]
    results["cpu_ms_per_batch_128"] = round((time.perf_counter() - t) / 50 * 1000, 2)
    print(f"CPU, one batch of 128 picked and unpacked: {results['cpu_ms_per_batch_128']} ms")

    for name, brain in (("as measured", c), ("reverse Cuthill-McKee", None)):
        if brain is None:
            A = sp.csr_matrix((np.ones(len(c.pre)), (c.post, c.pre)), shape=(c.n, c.n))
            brain = renumbered(c, reverse_cuthill_mckee((A + A.T).tocsr(), symmetric_mode=True))
            bw = lambda cc: float(np.median(np.abs(cc.post.astype(np.int64) - cc.pre)))
            print(f"median |post - pre| index distance: {bw(c):.0f} as measured, {bw(brain):.0f} reordered")
        model = FlyChess(brain)
        model(np.zeros((1, 768), np.float32))
        fwd = forward_fn(model)
        for batch in (() if args.skip_measured and name == "as measured" else (16, 32, 64, 128, 256)):   # TF's GPU kernel stops at 344
            x = tf.constant(np.resize(x_all, (batch, 768)))
            with Utilisation() as u:
                s = timed(fwd, (x,), reps)
            key = f"forward {name} {batch}"
            results[key] = {"ms": round(s * 1000, 1), "positions_per_s": round(batch / s, 1), "gpu_util": round(u.mean),
                            "gpu_mem_mb": u.memory}
            print(f"forward, {name}, batch {batch}: {s * 1000:.1f} ms, {batch / s:.0f} positions/s, GPU {u.mean:.0f}% busy")
        if name == "as measured":
            plastic = FlyChess(c, window=6)
            plastic(np.zeros((1, 768), np.float32))                    # build the readout, or it has no weights to train
            step = plastic_fn(plastic)
            for batch in (32, 64, 128, 256):
                sl = lambda k: tf.constant(np.resize(data[k], (batch,) + data[k].shape[1:]))
                xs = (tf.constant(np.resize(x_all, (batch, 768))), sl("legal"), sl("cand_idx"), sl("cand_win"), sl("values"))
                try:
                    with Utilisation() as u:
                        s = timed(step, xs, max(3, reps // 2))
                except (tf.errors.ResourceExhaustedError, tf.errors.InvalidArgumentError):
                    print(f"plastic step, batch {batch}: out of GPU memory")
                    results[f"plastic {batch}"] = "out of memory"
                    break
                results[f"plastic {batch}"] = {"ms": round(s * 1000, 1), "positions_per_s": round(batch / s, 1),
                                               "gpu_util": round(u.mean), "gpu_mem_mb": u.memory}
                print(f"plastic step (window 6), batch {batch}: {s * 1000:.1f} ms, {batch / s:.0f} positions/s, GPU {u.mean:.0f}% busy")
            del step, plastic
        del model, fwd
        tf.keras.backend.clear_session()
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

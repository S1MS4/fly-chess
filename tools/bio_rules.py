"""Which dopamine rule teaches the fly? The experiment behind the README's bio_rules chart.

  python tools/bio_rules.py            # about 20 minutes on the RTX 4060

Every rule starts from the same untrained mushroom body and learns from the same kind of
tutor lessons (Stockfish's move vs a random one). After every 50 batches it is scored on
the same 512 held-out pairs. At the end a logistic regression on the Kenyon cells (and on
the descending neurons) shows the best any linear readout could do on those pairs: the
three-factor rule is a restricted linear readout, so it can't beat that ceiling.
Writes runs/bio-rules/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flychess import connectome
from flychess.bio import BioFly
from flychess.io_maps import motor_output
from flychess.train_bio import lessons, unpack

BATCH, VAL, TRAIN = 64, 512, 16384
RULES = [                                                            # the one bio.py uses runs twice as long, to see it hold
    {"name": "summed doses, transmitter valence (the first rule)", "mean": False, "differential": False, "valence": "transmitter", "lr": 0.05, "batches": 250},
    {"name": "only the cells that differ, transmitter valence", "mean": True, "differential": True, "valence": "transmitter", "lr": 1.0, "batches": 250},
    {"name": "only the cells that differ, compartment valence", "mean": True, "differential": True, "valence": "compartment", "lr": 1.0, "batches": 250},
    {"name": "averaged doses, compartment valence, lr 1", "mean": True, "differential": False, "valence": "compartment", "lr": 1.0, "batches": 250},
    {"name": "averaged doses, compartment valence, lr 0.1 (what bio.py uses)", "mean": True, "differential": False, "valence": "compartment", "lr": 0.1, "batches": 500},
]


def lesson(fly, good, bad, rule):
    """One batch of tutor lessons with this rule. Same three factors every time: Kenyon cell
    tags, dopamine (as big as the chance the fly still prefers the bad move), the compartment."""
    n = len(good)
    rates = fly.run(np.concatenate([good, bad]))
    s = fly.score(rates)
    wrong = 1 / (1 + np.exp(np.clip((s[:n] - s[n:]) / fly.temp, -30, 30)))
    kc = fly.sparse_kc(rates).numpy()
    tag_good, tag_bad = kc[:, :n], kc[:, n:]
    if rule["differential"]:                                         # only the cells that tell the two apart
        tag_good, tag_bad = np.maximum(tag_good - tag_bad, 0), np.maximum(tag_bad - tag_good, 0)
    scale = 1 / n if rule["mean"] else 1.0
    fly.weaken(np.outer(tag_good @ wrong * scale, fly.pam_reach) + np.outer(tag_bad @ wrong * scale, fly.ppl1_reach))


def ceiling(d_train, d_val, l2s=(1e-4, 1e-3, 1e-2)):
    """Logistic regression, P(good beats bad) = sigmoid(w . (good - bad)). Best held-out over a few L2s."""
    X, best = tf.constant(d_train), (0.0, 0.0)
    for l2 in l2s:
        w = tf.Variable(tf.zeros(X.shape[1]))
        opt = tf.keras.optimizers.Adam(1e-2)
        for _ in range(300):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.softplus(-tf.linalg.matvec(X, w))) + l2 * tf.reduce_sum(w * w)
            opt.apply_gradients([(tape.gradient(loss, w), w)])
        tr, va = float((d_train @ w.numpy() > 0).mean()), float((d_val @ w.numpy() > 0).mean())
        best = max(best, (va, tr))
    return {"held_out": best[0], "train": best[1]}


def main() -> None:
    c = connectome.load()
    fly = BioFly(c)
    good, bad = lessons(VAL + 256 + TRAIN + 500 * BATCH)
    vg, vb = unpack(good[:VAL]), unpack(bad[:VAL])
    fly.calibrate(unpack(good[VAL:VAL + 256]))
    by_compartment, by_transmitter = fly.valence.copy(), np.where(c.nt[fly.mbon] == "acetylcholine", 1.0, -1.0).astype(np.float32)

    def pair_accuracy():
        s = np.concatenate([fly.score(fly.run(vg[i:i + 128])) for i in range(0, VAL, 128)])
        t = np.concatenate([fly.score(fly.run(vb[i:i + 128])) for i in range(0, VAL, 128)])
        return float((s > t).mean() + 0.5 * (s == t).mean())

    rng, out = np.random.default_rng(0), {"rules": []}
    for rule in RULES:
        fly.w, fly.lr = fly.w0.copy(), rule["lr"]
        fly.valence = by_compartment if rule["valence"] == "compartment" else by_transmitter
        curve = [(0, pair_accuracy())]
        for k in range(1, rule["batches"] + 1):
            i = rng.integers(VAL + 256 + TRAIN, len(good) - BATCH)
            lesson(fly, unpack(good[i:i + BATCH]), unpack(bad[i:i + BATCH]), rule)
            if k % 50 == 0:
                curve.append((k, pair_accuracy()))
        print(rule["name"], " ".join(f"{a:.3f}" for _, a in curve), flush=True)
        out["rules"].append({**rule, "curve": curve})

    fly.w, fly.valence = fly.w0.copy(), by_compartment
    dn = motor_output(c)

    def features(x):                                                 # Kenyon cells as the MBONs see them, and descending neurons
        kc, d = [], []
        for i in range(0, len(x), 128):
            r = fly.run(x[i:i + 128])
            kc.append(fly.sparse_kc(r).numpy().T)
            d.append(r.numpy()[dn].T)
        return np.concatenate(kc), np.concatenate(d)

    rows = slice(VAL + 256, VAL + 256 + TRAIN)
    (kg, dg), (kb, db) = features(unpack(good[rows])), features(unpack(bad[rows]))
    (vkg, vdg), (vkb, vdb) = features(vg), features(vb)
    out["ceiling_kenyon_cells"] = ceiling(kg - kb, vkg - vkb)
    out["ceiling_descending_neurons"] = ceiling(dg - db, vdg - vdb)
    print("ceilings:", out["ceiling_kenyon_cells"], out["ceiling_descending_neurons"], flush=True)
    Path("runs/bio-rules").mkdir(parents=True, exist_ok=True)
    Path("runs/bio-rules/results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

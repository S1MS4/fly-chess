"""model(x, training=True) must reach the hand-written backward pass, so the brain's synapses get a gradient,
and calibrate(calm=True) must not change what the brain does with the board.

  python -m pytest tests     (or just: python tests/test_training_flag.py)

A tiny made-up brain: 48 photoreceptors on a 4 x 4 grid of eye columns, 40 other neurons, 12 descending ones.
"""
import numpy as np
import tensorflow as tf

from flychess.connectome import Connectome
from flychess.model import FlyChess


def tiny_brain(seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    kinds = ["R1-R6"] * 2 + ["R7", "R8"]                                # per column: 2 piece-type cells, own, opponent
    cols = [(h1, h2) for h1 in range(1, 5) for h2 in range(1, 5)]
    types = [k + "_x" for _ in cols for k in kinds][:48] + ["cb"] * 40 + ["dn"] * 12
    region = ["photoreceptors"] * 48 + ["central brain"] * 40 + ["descending"] * 12
    hex1 = np.array([c[0] for c in cols for _ in kinds][:48] + [np.nan] * 52, np.float32)
    hex2 = np.array([c[1] for c in cols for _ in kinds][:48] + [np.nan] * 52, np.float32)
    n = len(types)
    pairs = np.unique(rng.integers(0, n, (900, 2)), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    s = lambda v: np.array(v)
    return Connectome(body_id=np.arange(n), type=s(types), cls=s([""] * n), superclass=s([""] * n), side=s(["R"] * n),
                      region=s(region), nt=s(["acetylcholine"] * n), pre=pairs[:, 0].astype(np.int32),
                      post=pairs[:, 1].astype(np.int32), synapses=rng.integers(5, 40, len(pairs)).astype(np.float32),
                      sign=rng.choice([-1.0, 1.0], len(pairs)).astype(np.float32), hex1=hex1, hex2=hex2,
                      soma=np.zeros((n, 3), np.float32))


def boards(n: int, seed: int = 1) -> np.ndarray:
    return (np.random.default_rng(seed).random((n, 768)) < 0.04).astype(np.float32)


def test_training_reaches_the_brain():
    model = FlyChess(tiny_brain(), window=6)
    model(np.zeros((1, 768), np.float32))
    model.brain.calibrate(boards(64))
    x = tf.constant(boards(8, seed=2))
    with tf.GradientTape() as tape:
        fl, tl, pl, v = model(x, training=True)
        loss = tf.reduce_sum(fl) + tf.reduce_sum(tl) + tf.reduce_sum(v)
    g_theta, g_gain = tape.gradient(loss, [model.brain.theta, model.brain.gain])
    assert g_theta is not None and float(tf.reduce_sum(tf.abs(g_theta))) > 0   # the synapses learn
    assert g_gain is not None and float(tf.reduce_sum(tf.abs(g_gain))) > 0


def test_calm_changes_nothing_for_the_measured_brain():
    c, x = tiny_brain(), boards(64)
    out = []
    for calm in (False, True):
        model = FlyChess(c, eye="fair")
        model(np.zeros((1, 768), np.float32))
        model.brain.calibrate(x, calm=calm)
        out.append(model.descending(model.brain(tf.constant(boards(16, seed=3)))).numpy())
    assert np.abs(out[0] - out[1]).max() < 1e-4


def test_reading_several_steps():
    c, x = tiny_brain(), tf.constant(boards(4, seed=7))
    last = FlyChess(c)
    last(np.zeros((1, 768), np.float32))
    last.brain.calibrate(boards(64))
    many = FlyChess(c, read=(9, 10, 11, 12))
    many(np.zeros((1, 768), np.float32))
    for a, b in zip(many.weights, last.weights):                       # the same brain and readout
        a.assign(b)
    trace = last.brain(x, return_trace=True)
    want = last.heads(tf.add_n([last.descending(trace[s - 1]) for s in (9, 10, 11, 12)]) / 4)
    for a, b in zip(many(x), want):
        assert np.allclose(a.numpy(), b.numpy(), atol=1e-5)
    assert np.allclose(last(x)[0].numpy(), last.heads(last.descending(trace[-1]))[0].numpy(), atol=1e-6)   # the old path


if __name__ == "__main__":
    test_training_reaches_the_brain()
    test_calm_changes_nothing_for_the_measured_brain()
    test_reading_several_steps()
    print("training reaches the brain, calm homeostasis is the same brain, reading several steps adds up")

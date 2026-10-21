"""The hand-written backprop in model.unroll() must match what TensorFlow works out by itself.

  python -m pytest tests     (or just: python tests/test_gradient.py)
"""
import numpy as np
import tensorflow as tf

from flychess.model import unroll


def autodiff_unroll(indices, n, steps, leak, w, x, gain, bias, mu, inv_sd):
    W = tf.sparse.SparseTensor(indices, w, (n, n))
    rates = tf.zeros_like(x)
    for t in range(steps):
        z = gain * (tf.sparse.sparse_dense_matmul(W, rates) + x - mu[t]) * inv_sd[t] + bias
        rates = (1 - leak) * rates + leak * tf.nn.relu(tf.tanh(z))
    return rates


def test_matches_autodiff():
    rng = np.random.default_rng(0)
    n, e, batch, steps = 300, 3000, 8, 6
    f32 = lambda a: tf.constant(np.asarray(a, np.float32))
    pairs = np.unique(rng.integers(0, n, (e, 2)), axis=0)                    # (post, pre), sorted row-major
    indices = tf.constant(pairs.astype(np.int64))
    w, x = f32(rng.normal(0, 0.3, len(pairs))), f32(rng.normal(0.2, 0.5, (n, batch)))
    gain, bias = f32(rng.uniform(0.3, 1, (n, 1))), f32(rng.normal(0.2, 0.2, (n, 1)))
    mu, inv_sd = f32(rng.normal(0, 0.2, (steps, n, 1))), f32(rng.uniform(0.5, 2, (steps, n, 1)))
    readout = f32(rng.normal(0, 1, (n, batch)))                              # any loss will do

    def grads(f):
        with tf.GradientTape() as tape:
            tape.watch([w, x, gain, bias])
            loss = tf.reduce_sum(f(w, x, gain, bias, mu, inv_sd) * readout)
        return tape.gradient(loss, [w, x, gain, bias])

    mine = grads(unroll(indices, n, steps, 0.5, window=steps))
    theirs = grads(lambda *a: autodiff_unroll(indices, n, steps, 0.5, *a))
    for a, b in zip(mine, theirs):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_matches_autodiff()
    print("hand-written gradient matches autodiff")

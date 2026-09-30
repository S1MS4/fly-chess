"""The fly brain as a TensorFlow model.

Every neuron has a firing rate between 0 and 1. Each step, every neuron adds up
its inputs (rate of each upstream neuron x connection weight), squashes the sum,
and moves part of the way towards that value:

    drive  = W @ rate + eye_input + bias
    rate   = (1 - leak) * rate + leak * relu(tanh(drive))

After STEPS steps the descending neurons are read out as move scores.

W is the connectome. Which neurons connect, and whether a connection excites or
inhibits, is fixed by the data. Only the strength of each connection can learn:

    W = sign * softplus(theta)

softplus is always positive, so an excitatory synapse can get stronger or weaker
but it can never flip into an inhibitory one.
"""
import numpy as np
import tensorflow as tf

from .connectome import Connectome
from .io_maps import eye_input, motor_output

STEPS = 12
EDGE_CHUNK = 262_144  # edges per slice when computing weight gradients, keeps memory flat


def spmm(indices, n, values, rates):
    """Sparse W (n x n) times dense rates (n x batch), with a hand-written gradient.

    TensorFlow's built-in gradient for this builds an (edges x batch) tensor, which
    for ~10M edges and a batch of 128 is about 5 GB. This version does the same maths
    in slices of EDGE_CHUNK edges, so it fits on an 8 GB laptop GPU."""
    post, pre = indices[:, 0], indices[:, 1]

    @tf.custom_gradient
    def _f(values, rates):
        w = tf.sparse.SparseTensor(indices, values, (n, n))
        out = tf.sparse.sparse_dense_matmul(w, rates)

        def grad(g):
            g_rates = tf.sparse.sparse_dense_matmul(w, g, adjoint_a=True)
            e = tf.shape(values)[0]
            chunks = (e + EDGE_CHUNK - 1) // EDGE_CHUNK
            ta = tf.TensorArray(values.dtype, size=chunks, infer_shape=False)

            def body(i, ta):
                lo = i * EDGE_CHUNK
                hi = tf.minimum(lo + EDGE_CHUNK, e)
                part = tf.reduce_sum(tf.gather(g, post[lo:hi]) * tf.gather(rates, pre[lo:hi]), axis=1)
                return i + 1, ta.write(i, part)

            _, ta = tf.while_loop(lambda i, _: i < chunks, body, (tf.constant(0), ta))
            return ta.concat(), g_rates

        return out, grad

    return _f(values, rates)


class FlyBrain(tf.keras.layers.Layer):
    """The whole connectome, unrolled for STEPS steps."""

    def __init__(self, c: Connectome, steps: int = STEPS, leak: float = 0.5, **kw):
        super().__init__(**kw)
        self.n, self.steps, self.leak = c.n, steps, leak

        # Sort edges by (post, pre): SparseTensor wants them in row-major order.
        order = np.lexsort((c.pre, c.post))
        pre, post = c.pre[order], c.post[order]
        self.indices = tf.constant(np.stack([post, pre], 1).astype(np.int64))
        self.sign = tf.constant(c.sign[order])

        # Starting strength: log of the synapse count, scaled so every neuron's total
        # input starts around 1. Big synapse counts start strong, but not linearly so.
        m = np.log1p(c.synapses[order])
        m = m / np.maximum(np.bincount(post, weights=m, minlength=c.n)[post], 1e-6)
        theta0 = np.log(np.expm1(np.maximum(m, 1e-4)))  # inverse softplus
        self.theta = self.add_weight(name="theta", shape=theta0.shape, trainable=True,
                                     initializer=tf.constant_initializer(theta0))
        self.bias = self.add_weight(name="bias", shape=(c.n, 1), initializer="zeros")

        n_idx, f_idx = eye_input(c)
        self.eye = tf.sparse.reorder(tf.sparse.SparseTensor(
            np.stack([n_idx, f_idx], 1), np.ones(len(n_idx), np.float32), (c.n, 768)))
        self.eye_gain = self.add_weight(name="eye_gain", shape=(), initializer=tf.constant_initializer(1.0))

    def weights_now(self):
        return self.sign * tf.nn.softplus(self.theta)

    def call(self, board_planes, return_trace: bool = False):
        """board_planes: (batch, 768). Returns rates (n, batch), plus every step if asked."""
        x = self.eye_gain * tf.sparse.sparse_dense_matmul(self.eye, board_planes, adjoint_b=True)
        w = self.weights_now()
        rates = tf.zeros((self.n, tf.shape(board_planes)[0]))
        trace = []
        for _ in range(self.steps):
            drive = spmm(self.indices, self.n, w, rates) + x + self.bias
            rates = (1 - self.leak) * rates + self.leak * tf.nn.relu(tf.tanh(drive))
            if return_trace:
                trace.append(rates)
        return (rates, trace) if return_trace else rates


class FlyChess(tf.keras.Model):
    """Eye -> connectome -> descending neurons -> move and value readouts."""

    def __init__(self, c: Connectome, steps: int = STEPS):
        super().__init__()
        self.brain = FlyBrain(c, steps)
        self.motor = tf.constant(motor_output(c))
        self.from_head = tf.keras.layers.Dense(64, name="from_square")
        self.to_head = tf.keras.layers.Dense(64, name="to_square")
        self.promo_head = tf.keras.layers.Dense(5, name="promotion")
        self.value_head = tf.keras.layers.Dense(1, activation="tanh", name="value")

    def call(self, board_planes, return_trace: bool = False):
        out = self.brain(board_planes, return_trace=return_trace)
        rates, trace = out if return_trace else (out, None)
        dn = tf.transpose(tf.gather(rates, self.motor))  # (batch, descending neurons)
        heads = (self.from_head(dn), self.to_head(dn), self.promo_head(dn), self.value_head(dn)[:, 0])
        return (heads, trace) if return_trace else heads

"""The fly brain as a TensorFlow model.

Every neuron has a firing rate between 0 and 1. Each step, every neuron adds up
its inputs (rate of each upstream neuron x connection weight), then homeostasis
steps in: the sum is centred and scaled by what that neuron usually gets at that
step, the neuron applies its own gain and bias, and its rate moves halfway there:

    u     = W @ rate + eye_input
    z     = gain * (u - mu[step]) / sd[step] + bias
    rate  = (1 - leak) * rate + leak * relu(tanh(z))

After STEPS steps the descending neurons are read out as move scores.

W is the connectome. Which neurons connect, and whether a connection excites or
inhibits, is fixed by the data. Only the strength of each connection can learn:

    W = sign * softplus(theta)

softplus is always positive, so an excitatory synapse can get stronger or weaker
but it can never flip into an inhibitory one.

The backward pass (backprop through time) is written out by hand in unroll().
It is the same maths TensorFlow would do, just without the huge temporary
tensors, and it can stop after the last few steps instead of going all the way
back to the eye.
"""
import numpy as np
import tensorflow as tf

from .connectome import Connectome
from .io_maps import eye_input, motor_output

STEPS = 12
# A neuron whose input barely moves between boards would otherwise get scaled up a
# thousandfold, and then the tiniest weight change throws it from silent to saturated.
SD_FLOOR = 0.02
# Below this variance the board hasn't reached a neuron at that step at all: every board gives it
# exactly the same input. calibrate(calm=True) gives those no extra gain instead of 1 / SD_FLOOR.
QUIET = 1e-10
plain = tf.autograph.experimental.do_not_convert                      # plain TensorFlow, no AutoGraph (it rereads files)


@tf.function(jit_compile=True)
def connection_grads(g_u, rates, post, pre):
    """How much each connection is to blame: summed over the batch, the error at its
    post neuron times the rate of its pre neuron. XLA turns the two gathers and the
    sum into one pass, about 10x faster than doing them one by one."""
    return tf.reduce_sum(tf.gather(g_u, post) * tf.gather(rates, pre), axis=1)


def unroll(indices, n: int, steps: int, leak: float, window: int):
    """The brain's dynamics for `steps` steps. Backprop only walks back through the
    last `window` steps (truncated backprop through time)."""
    post, pre = tf.cast(indices[:, 0], tf.int32), tf.cast(indices[:, 1], tf.int32)

    @tf.custom_gradient
    def run(w, x, gain, bias, mu, inv_sd):
        W = tf.sparse.SparseTensor(indices, w, (n, n))
        rates, us, zs = [tf.zeros_like(x)], [], []
        for t in range(steps):
            us.append(tf.sparse.sparse_dense_matmul(W, rates[-1]) + x)   # everything arriving at each neuron
            zs.append(gain * (us[-1] - mu[t]) * inv_sd[t] + bias)          # homeostasis, then its own gain and bias
            rates.append((1 - leak) * rates[-1] + leak * tf.nn.relu(tf.tanh(zs[-1])))

        def grad(g):                                                       # g: how the loss changes with the last rates
            g = tf.convert_to_tensor(g)                                    # only the descending neurons are read, so it arrives sparse
            g_w, g_x = tf.zeros_like(w), tf.zeros_like(x)
            g_gain, g_bias = tf.zeros_like(gain), tf.zeros_like(bias)
            for t in reversed(range(steps - window, steps)):
                th = tf.tanh(zs[t])
                g_z = g * leak * tf.where(zs[t] > 0, 1 - th * th, 0.0)     # back through relu(tanh(z))
                g_gain += tf.reduce_sum(g_z * (us[t] - mu[t]) * inv_sd[t], 1, keepdims=True)
                g_bias += tf.reduce_sum(g_z, 1, keepdims=True)
                g_u = g_z * gain * inv_sd[t]                               # back through the homeostasis
                g_x += g_u                                                 # the eye feeds every step
                g_w += connection_grads(g_u, rates[t], post, pre)          # blame for every connection
                g = (1 - leak) * g + tf.sparse.sparse_dense_matmul(W, g_u, adjoint_a=True)  # one step back in time
            return g_w, g_x, g_gain, g_bias, None, None

        return rates[-1], grad

    return run


class FlyBrain(tf.keras.layers.Layer):
    """The whole connectome, unrolled for STEPS steps."""

    def __init__(self, c: Connectome, steps: int = STEPS, leak: float = 0.5, window: int | None = None,
                 eye: str = "ranks", **kw):
        super().__init__(**kw)
        self.n, self.steps, self.leak = c.n, steps, leak

        # Sort edges by (post, pre): SparseTensor wants them in row-major order.
        order = np.lexsort((c.pre, c.post))
        pre, post = c.pre[order], c.post[order]
        self.indices = tf.constant(np.stack([post, pre], 1).astype(np.int64))
        self.sign = tf.constant(c.sign[order])
        self.run = unroll(self.indices, c.n, steps, leak, window or steps)

        # Starting strength: log of the synapse count, as a share of everything the post
        # neuron receives. Big synapse counts start strong, but not linearly so.
        m = np.log1p(c.synapses[order])
        m = m / np.maximum(np.bincount(post, weights=m, minlength=c.n)[post], 1e-6)
        theta0 = np.log(np.expm1(np.maximum(m, 1e-4)))                    # inverse softplus
        self.theta = self.add_weight(name="theta", shape=theta0.shape, initializer=tf.constant_initializer(theta0))

        # Per neuron: a gain and a bias that learn, and per step the usual input (mu) and its
        # spread (1 / sd) that calibrate() measures once and then keeps.
        self.gain = self.add_weight(name="gain", shape=(c.n, 1), initializer=tf.constant_initializer(0.5))
        self.bias = self.add_weight(name="bias", shape=(c.n, 1), initializer=tf.constant_initializer(0.3))
        self.mu = self.add_weight(name="mu", shape=(steps, c.n, 1), initializer="zeros", trainable=False)
        self.inv_sd = self.add_weight(name="inv_sd", shape=(steps, c.n, 1), initializer="ones", trainable=False)

        n_idx, f_idx = eye_input(c, eye)                                  # "ranks" (the first flies) or "fair", see io_maps
        self.eye = tf.sparse.reorder(tf.sparse.SparseTensor(
            np.stack([n_idx, f_idx], 1), np.ones(len(n_idx), np.float32), (c.n, 768)))

    @plain
    def weights_now(self):
        return self.sign * tf.nn.softplus(self.theta)

    @plain
    def inputs(self, board_planes):
        """What the eye sends each neuron, (neurons, batch). The same at every step."""
        return tf.sparse.sparse_dense_matmul(self.eye, board_planes, adjoint_b=True)

    def calibrate(self, boards, calm: bool = False) -> None:
        """Homeostasis, measured once. Real boards go through, and for every neuron and
        step this remembers its usual input and how much that input moves between boards.
        Without it the board fades a little at every synapse and is gone before the
        central brain. With it every neuron sees its input centred and scaled.

        calm: a neuron the board hasn't reached yet at some step gets exactly the same input
        for every board, so the floor would give it the biggest gain there is (x50). For the
        brain as measured that changes nothing, its input never moves; but once synapses learn
        it does, 11 times over per step (README, "Watching the error go back"). calm=True gives
        those neurons gain 1 instead."""
        x = self.inputs(boards)
        W = tf.sparse.SparseTensor(self.indices, self.weights_now(), (self.n, self.n))
        rates, mus, inv_sds = tf.zeros_like(x), [], []
        for _ in range(self.steps):
            u = tf.sparse.sparse_dense_matmul(W, rates) + x
            mean, var = tf.nn.moments(u, axes=[1], keepdims=True)
            mus.append(mean)
            inv = 1 / tf.maximum(tf.sqrt(var), SD_FLOOR)
            inv_sds.append(tf.where(var < QUIET, 1.0, inv) if calm else inv)
            z = self.gain * (u - mean) * inv_sds[-1] + self.bias
            rates = (1 - self.leak) * rates + self.leak * tf.nn.relu(tf.tanh(z))
        self.mu.assign(tf.stack(mus))
        self.inv_sd.assign(tf.stack(inv_sds))

    @plain
    def call(self, board_planes, training: bool = False, return_trace: bool = False):
        """board_planes: (batch, 768). Returns rates (n, batch), or the rates after every step.
        Training keeps every step for the backward pass; otherwise only the current one is kept."""
        x, w = self.inputs(board_planes), self.weights_now()
        if training:
            return self.run(w, x, *map(tf.convert_to_tensor, (self.gain, self.bias, self.mu, self.inv_sd)))
        W = tf.sparse.SparseTensor(self.indices, w, (self.n, self.n))
        rates, trace = tf.zeros_like(x), []
        for t in range(self.steps):
            z = self.gain * (tf.sparse.sparse_dense_matmul(W, rates) + x - self.mu[t]) * self.inv_sd[t] + self.bias
            rates = (1 - self.leak) * rates + self.leak * tf.nn.relu(tf.tanh(z))
            if return_trace:
                trace.append(rates)
        return trace if return_trace else rates


class FlyChess(tf.keras.Model):
    """Eye -> connectome -> descending neurons -> move and value readouts.

    read: the steps (1-based) whose descending-neuron rates the readout averages. (12,) is the last step only,
    what the first flies do; (9, 10, 11, 12) reads the command as it builds up, the way a motor circuit adds
    up its input over time. Only a frozen brain can read more than one step (the hand-written backward pass
    starts from the last step)."""

    def __init__(self, c: Connectome, steps: int = STEPS, window: int | None = None, eye: str = "ranks",
                 read: tuple[int, ...] = (STEPS,)):
        super().__init__()
        self.read = tuple(read)
        self.brain = FlyBrain(c, steps, window=window, eye=eye)
        self.motor = tf.constant(motor_output(c))
        self.from_head = tf.keras.layers.Dense(64, name="from_square")
        self.to_head = tf.keras.layers.Dense(64, name="to_square")
        self.promo_head = tf.keras.layers.Dense(5, name="promotion")
        self.value_head = tf.keras.layers.Dense(1, activation="tanh", name="value")

    def readout_weights(self):
        return [w for h in (self.from_head, self.to_head, self.promo_head, self.value_head) for w in h.trainable_weights]

    @plain
    def descending(self, rates):
        return tf.transpose(tf.gather(rates, self.motor))                  # (batch, descending neurons)

    @plain
    def heads(self, dn):
        return self.from_head(dn), self.to_head(dn), self.promo_head(dn), self.value_head(dn)[:, 0]

    @plain
    def read_out(self, trace):
        """The descending neurons the readout sees: their mean over the steps in self.read."""
        return tf.add_n([self.descending(trace[s - 1]) for s in self.read]) / len(self.read)

    @plain
    def call(self, board_planes, training: bool = False, return_trace: bool = False):
        if self.read == (self.brain.steps,) and not return_trace:
            return self.heads(self.descending(self.brain(board_planes, training=training)))
        if training:
            raise ValueError("training the brain only works when the readout reads the last step")
        trace = self.brain(board_planes, return_trace=True)
        out = self.heads(self.read_out(trace))
        return (out, trace) if return_trace else out

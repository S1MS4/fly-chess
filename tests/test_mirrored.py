"""Data parallel training (tf.distribute.MirroredStrategy, e.g. Kaggle's 2 x T4) works with the hand-written
backward pass and the sparse brain: two replicas on half a batch each give the same gradient as one on all of it.

  python tests/test_mirrored.py      (runs on two logical CPUs, no GPU needed)
"""
import numpy as np
import tensorflow as tf

tf.config.set_logical_device_configuration(tf.config.list_physical_devices("CPU")[0],
                                           [tf.config.LogicalDeviceConfiguration()] * 2)

from flychess.model import FlyChess  # noqa: E402  (after the devices are set up)
from tests.test_training_flag import boards, tiny_brain  # noqa: E402


def gradients(model, x):
    with tf.GradientTape() as tape:
        fl, tl, pl, v = model(x, training=True)
        loss = tf.reduce_mean(tf.reduce_sum(fl * fl, 1) + tf.reduce_sum(tl, 1) + v)
    return tape.gradient(loss, [model.brain.theta, model.brain.gain])


def test_two_replicas_match_one():
    c, x = tiny_brain(), boards(16, seed=5)
    one = FlyChess(c, window=6)
    one(np.zeros((1, 768), np.float32))
    one.brain.calibrate(boards(64))
    g_one = gradients(one, tf.constant(x))

    strategy = tf.distribute.MirroredStrategy(["/cpu:0", "/cpu:1"])
    with strategy.scope():
        two = FlyChess(c, window=6)
        two(np.zeros((1, 768), np.float32))
    for a, b in zip(two.weights, one.weights):                           # the same brain and readout as `one`
        a.assign(b)
    halves = strategy.experimental_distribute_values_from_function(lambda ctx: tf.constant(x[ctx.replica_id_in_sync_group * 8:][:8]))
    per_replica = strategy.run(lambda xs: gradients(two, xs), args=(halves,))
    g_two = [strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None) for g in per_replica]
    for a, b in zip(g_one, g_two):
        assert np.allclose(a.numpy(), b.numpy(), atol=1e-5), np.abs(a.numpy() - b.numpy()).max()


if __name__ == "__main__":
    test_two_replicas_match_one()
    print("two replicas give the same gradient as one")

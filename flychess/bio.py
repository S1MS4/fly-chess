"""Biological version: the fly learns the way flies learn, no backprop.

Almost the whole brain is frozen at its measured wiring. The one place that
learns is the mushroom body, specifically the synapses from Kenyon cells (KCs)
onto mushroom body output neurons (MBONs). That is where real flies store
"this smell was good" and "this smell was bad".

How a fly learns an odour, and how this fly learns a move:
  1. A position (an "odour") activates a sparse set of Kenyon cells.
  2. Kenyon cells drive MBONs. Some MBONs push the fly towards things
     (approach), some push it away (avoid).
  3. Recently active Kenyon cells stay "tagged" for a while (eligibility trace).
  4. Something good happens: reward dopamine neurons (PAM) fire.
     Something bad happens: punishment dopamine neurons (PPL1) fire.
  5. Dopamine weakens the tagged KC -> MBON synapses in the compartments those
     dopamine neurons cover. PAM covers mostly avoid MBONs, so after a reward
     the position drives "avoid" less and wins more "approach". PPL1 does the
     opposite.

Which dopamine neuron covers which MBON is read from the connectome (DAN -> MBON
synapses), not hand-written. Whether an MBON means approach or avoid comes from
its transmitter: cholinergic MBONs count as approach, glutamatergic and GABAergic
ones as avoid. That split is a rough rule from fly research, not a law.
"""
import numpy as np
import tensorflow as tf

from .connectome import Connectome
from .model import FlyBrain, spmm


class BioFly:
    def __init__(self, c: Connectome, steps: int = 12, lr: float = 0.02, trace_decay: float = 0.8,
                 recovery: float = 0.001, temperature: float = 0.05, seed: int = 0):
        self.kc, self.mbon = c.where("kenyon cells"), c.where("MBON")
        self.pam, self.ppl1 = c.where("dopamine (PAM)"), c.where("dopamine (PPL1)")

        # Pull every KC -> MBON connection out of the frozen brain into a small
        # dense matrix that can change.
        is_kc, is_mbon = np.isin(c.pre, self.kc), np.isin(c.post, self.mbon)
        plastic = is_kc & is_mbon
        kc_pos = {n: i for i, n in enumerate(self.kc)}
        mb_pos = {n: i for i, n in enumerate(self.mbon)}
        self.brain = FlyBrain(Connectome(**{**c.__dict__, "pre": c.pre[~plastic], "post": c.post[~plastic],
                                            "synapses": c.synapses[~plastic], "sign": c.sign[~plastic]}), steps)
        self.w0 = np.zeros((len(self.kc), len(self.mbon)), np.float32)
        for p, q, s in zip(c.pre[plastic], c.post[plastic], c.synapses[plastic]):
            self.w0[kc_pos[p], mb_pos[q]] = np.log1p(s)
        self.w0 /= np.maximum(self.w0.sum(0, keepdims=True), 1e-6)
        self.mask = self.w0 > 0
        self.w = self.w0.copy()

        # Approach (+1) or avoid (-1) per MBON, from its transmitter.
        self.valence = np.where(c.nt[self.mbon] == "acetylcholine", 1.0, -1.0).astype(np.float32)

        # Dopamine compartments: how strongly each DAN group reaches each MBON.
        def reach(dans):
            m = np.zeros(len(self.mbon), np.float32)
            sel = np.isin(c.pre, dans) & is_mbon
            np.add.at(m, [mb_pos[q] for q in c.post[sel]], c.synapses[sel])
            return m / max(m.max(), 1e-6)
        self.pam_reach, self.ppl1_reach = reach(self.pam), reach(self.ppl1)

        self.lr, self.decay, self.recovery, self.temp = lr, trace_decay, recovery, temperature
        self.trace = np.zeros(len(self.kc), np.float32)
        self.rng = np.random.default_rng(seed)
        self._kc_t = tf.constant(self.kc)
        self._mbon_t = tf.constant(self.mbon[:, None])

    def run(self, board_planes: np.ndarray, return_trace: bool = False):
        """Same dynamics as FlyBrain, plus the plastic KC -> MBON path each step."""
        b = self.brain
        x = b.eye_gain * tf.sparse.sparse_dense_matmul(b.eye, board_planes, adjoint_b=True)
        w, wkm = b.weights_now(), tf.constant(self.w)
        rates = tf.zeros((b.n, board_planes.shape[0]))
        trace = []
        for _ in range(b.steps):
            drive = spmm(b.indices, b.n, w, rates) + x + b.bias
            kc_to_mbon = tf.matmul(wkm, tf.gather(rates, self._kc_t), transpose_a=True)
            drive = tf.tensor_scatter_nd_add(drive, self._mbon_t, kc_to_mbon)
            rates = (1 - b.leak) * rates + b.leak * tf.nn.relu(tf.tanh(drive))
            if return_trace:
                trace.append(rates)
        return (rates, trace) if return_trace else rates

    def choose(self, options: np.ndarray, greedy: bool = False, return_trace: bool = False):
        """options: (n_moves, 768), one row per position the fly could move into.
        Returns the chosen index and the approach-minus-avoid score of every option."""
        out = self.run(options, return_trace)
        rates, trace = out if return_trace else (out, None)
        mb = tf.gather(rates, self.mbon).numpy()  # (mbon, n_moves)
        score = self.valence @ mb
        if greedy:
            i = int(score.argmax())
        else:
            p = np.exp((score - score.max()) / self.temp)
            i = int(self.rng.choice(len(score), p=p / p.sum()))
        kc_rates = tf.gather(rates, self.kc)[:, i].numpy()
        self.trace = self.decay * self.trace + kc_rates  # tag the Kenyon cells that were active
        return (i, score, trace) if return_trace else (i, score)

    def dopamine(self, reward: float) -> None:
        """reward > 0 fires PAM, reward < 0 fires PPL1. Tagged KC -> MBON synapses in the
        reached compartments get weaker, then everything drifts slowly back to baseline."""
        reach = self.pam_reach * max(reward, 0) + self.ppl1_reach * max(-reward, 0)
        self.w -= self.lr * np.outer(self.trace, reach) * self.mask
        self.w += self.recovery * (self.w0 - self.w)
        np.clip(self.w, 0, None, out=self.w)

    def reset_trace(self) -> None:
        self.trace[:] = 0

    def save(self, path) -> None:
        np.save(path, self.w)

    def load(self, path) -> None:
        self.w = np.load(path)

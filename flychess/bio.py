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
synapses), not hand-written, and so is whether an MBON means approach or avoid:
the opposite of the dopamine that reaches it, the pairing Aso et al. (2014) found
across the mushroom body.
"""
import numpy as np
import tensorflow as tf

from .connectome import Connectome
from .model import SD_FLOOR, FlyBrain


class BioFly:
    def __init__(self, c: Connectome, steps: int = 12, lr: float = 0.1, trace_decay: float = 0.8,
                 recovery: float = 0.001, temperature: float = 1.0, sparse: float = 0.05, seed: int = 0, eye: str = "ranks"):
        self.kc, self.mbon = c.where("kenyon cells"), c.where("MBON")
        self.brain = FlyBrain(c, steps, eye=eye)
        self.k = int(sparse * len(self.kc))                                 # Kenyon cells allowed to fire per position

        # Pull every KC -> MBON connection out of the frozen brain into a small dense
        # matrix that can change. It starts at exactly the strength the brain gave it.
        post, pre = self.brain.indices.numpy().T
        w = self.brain.weights_now().numpy()
        plastic = np.isin(pre, self.kc) & np.isin(post, self.mbon)
        self.frozen = tf.constant(np.where(plastic, 0.0, w).astype(np.float32))
        kc_pos = {n: i for i, n in enumerate(self.kc)}
        mb_pos = {n: i for i, n in enumerate(self.mbon)}
        self.w0 = np.zeros((len(self.kc), len(self.mbon)), np.float32)
        for p, q, s in zip(pre[plastic], post[plastic], w[plastic]):
            self.w0[kc_pos[p], mb_pos[q]] = abs(s)
        self.w = self.w0.copy()

        # Dopamine compartments: how many synapses each DAN group makes onto each MBON.
        is_mbon = np.isin(c.post, self.mbon)

        def synapses_from(dans):
            m = np.zeros(len(self.mbon), np.float32)
            sel = np.isin(c.pre, dans) & is_mbon
            np.add.at(m, [mb_pos[q] for q in c.post[sel]], c.synapses[sel])
            return m
        pam, ppl1 = synapses_from(c.where("dopamine (PAM)")), synapses_from(c.where("dopamine (PPL1)"))
        self.pam_reach, self.ppl1_reach = pam / pam.max(), ppl1 / ppl1.max()

        # Approach (+1) or avoid (-1) per MBON: the opposite of the dopamine that teaches it (Aso et al.
        # 2014). Where reward dopamine (PAM) lands the MBON pushes away, where punishment dopamine (PPL1)
        # lands it pulls in, and the transmitter only decides when no dopamine neuron reaches it. Going
        # by transmitter alone taught the fly backwards: PPL1's biggest target, MBON11, is a GABA neuron,
        # and punishment works in real flies by turning exactly that one down.
        by_transmitter = np.where(c.nt[self.mbon] == "acetylcholine", 1.0, -1.0)
        self.valence = np.where(ppl1 > pam, 1.0, np.where(pam > ppl1, -1.0, by_transmitter)).astype(np.float32)

        self.lr, self.decay, self.recovery, self.temp = lr, trace_decay, recovery, temperature
        self.trace = np.zeros(len(self.kc), np.float32)
        self.rng = np.random.default_rng(seed)
        self._kc_t = tf.constant(self.kc)
        self._mbon_t = tf.constant(self.mbon[:, None])

    def sparse_kc(self, rates):
        """Only the most active 5% of Kenyon cells pass their signal on, the way the APL neuron's
        feedback inhibition keeps them sparse in a real fly. Without it about 64% fire for every
        board, and every lesson would overwrite the last one. The cut is a steep sigmoid, not a
        step: with a step, a cell right at the edge flipped on rounding noise and the jump went
        round the brain's loops, so the same position scored twice disagreed 36% of the time."""
        kc = tf.gather(rates, self._kc_t)                                    # (kenyon cells, batch)
        cut = tf.math.top_k(tf.transpose(kc), self.k).values[:, -1]          # the k-th strongest, per position
        return kc * tf.sigmoid((kc - cut) / 0.01)                            # full rate above the cut, ~0 below, smooth in between

    def run(self, board_planes: np.ndarray, return_trace: bool = False, calibrate: bool = False):
        """Same dynamics as FlyBrain, with the plastic KC -> MBON path added every step.
        With calibrate=True it also measures the homeostasis statistics on these boards."""
        b = self.brain
        x = b.inputs(board_planes)
        W = tf.sparse.SparseTensor(b.indices, self.frozen, (b.n, b.n))
        wkm = tf.constant(self.w)
        rates, trace, mus, inv_sds = tf.zeros_like(x), [], [], []
        for t in range(b.steps):
            u = tf.sparse.sparse_dense_matmul(W, rates) + x
            u = tf.tensor_scatter_nd_add(u, self._mbon_t, tf.matmul(wkm, self.sparse_kc(rates), transpose_a=True))
            if calibrate:
                mean, var = tf.nn.moments(u, axes=[1], keepdims=True)
                mus.append(mean)
                inv_sds.append(1 / tf.maximum(tf.sqrt(var), SD_FLOOR))
            mu, inv_sd = (mus[-1], inv_sds[-1]) if calibrate else (b.mu[t], b.inv_sd[t])
            rates = (1 - b.leak) * rates + b.leak * tf.nn.relu(tf.tanh(b.gain * (u - mu) * inv_sd + b.bias))
            if return_trace:
                trace.append(rates)
        if calibrate:
            b.mu.assign(tf.stack(mus))
            b.inv_sd.assign(tf.stack(inv_sds))
        return trace if return_trace else rates

    def calibrate(self, boards: np.ndarray) -> None:
        self.run(boards, calibrate=True)

    def score(self, rates) -> np.ndarray:
        """Approach minus avoid, one number per position."""
        return self.valence @ tf.gather(rates, self.mbon).numpy()

    def choose(self, options: np.ndarray, greedy: bool = False, tag: bool = True, return_trace: bool = False):
        """options: (n_moves, 768), one row per position the fly could move into.
        Returns the chosen index and the approach-minus-avoid score of every option."""
        trace = self.run(options, return_trace=True)
        score = self.score(trace[-1])
        if greedy:
            i = int(score.argmax())
        else:
            p = np.exp((score - score.max()) / self.temp)
            i = int(self.rng.choice(len(score), p=p / p.sum()))
        if tag:
            kc_rates = self.sparse_kc(trace[-1])[:, i].numpy()
            self.trace = self.decay * self.trace + kc_rates             # tag the Kenyon cells that were active
        return (i, score, trace) if return_trace else (i, score)

    def dopamine(self, reward: float) -> None:
        """reward > 0 fires PAM, reward < 0 fires PPL1. Tagged KC -> MBON synapses in the
        reached compartments get weaker (by a share of their own strength), then everything
        drifts slowly back to baseline."""
        reach = self.pam_reach * max(reward, 0) + self.ppl1_reach * max(-reward, 0)
        self.weaken(np.outer(self.trace, reach))

    def weaken(self, dose: np.ndarray) -> None:
        """dose (Kenyon cells x MBONs): how much of each synapse's starting strength to take away, times lr."""
        self.w -= self.lr * self.w0 * dose
        self.w += self.recovery * (self.w0 - self.w)
        np.clip(self.w, 0, None, out=self.w)

    def tutor(self, good: np.ndarray, bad: np.ndarray) -> float:
        """A lesson with a teacher, the way flies are trained in the lab: after the move
        Stockfish liked, sugar (PAM); after some other move, a shock (PPL1). Each dose is
        as big as the chance the fly would still pick the bad one, so moves it already
        gets right stop teaching it anything. Same three factors as dopamine(): the Kenyon
        cells that were active, the dopamine, and the compartment it reaches. The doses are
        averaged over the batch, so the batch size doesn't change how fast it learns.
        good, bad: (batch, 768) positions after each move. Returns how often it was right."""
        rates = self.run(np.concatenate([good, bad]))
        s = self.score(rates)
        n = len(good)
        wrong = 1 / (1 + np.exp(np.clip((s[:n] - s[n:]) / self.temp, -30, 30)))  # chance it prefers the bad move
        kc = self.sparse_kc(rates).numpy()
        sugar = kc[:, :n] @ wrong / n                                       # KC tags from the good positions
        shock = kc[:, n:] @ wrong / n                                       # and from the bad ones
        self.weaken(np.outer(sugar, self.pam_reach) + np.outer(shock, self.ppl1_reach))
        return float((s[:n] > s[n:]).mean())

    def reset_trace(self) -> None:
        self.trace[:] = 0

    def save(self, path) -> None:
        np.savez(path, w=self.w, mu=self.brain.mu.numpy(), inv_sd=self.brain.inv_sd.numpy())

    def load(self, path) -> None:
        d = np.load(path)
        self.w = d["w"]
        self.brain.mu.assign(d["mu"])
        self.brain.inv_sd.assign(d["inv_sd"])

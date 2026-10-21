"""Turn the MaleCNS tables into something a neural network can use.

The raw download is three tables. This file squeezes them into one compact
data/connectome/brain.npz with:
  - a list of neurons (id, type, class, superclass, transmitter, region, soma position)
  - a list of connections (pre index, post index, synapse count, sign)
  - where each photoreceptor sits on the eye, so the board can be "shown" to it

Run once after downloading:  python -m flychess.connectome --build
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/connectome")
BRAIN = RAW / "brain.npz"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

# Fast transmitters decide the sign. Acetylcholine excites. GABA and glutamate
# (through GluCl channels) mostly inhibit in the fly. Histamine is what
# photoreceptors release and it hyperpolarises the next cell. The slow
# modulators (dopamine, serotonin, octopamine) are treated as weak excitation,
# which is a simplification.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0}

# Order matters: it is also the order of rows in the brain viewer.
REGIONS = ["photoreceptors", "optic lobe", "visual projection", "antennal lobe",
           "kenyon cells", "MBON", "dopamine (PAM)", "dopamine (PPL1)", "central complex",
           "central brain", "descending", "ascending", "nerve cord", "other"]


def region_of(row) -> str:
    t, c, s = str(row.type), str(row["class"]), str(row.superclass)
    if t.startswith(("R1-R6", "R7", "R8")):
        return "photoreceptors"
    if c == "Kenyon_Cell":
        return "kenyon cells"
    if c == "MBON":
        return "MBON"
    if c == "DAN":
        return "dopamine (PAM)" if t.startswith("PAM") else "dopamine (PPL1)" if t.startswith("PPL1") else "central brain"
    if c == "CX":
        return "central complex"
    if c in ("olfactory", "ALPN", "ALLN", "ALIN", "ALON"):
        return "antennal lobe"
    if s.startswith("ol_") or s == "visual_centrifugal":
        return "optic lobe"
    if s.startswith("visual_projection"):
        return "visual projection"
    if s == "descending_neuron":
        return "descending"
    if s == "ascending_neuron":
        return "ascending"
    if s.startswith("cb_"):
        return "central brain"
    if s.startswith("vnc_"):
        return "nerve cord"
    return "other"


def transmitter(nt: pd.DataFrame, ids) -> pd.Series:
    """Best guess per neuron: the consensus call, else the cell type's, else its own prediction."""
    t = nt.drop_duplicates("body").set_index("body").reindex(ids)
    known = lambda s: s.notna() & (s != "unclear")
    tx = t.consensus_nt.where(known(t.consensus_nt), t.celltype_predicted_nt)
    tx = tx.where(known(tx), t.predicted_nt)
    return tx.where(known(tx), "acetylcholine").str.lower()


def build(min_synapses: int, brain_only: bool) -> None:
    ann = pd.read_feather(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = pd.read_feather(RAW / "body-neurotransmitters-male-cns-v1.0.feather")
    w = pd.read_feather(RAW / WEIGHTS, columns=["body_pre", "body_post", "weight"])
    print(f"raw: {len(ann)} bodies, {len(w)} connections")

    # Keep real neurons: traced, with a superclass, not glia.
    ann = ann[ann.status.isin(["Traced", "Anchor"]) & ann.superclass.notna()].drop_duplicates("bodyId")
    if brain_only:
        ann = ann[~ann.superclass.str.startswith("vnc_")]
    w = w[(w.weight >= min_synapses) & (w.body_pre != w.body_post)
          & w.body_pre.isin(ann.bodyId) & w.body_post.isin(ann.bodyId)]

    # Drop neurons left with no connections at all.
    used = np.union1d(w.body_pre.values, w.body_post.values)
    ann = ann[ann.bodyId.isin(used)].reset_index(drop=True)
    ann["region"] = ann.apply(region_of, axis=1)
    ann["nt"] = transmitter(nt, ann.bodyId).values
    idx = pd.Series(np.arange(len(ann)), index=ann.bodyId)

    pre = idx[w.body_pre.values].values
    post = idx[w.body_post.values].values
    sign = ann.nt.map(SIGN).fillna(1.0).values[pre]
    # Photoreceptors are histaminergic, and in a real eye light makes the lamina cells
    # behind them fire less. A rate model starts with every neuron at zero, and you can't
    # fire less than zero, so treated as inhibition the board would never get past the
    # eye. It didn't: in the first version every neuron behind the eye stayed at exactly 0.
    sign[ann.region.values[pre] == "photoreceptors"] = 1.0
    print(f"kept: {len(ann)} neurons, {len(pre)} connections (>= {min_synapses} synapses)")

    # Where does each photoreceptor look? Photoreceptors have no column label in
    # the tables, but the lamina and medulla cells they feed do. Each R cell
    # inherits the column of its strongest labelled partner.
    hex1 = np.full(len(ann), np.nan)
    hex2 = np.full(len(ann), np.nan)
    has_hex = ann.assignedOlHex1.notna().values
    photo = np.where(ann.region.values == "photoreceptors")[0]
    e = pd.DataFrame({"pre": pre, "post": post, "w": w.weight.values})
    e = e[e.pre.isin(photo) & has_hex[e.post.values]].sort_values("w", ascending=False).drop_duplicates("pre")
    hex1[e.pre.values] = ann.assignedOlHex1.values[e.post.values]
    hex2[e.pre.values] = ann.assignedOlHex2.values[e.post.values]
    print(f"photoreceptors placed on the eye: {len(e)} of {len(photo)}")

    # Cell body positions in nanometres, for the 3D brain in the interface. Some neurons have none.
    soma = np.full((len(ann), 3), np.nan, np.float32)
    has_soma = ann.somaLocation.notna().values
    soma[has_soma] = np.stack(ann.somaLocation.values[has_soma])

    text = lambda s: s.fillna("").to_numpy(dtype=str)  # plain numpy strings, so loading needs no pickle
    np.savez_compressed(
        BRAIN,
        body_id=ann.bodyId.to_numpy(), type=text(ann.type), cls=text(ann["class"]), superclass=text(ann.superclass),
        side=text(ann.rootSide.fillna(ann.somaSide)), region=text(ann.region), nt=text(ann.nt),
        pre=pre.astype(np.int32), post=post.astype(np.int32),
        synapses=w.weight.values.astype(np.float32), sign=sign.astype(np.float32),
        hex1=hex1.astype(np.float32), hex2=hex2.astype(np.float32), soma=soma,
    )
    print(f"saved {BRAIN}")


@dataclass
class Connectome:
    body_id: np.ndarray
    type: np.ndarray
    cls: np.ndarray
    superclass: np.ndarray
    side: np.ndarray
    region: np.ndarray
    nt: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    synapses: np.ndarray
    sign: np.ndarray
    hex1: np.ndarray
    hex2: np.ndarray
    soma: np.ndarray

    @property
    def n(self) -> int:
        return len(self.body_id)

    def where(self, region: str) -> np.ndarray:
        return np.where(self.region == region)[0]

    def with_edges(self, keep: np.ndarray, post: np.ndarray | None = None) -> "Connectome":
        """Same neurons, only the connections in `keep` (optionally pointing somewhere else)."""
        return Connectome(**{**self.__dict__, "pre": self.pre[keep], "post": (self.post if post is None else post)[keep],
                             "synapses": self.synapses[keep], "sign": self.sign[keep]})

    def shuffled(self, seed: int = 0) -> "Connectome":
        """Control brain: every neuron keeps its outputs' count, signs and strengths,
        but the targets are scrambled. Same statistics, wrong wiring."""
        post = np.random.default_rng(seed).permutation(self.post)
        return self.with_edges(post != self.pre, post)

    def lesioned(self, regions: list[str]) -> "Connectome":
        """Silence whole regions by cutting every connection out of them."""
        return self.with_edges(~np.isin(self.region, regions)[self.pre])


def load(shuffle: bool = False, lesion: list[str] | None = None) -> Connectome:
    if not BRAIN.exists():
        raise SystemExit("no brain.npz yet: run scripts/download_connectome.py then python -m flychess.connectome --build")
    d = np.load(BRAIN, allow_pickle=False)
    c = Connectome(**{k: d[k] for k in d.files})
    if shuffle:
        c = c.shuffled()
    if lesion:
        c = c.lesioned(lesion)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument("--brain-only", action="store_true", help="drop the ventral nerve cord to save memory")
    args = ap.parse_args()
    if args.build:
        build(args.min_synapses, args.brain_only)
    c = load()
    print(f"{c.n} neurons, {len(c.pre)} connections")
    for r in REGIONS:
        print(f"  {r:20s} {len(c.where(r)):7d}")

"""Download the MaleCNS v1.0 flat connectome (public, no account needed).

Three Feather tables, about 1.2 GB in total, land in data/connectome/:
  body-annotations   one row per body: type, class, superclass, side
  body-neurotransmitters   predicted transmitter per body
  connectome-weights   one row per connected pair: body_pre, body_post, weight (synapse count)

Usage: python scripts/download_connectome.py
"""
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]
OUT = Path("data/connectome")


def fetch(name: str) -> None:
    dest = OUT / name
    if dest.exists():
        print(f"already have {name}")
        return
    tmp = dest.with_suffix(".part")
    with requests.get(BASE + name, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name[:40]) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name)
    print("done, now run: python -m flychess.connectome --build")

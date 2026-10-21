#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu or Fedora): Python 3.12, TensorFlow with CUDA, Stockfish.
#   bash scripts/setup_wsl.sh
# Afterwards, every new terminal:  source ~/.venvs/flychess/bin/activate
set -euo pipefail
VENV=~/.venvs/flychess                     # lives on the Linux side: pip on /mnt/c is very slow
REPO=$(cd "$(dirname "$0")/.." && pwd)

if command -v dnf >/dev/null; then sudo dnf install -y gawk tar gzip stockfish
else sudo apt-get update && sudo apt-get install -y gawk tar gzip stockfish curl; fi

export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
[ -d $VENV ] || uv venv --python 3.12 $VENV
source $VENV/bin/activate
uv pip install -e "$REPO"

# TensorFlow finds most CUDA libraries from pip by itself, but not all of them.
LINE='export LD_LIBRARY_PATH=$(ls -d "$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/*/lib | tr "\n" ":")/usr/lib/wsl/lib'
grep -qF 'nvidia/*/lib' $VENV/bin/activate || echo "$LINE" >> $VENV/bin/activate
source $VENV/bin/activate
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

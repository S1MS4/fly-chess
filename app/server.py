"""Local web interface.

  uvicorn app.server:app --port 8000     then open http://localhost:8000

  POST /move      {"fen": "...", "fly": "dl" | "bio"} -> the fly's move, its top 5, brain activity per step
  GET  /metrics   training curves from runs/*/metrics.jsonl
"""
import json
from functools import lru_cache
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flychess import connectome
from flychess.connectome import REGIONS

app = FastAPI(title="fly-chess")


@lru_cache(maxsize=1)
def brain():
    return connectome.load()


@lru_cache(maxsize=2)
def fly(kind: str):
    from flychess.play import BioPlayer, DLFly  # TensorFlow loads on first use, not at startup
    return DLFly(c=brain()) if kind == "dl" else BioPlayer(c=brain())


class MoveRequest(BaseModel):
    fen: str
    fly: str = "dl"


@app.post("/move")
def move(req: MoveRequest):
    board = chess.Board(req.fen)
    if board.is_game_over():
        raise HTTPException(400, "game is over")
    player = fly(req.fly)
    m, info = player.think(board)
    return {"move": m.uci(), "san": board.san(m), "trained": player.trained, **info}


@app.get("/regions")
def regions():
    c = brain()
    return [{"name": r, "neurons": int(len(c.where(r)))} for r in REGIONS]


@app.get("/metrics")
def metrics():
    out = {}
    for f in Path("runs").glob("*/metrics.jsonl"):
        out[f.parent.name] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

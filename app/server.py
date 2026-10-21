"""Local web interface: a chess board, the fly and its brain.

  uvicorn app.server:app --port 8000     then open http://localhost:8000

  GET  /api/flies           which flies are trained and can play
  GET  /api/brain           how many neurons have a cell body, the regions, the steps
  GET  /api/brain/points    their positions (float32 x, y, z), as bytes
  GET  /api/legal?fen=      the legal moves, each with the position it leads to and its opening
  POST /api/play            {"fen", "uci", "fly", "seen", "imagine"}: your move, then the fly's answer
  GET  /api/activity/{id}   every neuron's firing at every step of one think, as bytes
  POST /api/warm/{fly}      build a fly before its first move is needed
  GET  /api/opening?moves=  the opening's name for the game so far
  POST /api/analyse         {"fen", "moves"}: Stockfish's best move and its opinion of each move
  GET  /api/metrics         training curves from runs/*/metrics.jsonl
"""
import json
import math
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from itertools import count
from pathlib import Path

import chess
import chess.engine
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flychess import connectome
from flychess.connectome import REGIONS
from flychess.model import STEPS

DEPTH = 12          # Stockfish search depth for the accuracy numbers, about 0.2 s
SHOWN = 10          # candidate moves sent to the interface
think_lock, engine_lock = threading.Lock(), threading.Lock()
traces: OrderedDict[int, bytes] = OrderedDict()                        # the last few thinks, for the 3D brain
trace_ids = count()


@lru_cache(maxsize=1)
def brain():
    c = connectome.load()
    return c, np.where(~np.isnan(c.soma[:, 0]))[0]                     # the neurons the 3D view can place


FLY_KINDS = ("dl", "day", "bio", "quick", "shuffled", "no-optic-lobe", "no-central-brain", "newborn")  # flychess.play.FLIES
TRAINED = {"dl": "checkpoints/dl/checkpoint", "day": "checkpoints/dl-day/best/checkpoint", "bio": "checkpoints/bio/kc_mbon.npz",
           "quick": "checkpoints/probe-real/checkpoint", "shuffled": "checkpoints/probe-shuffled/checkpoint",
           "no-optic-lobe": "checkpoints/probe-no-optic-lobe/checkpoint",
           "no-central-brain": "checkpoints/probe-no-central-brain/checkpoint"}   # what each fly needs to exist


@lru_cache(maxsize=3)
def fly(kind: str):
    """One of the flies in flychess.play.FLIES. Three stay loaded, another one takes a few seconds to build."""
    from flychess.play import make_fly                                 # TensorFlow loads on first use, not at import
    return make_fly(kind, brain()[0])


@lru_cache(maxsize=len(FLY_KINDS))
def usual(kind: str) -> np.ndarray:
    """Each neuron's average rate at every step over random positions, (steps, neurons).
    The 3D brain shows the difference from this, so you see what this board changed."""
    from flychess.play import sample_boards
    return fly(kind).trace(sample_boards(64, seed=1)).mean(2)


def brain_view(kind: str, rates: np.ndarray) -> tuple[bytes, list[dict]]:
    """Bytes for the 3D brain (128 = as usual) and how much each region moved, per step."""
    c, has = brain()
    change = rates - usual(kind)
    per_region = [{r: round(float(np.abs(change[s, c.region == r]).mean()), 5) for r in REGIONS if (c.region == r).any()}
                  for s in range(len(change))]
    return np.clip(128 + change[:, has] * 255, 0, 255).astype(np.uint8).tobytes(), per_region


@lru_cache(maxsize=1)
def engine():
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "stockfish"))
    sf.configure({"Threads": max(1, (os.cpu_count() or 2) // 2), "Hash": 64})
    return sf


def warm_up(kind: str) -> None:
    """Build a fly (and its usual activity) ahead of time. Under the think lock, so two requests never build
    the same fly at once: lru_cache alone doesn't stop that, and it would cost the GPU memory twice."""
    with think_lock:
        fly(kind)
        usual(kind)


@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=warm_up, args=("dl",), daemon=True).start()  # while the page loads
    yield


app = FastAPI(title="fly-chess", lifespan=lifespan)


def describe(board: chess.Board, m: chess.Move) -> dict:
    """Everything the board needs to animate a move, and its name."""
    d = {"uci": m.uci(), "san": board.san(m), "from": chess.square_name(m.from_square),
         "to": chess.square_name(m.to_square), "piece": board.piece_at(m.from_square).symbol(),
         "promo": chess.piece_symbol(m.promotion) if m.promotion else None, "capture": None, "rook": None}
    if board.is_en_passant(m):
        d["capture"] = chess.square_name(m.to_square + (-8 if board.turn == chess.WHITE else 8))
    elif board.piece_at(m.to_square):
        d["capture"] = d["to"]
    if board.is_castling(m):
        rank, short = chess.square_rank(m.from_square), board.is_kingside_castling(m)
        d["rook"] = [chess.square_name(chess.square(7 if short else 0, rank)), chess.square_name(chess.square(5 if short else 3, rank))]
    return d


def candidates(board: chess.Board, probs) -> list[dict]:
    return [{"uci": m.uci(), "san": board.san(m), "piece": board.piece_at(m.from_square).symbol(), "p": round(p, 4)}
            for m, p in probs[:SHOWN]]


@lru_cache(maxsize=1)
def openings() -> tuple[dict, dict]:
    """Lichess's opening names (CC0, from scripts/download_openings.py), by position and by the line's moves."""
    path = Path(__file__).parent / "openings.tsv"
    by_position, by_moves = {}, {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            eco, name, moves, epd = line.split("\t")
            by_position[epd] = by_moves[moves] = (eco, name)
    return by_position, by_moves


def parse_move(board: chess.Board, uci: str) -> chess.Move | None:
    """The move, if it's a well-formed UCI string and legal here."""
    try:
        m = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return m if m in board.legal_moves else None


def position_key(board: chess.Board) -> str:
    """The same as flychess.play.position_key, without loading TensorFlow: pieces, side to move, castling, en passant."""
    return " ".join(board.fen().split()[:4])


def opening(board: chess.Board) -> dict | None:
    """The named position this board is, if any, however the game got here: what makes a move a book move."""
    hit = openings()[0].get(board.epd())
    return {"eco": hit[0], "name": hit[1]} if hit else None


def position(board: chess.Board, seen=()) -> dict:
    """What the browser needs about a position: FEN, legal moves (each with the position it leads to,
    so the board only ever draws positions from here), a king in check, the opening and the result."""
    legal = []
    for m in board.legal_moves:
        d = describe(board, m)
        board.push(m)
        d["after"], d["opening"] = board.fen(), opening(board)
        board.pop()
        legal.append(d)
    check = chess.square_name(board.king(board.turn)) if board.is_check() else None
    return {"fen": board.fen(), "legal": legal, "check": check, "opening": opening(board), "over": outcome(board, seen)}


def outcome(board: chess.Board, seen=()) -> str | None:
    """The result, or None while the game goes on. A board from a FEN has no history, so the browser
    sends the positions so far (`seen`). Draws by repetition or the 50-move rule only count once they
    have really happened, not when the next move could make them happen."""
    o = board.outcome()                                                # checkmate, stalemate, not enough material
    if o is None:
        if list(seen).count(position_key(board)) >= 2:
            return "threefold repetition, draw"
        return "fifty moves, draw" if board.halfmove_clock >= 100 else None
    who = {chess.WHITE: "white", chess.BLACK: "black", None: None}[o.winner]
    return f"{o.termination.name.lower().replace('_', ' ')}" + (f", {who} wins" if who else ", draw")


@app.get("/api/brain")
def brain_meta():
    _, has = brain()
    return {"regions": REGIONS, "count": int(len(has)), "steps": STEPS}


@app.get("/api/brain/points")
def brain_points():
    c, has = brain()
    xyz = c.soma[has].astype(np.float64)
    xyz -= (xyz.min(0) + xyz.max(0)) / 2
    xyz /= np.abs(xyz).max()                                           # fits in a -1..1 box
    return Response(xyz.astype(np.float32).tobytes(), media_type="application/octet-stream")


@app.get("/api/flies")
def flies():
    """Which flies can play: a trained one only once its checkpoint is there (the newborn never needs one)."""
    return {k: k not in TRAINED or Path(TRAINED[k]).exists() for k in FLY_KINDS}


@app.get("/api/legal")
def legal(fen: str = chess.STARTING_FEN):
    return position(chess.Board(fen))


class Play(BaseModel):
    fen: str
    uci: str | None = None       # your move, or none to let the fly start
    fly: str = "dl"
    seen: list[str] = []         # earlier positions of this game (the FEN without its move counters)
    imagine: bool = False        # a backprop fly looks one move ahead first (flychess.play.look_ahead)


@app.post("/api/play")
def play(req: Play):
    if req.fly not in FLY_KINDS:
        raise HTTPException(400, f"there's no fly called {req.fly}")
    board = chess.Board(req.fen)
    seen = list(req.seen)
    out = {"user": None, "fly": None}
    if req.uci:
        m = parse_move(board, req.uci)
        if m is None:
            raise HTTPException(400, f"{req.uci} is not legal here")
        out["user"] = describe(board, m)
        board.push(m)
    if outcome(board, seen) is None:
        if req.uci:
            seen.append(position_key(board))                           # the position your move made
        with think_lock:
            player = fly(req.fly)                                      # built under the lock, see warm_up
            ahead = {"imagine": True} if req.imagine and hasattr(player, "imagine") else {}   # not the dopamine fly
            m, info = player.think(board, seen=seen, **ahead)          # it won't walk into a repetition
            expect = None
            after = board.copy()
            after.push(m)
            if outcome(after, seen) is None:
                expect = candidates(after, player.policy(after))       # what it thinks you'll play next
            view, per_region = brain_view(req.fly, info["rates"])
        trace = next(trace_ids)
        traces[trace] = view
        while len(traces) > 6:
            traces.popitem(last=False)
        out["fly"] = {"fen": board.fen(), "move": describe(board, m), "candidates": candidates(board, info["candidates"]),
                      "value": info.get("value"), "activity": per_region, "trace": trace, "trained": player.trained,
                      "imagined": {d["uci"]: {"end": d["end"], "value": d["value"]} for d in info["imagined"]}
                      if "imagined" in info else None}                 # per move: how the game ends after it, its value
        board.push(m)
        out["expect"] = expect
    out.update(position(board, seen))
    return out


@app.post("/api/warm/{kind}")
def warm(kind: str):
    """Build a fly before its first move is needed: a lab fly takes a few seconds the first time."""
    if kind not in FLY_KINDS:
        raise HTTPException(400, f"there's no fly called {kind}")

    threading.Thread(target=warm_up, args=(kind,), daemon=True).start()
    return {"warming": kind}


@app.get("/api/opening")
def opening_name(moves: str = ""):
    """The game's opening (`moves`: its UCI moves, comma separated), named by move order like chess.com
    does: the longest named line the game follows. By position alone, 1.e4 Nf6 2.Nf3 would be called the
    Zukertort's Lemberger Gambit, because 1.Nf3 Nf6 2.e4 reaches the same position."""
    played = [m for m in moves.split(",") if m]
    by_moves = openings()[1]
    for n in range(min(len(played), 40), 0, -1):
        if hit := by_moves.get(" ".join(played[:n])):
            return {"eco": hit[0], "name": hit[1]}
    return None


@app.get("/api/activity/{trace}")
def activity(trace: int):
    if trace not in traces:
        raise HTTPException(404, "too old, think again")
    return Response(traces[trace], media_type="application/octet-stream")


def win_percent(cp: float) -> float:
    """Lichess's chance-to-win curve: centipawns (capped at 10 pawns either way) -> 0..100."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * max(-1000, min(1000, cp)))) - 1)


def accuracy(before: float, after: float) -> float:
    """Lichess's move accuracy from the win% before and after, both for the player who moved."""
    if after >= before:
        return 100.0
    return min(100.0, max(0.0, 103.1668100711649 * math.exp(-0.04354415386753951 * (before - after)) - 3.166924740191411 + 1))


def verdict(drop: float, best: bool) -> str:
    """Lichess's names for how much winning chance a move gave away."""
    return "best" if best else "good" if drop < 5 else "inaccuracy" if drop < 10 else "mistake" if drop < 15 else "blunder"


class Analyse(BaseModel):
    fen: str
    moves: list[str]


@app.post("/api/analyse")
def analyse(req: Analyse):
    """Stockfish's best move (the hint button sends no moves and only wants this), and for each move:
    its score, the accuracy Lichess would give it (100 = nothing given away), and a verdict.

    accuracy = 103.17 * exp(-0.0435 * (win% before - win% after)) - 3.17 + 1"""
    board = chess.Board(req.fen)
    moves = [m for m in (parse_move(board, u) for u in req.moves) if m]
    if board.is_game_over():
        return {"best": None, "moves": {}}
    limit = chess.engine.Limit(depth=DEPTH)
    with engine_lock:
        for attempt in range(2):
            try:
                top = engine().analyse(board, limit)
                each = engine().analyse(board, limit, multipv=len(moves), root_moves=moves) if moves else []
                break
            except chess.engine.EngineTerminatedError:                 # Stockfish died (killed, crashed): start a new one
                engine.cache_clear()
                if attempt:
                    raise
    score = lambda info: info["score"].pov(board.turn).score(mate_score=10_000)
    best_move, best_cp = top["pv"][0], score(top)
    out = {}
    for info in each:
        m, cp = info["pv"][0], score(info)
        before, after = win_percent(best_cp), win_percent(cp)
        out[m.uci()] = {"cp": cp, "accuracy": round(accuracy(before, after), 1),
                        "verdict": verdict(max(0.0, before - after), m == best_move)}
    return {"best": {"uci": best_move.uci(), "san": board.san(best_move), "cp": best_cp}, "moves": out}


@app.get("/api/metrics")
def metrics():
    out = {}
    for f in Path("runs").glob("*/metrics.jsonl"):
        if not f.parent.name.startswith(("sweep", "smoke", "probe", "pilot")):  # probes are single points, pilots short tests
            out[f.parent.name] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

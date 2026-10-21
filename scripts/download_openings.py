"""Download Lichess's opening names (CC0) for the web interface's opening line.

Saves app/openings.tsv: ECO code, name, the line's moves (UCI) and the position it ends in
(EPD). The server names a game by the longest line it follows, and calls a move a book move
when it reaches one of these positions, however the game got there.

Usage: python scripts/download_openings.py
"""
from pathlib import Path

import chess
import requests

URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{}.tsv"
OUT = Path("app/openings.tsv")


def main():
    rows = []
    for part in "abcde":
        lines = requests.get(URL.format(part), timeout=60).text.splitlines()
        for line in lines[1:]:                                            # eco, name, pgn
            eco, name, pgn = line.split("\t")
            board, moves = chess.Board(), []
            for san in pgn.split():
                if not san[0].isdigit():                                  # skip the move numbers
                    moves.append(board.push_san(san).uci())
            rows.append(f"{eco}\t{name}\t{' '.join(moves)}\t{board.epd()}")
    OUT.write_text("eco\tname\tuci\tepd\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} openings in {OUT}")


if __name__ == "__main__":
    main()

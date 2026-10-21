"""A ChessBench download that breaks halfway gets tried again instead of ending the run (one did, 7 files into
the day run), and gives up with the error after its last try, leaving no half-written shard.

  python -m pytest tests     (or just: python tests/test_download_retry.py)
"""
import importlib.util
import tempfile
from pathlib import Path

import pandas as pd
import urllib3

spec = importlib.util.spec_from_file_location("download_chessbench", Path(__file__).parent.parent / "scripts" / "download_chessbench.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
ROW = ("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2", 0.0, "a1a2", "50.0")


def flaky(breaks: int):
    """A fetch whose connection breaks the first `breaks` times."""
    calls = []

    def fetch(i):
        calls.append(i)
        if len(calls) <= breaks:
            raise urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
        return [ROW]
    return fetch, calls


def test_tries_again():
    dc.WAIT = 0
    dc.fetch, calls = flaky(2)
    with tempfile.TemporaryDirectory() as d:
        path = dc.download(13, Path(d))
        assert len(calls) == 3 and pd.read_parquet(path)["fen"].tolist() == [ROW[0]]


def test_gives_up():
    dc.WAIT = 0
    dc.fetch, calls = flaky(99)
    with tempfile.TemporaryDirectory() as d:
        try:
            dc.download(13, Path(d), tries=3)
        except urllib3.exceptions.ProtocolError:
            assert len(calls) == 3 and not list(Path(d).iterdir())
        else:
            raise AssertionError("it should have given up")


if __name__ == "__main__":
    test_tries_again()
    test_gives_up()
    print("ok")

"""How the training is going, live in the console. It only reads the logs, so open and close it whenever.

  python tools/watch.py            # from the fly-chess folder, in any terminal (Windows or WSL); Ctrl+C to leave
  python tools/watch.py --once     # print it once

Shows what the long run (scripts/train_day.sh, then scripts/train_more.sh) is doing right now with a bar,
the best fly so far, the temperatures and the heat guard, and the disk.
"""
import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
RUN = "dl-day"
WIDTH = 24
NOISE = re.compile(r"oneDNN|E0000|I0000|W0000|WARNING|rebuild|AVX|^\s|To enable|it/s|file/s")
EXTRACT = re.compile(r"^(\S+)\.npy: ([\d,]+) of ([\d,]+) positions, (\d+) a second")


def bar(share: float) -> str:
    share = max(0.0, min(1.0, share))
    full = int(share * WIDTH)
    return "[" + "█" * full + "░" * (WIDTH - full) + f"] {share:4.0%}"


def clock(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def lines(path: Path) -> list[str]:
    try:
        return [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line and not NOISE.search(line)]
    except OSError:
        return []


def active_log() -> Path | None:
    """The log the run is writing to now: the extension's once it has started, else the newest day log."""
    more = RUNS / "train_more.log"
    if any(re.match(r"\d{4}-\d\d-\d\d", line) for line in lines(more)):
        return more
    days = sorted(RUNS.glob("train_day*.log"), key=lambda p: p.stat().st_mtime)
    return days[-1] if days else None


def now_doing(log: Path) -> list[str]:
    """What the last lines of the log say is happening, as lines of the dashboard."""
    known = re.compile(r"^(\S+\.npy: |\{'epoch'|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d |\S+ vs \S+: |file \d+: |couldn't get )")
    ls = [line for line in lines(log) if known.match(line)]           # not the download helper's own chatter
    if not ls:
        return ["now                waiting for the day run to finish"]
    last = ls[-1]
    age = time.time() - log.stat().st_mtime
    if m := EXTRACT.match(last):
        name, done, total, speed = m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", "")), int(m.group(4))
        if done < 64_000:                                              # a file's first line still counts the start-up:
            settled = [int(x.group(4)) for x in map(EXTRACT.match, ls) if x and int(x.group(2).replace(",", "")) >= 64_000]
            speed = settled[-1] if settled else speed                  # the last file's speed is the better guess
        done = min(total, done + speed * age) if done < total else total   # between log lines, carry on at its speed
        what = f"ChessBench file {int(name.split('_')[1])}" if name.startswith("more_") else name
        left = (total - done) / max(speed, 1)
        return [f"now                the brain is going over {what}",
                f"this file          {bar(done / total)}   {int(done):,} of {total:,}, {speed} a second, {clock(left)} left"]
    if last.startswith("{'epoch'"):
        try:
            rec = ast.literal_eval(last)
        except (ValueError, SyntaxError):
            rec = {}
        if rec.get("epoch") == "final":
            return ["now                checking the new readout through the real brain, then saving it"]
        return [f"now                fitting the readout ({rec.get('fit', '')}), pass {rec.get('epoch')}",
                f"this fit           {rec.get('minutes', 0):.0f} min so far, on {rec.get('positions', 0) / 1e6:.1f} M positions: "
                f"loss {rec.get('held_out_loss', 0):.3f}, top-1 {100 * rec.get('top1', 0):.1f}% (stops when the loss stops falling)"]
    if " vs " in last and "%" in last:
        return [f"now                Elo games: {last.split('|')[0].strip()}"]
    if m := re.match(r"file (\d+): (\S+), trying again in (\d+) s", last):
        return [f"now                the download of ChessBench file {m.group(1)} broke ({m.group(2)}), trying again in {m.group(3)} s"]
    if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", last):
        return [f"now                {last[20:]}"]
    return [f"now                {last[:90]}"]


def best_fly() -> str:
    try:
        finals = [json.loads(line) for line in (RUNS / RUN / "metrics.jsonl").read_text().splitlines() if '"final"' in line]
    except OSError:
        finals = []
    if not finals:
        return "best fly so far    none saved yet"
    f = finals[-1]
    saved = Path(RUNS.parent / "checkpoints" / RUN / "best" / "checkpoint")
    when = time.strftime("%a %H:%M", time.localtime(saved.stat().st_mtime)) if saved.exists() else "?"
    return (f"best fly so far    top-1 {100 * f['top1']:.1f}%, good moves {100 * f['good_moves']:.1f}%, loss {f['held_out_loss']:.3f}, "
            f"finishing {100 * f.get('finish_top1', 0):.1f}%\n"
            f"                   saved {when}, learned from {f['positions'] / 1e6:.1f} M positions")


def heat() -> str:
    try:
        rows = (RUNS / "heat.csv").read_text().splitlines()[1:]
    except OSError:
        return "heat               no readings (is tools/heat_guard.py running?)"
    if not rows:
        return "heat               no readings yet"
    last = rows[-1].split(",")
    hour = [r.split(",") for r in rows[-180:]]                         # one reading every 20 s: the last hour
    paused = sum(r[5] == "1" for r in hour) / max(len(hour), 1)
    stale = time.time() - time.mktime(time.strptime(last[0], "%Y-%m-%d %H:%M:%S")) > 120
    duty = float(last[6]) if len(last) > 6 else 1.0                   # the share of each second the jobs may run
    state = ("PAUSED to cool down" if last[5] == "1" else
             f"running at {duty:.0%} to stay cool" if duty < 1 else "running flat out")
    gpu = f"{float(last[1]):.0f}" if last[1] not in ("", "None") else "?"
    return (f"heat               GPU {gpu}°C, CPU {last[4] or '?'}°C, {state}; paused {paused:.0%} of the last hour"
            + ("   (no reading for 2 min: the guard may have stopped)" if stale else ""))


def disk() -> str:
    """The cache's size and the free space where it really is (runs/<run>/cache can be a junction to D:)."""
    cache = RUNS / RUN / "cache"
    try:
        size = sum(f.stat().st_size for f in cache.iterdir() if f.is_file()) / 1e9
        free = shutil.disk_usage(cache).free / 1e9
    except OSError:
        return "disk               no cache yet"
    real = os.path.realpath(cache)
    where = f"{real[5].upper()}:" if real.startswith("/mnt/") else real[:2]   # /mnt/d/... in WSL, D:\... on Windows
    try:
        cap = f"past {float((RUNS / RUN / 'max-cache-gb').read_text()):.0f} GB or "
    except (OSError, ValueError):
        cap = "at "
    return (f"disk               cache {size:.0f} GB on {where}, {free:.0f} GB free there; {cap}25 GB free "
            f"the oldest file makes room")


def files_added() -> str:
    cache = RUNS / RUN / "cache"
    n = len(list(cache.glob("more_*_t.npz")))
    return f"files added        {n} ChessBench files, about {n * 0.515 + 3.09:.1f} M positions in all"


def overall() -> list[str]:
    try:
        start = int((RUNS / RUN / "started").read_text())
    except (OSError, ValueError):
        return ["the whole run      not started"]
    if (done := RUNS / RUN / "more-done").exists():
        return [f"the whole run      stopped as you asked, after {clock(done.stat().st_mtime - start)}"]
    elapsed = clock(time.time() - start)
    if (RUNS / RUN / "stop").exists():
        return [f"the whole run      {elapsed} so far; stopping as you asked: this file, a fit, its Elo, then it ends"]
    out = [f"the whole run      {elapsed} so far, and no end: it keeps going until you stop it (touch runs/{RUN}/stop)"]
    f = RUNS / RUN / "cycle-ends"                                      # train_more.sh writes it as a cycle starts
    try:
        end, begun = int(f.read_text()), f.stat().st_mtime
    except (OSError, ValueError):
        return out
    found = [m.group(1) for line in lines(RUNS / "train_more.log") if (m := re.search(r" cycle (\d+)\b", line))]
    n = found[-1] if found else "?"
    if time.time() < end:
        return out + [f"this cycle ({n})     {bar((time.time() - begun) / (end - begun))}   new files until "
                      f"{time.strftime('%a %H:%M', time.localtime(end))}, then a fit on everything and its Elo"]
    return out + [f"this cycle ({n})     {bar(1)}   the new files are in: fitting on everything, then its Elo"]


def screen() -> str:
    log = active_log()
    out = [f"fly-chess: training the day-long fly{'':>34}{time.strftime('%a %H:%M:%S')}", "", *overall()]
    if (RUNS / RUN / "more-done").exists():
        out.append("now                nothing: it stopped, as you asked")
    elif log is not None:
        out += now_doing(log)
        quiet = time.time() - log.stat().st_mtime
        if quiet > 45 * 60:                                            # its slowest step writes a line every 15 min or so
            out.append(f"WARNING            nothing new in {log.name} for {clock(quiet)}: it may be stuck")
    stopped = (RUNS / RUN / "more-done").exists()
    out += [files_added(), best_fly(), "heat               nothing is running, the heat guard is off" if stopped else heat(), disk(), "",
            "Ctrl+C leaves this." if stopped else "Ctrl+C leaves this; the training keeps going."]
    return "\n".join(out)


def escape_codes_on() -> None:
    """Windows consoles only move the cursor on escape codes once they're told to."""
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        out, mode = k.GetStdHandle(-11), ctypes.c_uint32()
        if k.GetConsoleMode(out, ctypes.byref(mode)):
            k.SetConsoleMode(out, mode.value | 0x0004)                 # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=5, help="seconds between refreshes")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")                           # the bars, even when Windows pipes it as cp1252
    if args.once:
        print(screen())
        return
    escape_codes_on()
    drawn = 0
    sys.stdout.write("\x1b[?25l")                                      # hide the cursor while it redraws
    try:
        while True:
            width = shutil.get_terminal_size().columns - 1             # a wrapped line would throw the redraw off
            rows = [row[:width] for row in screen().split("\n")]
            up = f"\x1b[{drawn}F" if drawn else ""                     # back to the first line drawn last time
            sys.stdout.write(up + "".join(row + "\x1b[K\n" for row in rows) + "\x1b[J")   # overwrite, clear what's left
            sys.stdout.flush()
            drawn = len(rows)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")                                # the cursor back


if __name__ == "__main__":
    main()

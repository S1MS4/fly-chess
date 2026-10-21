"""Keeps the laptop from cooking itself during long runs: throttles the heavy jobs to hold a steady temperature,
pauses them outright if it still gets too hot, and stops the biggest one if WSL is about to run out of memory
(that took the whole of WSL down once, every job with it).

  python tools/heat_guard.py &           # in WSL, next to the training (scripts/train_day.sh starts it)

The throttle: every training, tool and Stockfish process runs `duty` of every second and gets SIGSTOP for
the rest, so the GPU and CPU work at that share of flat out. Every 20 seconds it reads the GPU's temperature
(nvidia-smi) and the hottest of Windows' thermal zones (tools/windows_helper.py passes that one over from
Windows; on my Lenovo it sits right on the CPU), and moves the duty down when either is above its target and
slowly back up when both are well under. So the laptop settles at a steady temperature instead of swinging
between flat out and paused. A number in runs/duty.txt holds the duty there instead (for measuring).

The brake: if either still reaches its stop line, everything pauses until both are back under the resume line.
Every reading goes to runs/heat.csv.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PATTERNS = ("flychess", "tools/", "scripts/make_curriculum", "stockfish")   # what counts as a heavy job
LOG = Path("runs/heat.csv")
DUTY = Path("runs/duty.txt")


def gpu_temp() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        t, p, u = (float(x) for x in out.strip().split(",")[:3])
        return t, p, u
    except Exception:
        return None


def cpu_temp() -> float | None:
    """The hottest ACPI thermal zone. WSL can't see the sensors (or run Windows programs here), so
    tools/windows_helper.py writes it to runs/zone.txt; a reading older than 90 seconds doesn't count."""
    f = Path("runs/zone.txt")
    try:
        return float(f.read_text()) if time.time() - f.stat().st_mtime < 90 else None
    except (OSError, ValueError):
        return None


def available_mb() -> float:
    """WSL's MemAvailable. Running out takes the whole WSL machine down, every process in it."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024
    return float("inf")


def biggest(pids: list[int]) -> int | None:
    def rss(pid):
        try:
            return int((Path("/proc") / str(pid) / "statm").read_text().split()[1])
        except (OSError, IndexError):
            return 0
    return max(pids, key=rss) if pids else None


def heavy_pids(patterns=PATTERNS) -> list[int]:
    me, out = os.getpid(), []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if any(s in cmd for s in patterns) and "heat_guard" not in cmd:
            out.append(int(p.name))
    return out


def send(pids, sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def throttle(duty: float, temps, targets, band: float = 3.0, low: float = 0.3) -> float:
    """The next duty: down 0.03 per degree the hottest one is over its target (0.15 at most), up 0.02 when
    every one is `band` degrees under its own."""
    over = [t - target for t, target in zip(temps, targets) if t is not None and target is not None]
    if not over:
        return duty
    if max(over) > 0:
        return max(low, duty - min(0.15, 0.03 * max(over)))
    if max(over) < -band:
        return min(1.0, duty + 0.02)
    return duty


def fixed_duty() -> float | None:
    try:
        return min(1.0, max(0.05, float(DUTY.read_text())))
    except (OSError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-target", type=float, default=75, help="the throttle holds the GPU around here")
    ap.add_argument("--cpu-target", type=float, default=85, help="and the thermal zone around here")
    ap.add_argument("--min-duty", type=float, default=0.3, help="the throttle never goes under this share")
    ap.add_argument("--period", type=float, default=1.0, help="seconds per throttle cycle")
    ap.add_argument("--gpu-stop", type=float, default=84, help="pause at this GPU temperature (its own limit is about 86)")
    ap.add_argument("--gpu-go", type=float, default=76)
    ap.add_argument("--cpu-stop", type=float, default=95, help="pause at this thermal zone temperature (the Ryzen's limit is 100)")
    ap.add_argument("--cpu-go", type=float, default=85)
    ap.add_argument("--every", type=float, default=20)
    ap.add_argument("--min-memory", type=float, default=700, help="MB of WSL memory under which the biggest job gets stopped")
    ap.add_argument("--patterns", nargs="+", default=list(PATTERNS), help=argparse.SUPPRESS)   # a test's own dummy job
    args = ap.parse_args()
    heavy = lambda: heavy_pids(args.patterns)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(0))                     # so the finally below runs when it's stopped
    new = not LOG.exists()
    log = open(LOG, "a")
    if new:
        log.write("time,gpu_c,gpu_w,gpu_util,zone_c,paused,duty\n")
    send(heavy(), signal.SIGCONT)                                      # anything an earlier guard left stopped
    paused: list[int] = []
    duty, smooth, shown, next_read = 1.0, [None, None], 1.0, 0.0
    try:
        while True:
            if time.time() >= next_read:
                next_read = time.time() + args.every
                if available_mb() < args.min_memory and (pid := biggest(heavy())):   # better one job than all of WSL
                    os.kill(pid, signal.SIGTERM)
                    print(f"{time.strftime('%H:%M:%S')} only {available_mb():.0f} MB left: stopped {pid}", flush=True)
                g, z = gpu_temp(), cpu_temp()
                gt = g[0] if g else None
                smooth = [t if s is None or t is None else (s + t) / 2 for s, t in zip(smooth, (gt, z))]  # the zone jumps 3 C between readings
                hot = (gt is not None and gt >= args.gpu_stop) or (z is not None and z >= args.cpu_stop)
                cool = (gt is None or gt <= args.gpu_go) and (z is None or z <= args.cpu_go)
                if hot and not paused:
                    paused = heavy()
                    send(paused, signal.SIGSTOP)
                    duty = max(args.min_duty, duty - 0.2)                  # flat out was too much: start lower after
                    print(f"{time.strftime('%H:%M:%S')} hot (GPU {gt}, zone {z}): paused {len(paused)} processes", flush=True)
                elif paused and cool:
                    send(paused, signal.SIGCONT)
                    print(f"{time.strftime('%H:%M:%S')} cool again (GPU {gt}, zone {z}): resumed at {duty:.0%}", flush=True)
                    paused = []
                elif not paused:
                    was = duty
                    duty = fixed_duty() or throttle(duty, smooth, (args.gpu_target, args.cpu_target), low=args.min_duty)
                    if z is None and fixed_duty() is None:                 # no CPU reading (the Windows helper is gone):
                        duty = min(duty, was)                              # slow down if the GPU says so, never speed up
                if abs(duty - shown) >= 0.1:
                    print(f"{time.strftime('%H:%M:%S')} GPU {gt}, zone {z}: running {duty:.0%} of the time", flush=True)
                    shown = duty
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{gt},{g[1] if g else ''},{g[2] if g else ''},"
                          f"{'' if z is None else round(z, 1)},{int(bool(paused))},{duty:.2f}\n")
                log.flush()
            if paused or duty >= 1:
                time.sleep(max(0.05, min(args.period, next_read - time.time())))
                continue
            pids = heavy()
            send(pids, signal.SIGSTOP)
            time.sleep(args.period * (1 - duty))
            send(pids, signal.SIGCONT)
            time.sleep(args.period * duty)
    finally:
        send(heavy(), signal.SIGCONT)                                  # never leave a job stopped behind


if __name__ == "__main__":
    main()

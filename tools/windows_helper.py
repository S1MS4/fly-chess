"""The Windows half of a long run: keeps the laptop awake and tells WSL how hot it is.

  pythonw tools\\windows_helper.py --hours 30          (on Windows, scripts/train_day.sh says how)

WSL can't read the laptop's temperature sensors and can't keep Windows from going to sleep, so this
runs on Windows next to the training:
  - while it runs, Windows won't go to sleep by itself (SetThreadExecutionState, the call video players
    use). No setting changes: when it stops, sleep works as before. Closing the lid still sleeps.
  - every 15 seconds it writes the hottest ACPI thermal zone (in C) to runs/zone.txt, which
    tools/heat_guard.py reads on the WSL side.
It stops after --hours, or as soon as the file runs/stop-helper exists. Anything that goes wrong goes to
runs/windows_helper.log (pythonw has nowhere else to print), and it keeps going.
"""
import argparse
import ctypes
import subprocess
import time
import traceback
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
RUNS = Path(__file__).resolve().parent.parent / "runs"


def zone_temp() -> float | None:
    try:
        out = subprocess.run(["typeperf", r"\Thermal Zone Information(*)\Temperature", "-sc", "1"], capture_output=True,
                             text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        rows = [line for line in out.splitlines() if line.startswith('"')]   # a header row, then the numbers (Kelvin)
        values = [float(v.strip().strip('"')) for v in rows[1].split(",")[1:] if v.strip().strip('"')]
        return max(values) - 273.15 if values else None
    except Exception:
        return None


def log(text: str) -> None:
    with open(RUNS / "windows_helper.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=30)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    stop, end = RUNS / "stop-helper", time.time() + args.hours * 3600
    stop.unlink(missing_ok=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log(f"started, for {args.hours:g} hours")
    try:
        while time.time() < end and not stop.exists():
            t = zone_temp()
            if t is not None:
                try:
                    tmp = RUNS / "zone.tmp"
                    tmp.write_text(f"{t:.1f}")
                    tmp.replace(RUNS / "zone.txt")
                except OSError as err:                                 # WSL had zone.txt open right then (it killed
                    log(f"couldn't write zone.txt, next time: {err}")  # the helper once, 2026-10-21 15:02)
            time.sleep(15)
    except BaseException:
        log("stopped by an error:\n" + traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)   # sleep as usual again
        log("stopped")


if __name__ == "__main__":
    main()

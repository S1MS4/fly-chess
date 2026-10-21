#!/bin/bash
# Keeps the day-long fly learning after scripts/train_day.sh until you tell it to stop, in cycles: new ChessBench
# files for CYCLE_HOURS (8 by default), then the readout fitted on everything and saved, then its Elo without and
# with one-move imagination (tools/imagine_lab.py tunes it first), again and again. When the cache passes MAX_CACHE_GB (100 by default) or its disk gets down to 25 GB free, the oldest
# cached file makes room for the next. It waits for train_day.sh to finish first, so it can be started at any time.
# On my laptop runs/dl-day/cache and data/positions_more are junctions to D:\fly-chess, the second SSD:
#   mklink /J runs\dl-day\cache D:\fly-chess\dl-day-cache      (in cmd, after moving the folder there)
#
#   setsid nohup bash scripts/train_more.sh >> runs/train_more.log 2>&1 &      # in WSL
#   pythonw tools\windows_helper.py --hours 1000                              # on Windows, so it doesn't sleep
#
# Stop:   touch runs/dl-day/stop      (it finishes the file it's on, fits, saves, plays its Elo games, then stops)
#         or at once: pkill -f train_more.sh; pkill -f flychess      (running it again carries on where it was)
set -u
cd "$(dirname "$0")/.."
source ~/.venvs/flychess/bin/activate
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN=${FLY_RUN:-dl-day}                                   # not NAME: WSL sets that to the computer's name
CYCLE=${CYCLE_HOURS:-8}
MAX_CACHE=${MAX_CACHE_GB:-100}
FLAGS="--eye fair --calm-unreached --read 9 10 11 12 --curriculum 0.1"
say() { echo "$(date '+%F %T') $*"; }

until [ -f "runs/$RUN/done" ]; do sleep 60; done          # the day run, its fit and its rating come first
rm -f "runs/$RUN/more-done"
pgrep -f "[h]eat_guard.py" > /dev/null || { setsid nohup python tools/heat_guard.py --cpu-stop 93 --cpu-go 84 >> runs/heat_guard.log 2>&1 < /dev/null & }

cycle=1
until [ -f "runs/$RUN/stop" ]; do
    if pgrep -f "[f]lychess.train_cached --name $RUN" > /dev/null; then   # this script was restarted mid-cycle
        say "cycle $cycle is already under way: waiting for it"
        while pgrep -f "[f]lychess.train_cached --name $RUN" > /dev/null; do sleep 60; done
    else
        say "cycle $cycle: new ChessBench files for $CYCLE hours, then a fit on everything"
        echo $(( $(date +%s) + CYCLE * 3600 )) > "runs/$RUN/cycle-ends"  # for tools/watch.py
        echo "$MAX_CACHE" > "runs/$RUN/max-cache-gb"
        if ! python -m flychess.train_cached --name "$RUN" $FLAGS --more-hours "$CYCLE" --refit-hours 1000 --skip-first-fit \
                --max-cache-gb "$MAX_CACHE"; then
            say "train_cached failed, trying again in 10 minutes"     # it picks up where it was
            sleep 600
            continue
        fi
    fi
    say "imagination: how much to trust the value readout"
    timeout 1h python tools/imagine_lab.py --fly day                    # about 6 min on the 4,096 held-out positions
    say "its Elo, without and with imagination"
    timeout 3h python -m flychess.rating --games 20 --flies day day-imagine --fresh   # about 40 min; a hang can't hold up the rest
    python - "$RUN" <<'PY'                                             # both Elos after every cycle, for the README later
import json, sys, time
run = sys.argv[1]
fit = [json.loads(l) for l in open(f"runs/{run}/metrics.jsonl") if '"final"' in l][-1]
r = json.load(open("app/static/ratings.json"))["ratings"]
im = json.load(open("checkpoints/dl-day/best/imagine.json"))
rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "positions": fit["positions"], "top1": fit["top1"],
       "held_out_loss": fit["held_out_loss"], "elo": r.get("day", {}).get("elo"), "low": r.get("day", {}).get("low"),
       "high": r.get("day", {}).get("high"), "elo_imagine": r.get("day-imagine", {}).get("elo"),
       "imagine_low": r.get("day-imagine", {}).get("low"), "imagine_high": r.get("day-imagine", {}).get("high"),
       "imagine_weight": im["weight"]}
open(f"runs/{run}/elo.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec, flush=True)
PY
    cycle=$((cycle + 1))
done
say "stopping, as asked: the pictures"
python tools/readme_figures.py
rm -f "runs/$RUN/stop" "runs/$RUN/cycle-ends"
touch "runs/$RUN/more-done"
say "done"

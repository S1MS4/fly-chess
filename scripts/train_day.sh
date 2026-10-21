#!/bin/bash
# A day of training on my laptop, then the new fly's Elo and the pictures. Kill it whenever, run it again,
# and it carries on: every step below picks up where it stopped.
#
#   setsid nohup bash scripts/train_day.sh > runs/train_day.log 2>&1 &      # in WSL
#   pythonw tools\windows_helper.py --hours 30                              # on Windows: no sleep, temperatures
#
# Stop:   pkill -f train_day.sh; pkill -f flychess       (a rerun loses at most a few minutes)
# Watch:  tail -f runs/train_day.log, runs/heat.csv, or Training in the app
#
#   0. a quick test of the whole path on 8,192 positions, so a mistake shows now and not hours later
#   A. every downloaded position through the frozen brain once, then the readout fitted on that
#      (train_cached.py): the first day-long fly after about 3 hours
#   P. three 15-minute pilots of the brain learning on a leash, against the readout alone (README, item 5)
#   B. more ChessBench files through the brain until the day is nearly over, the readout fitted again
#      on everything every 6 hours and at the end
#   then its Elo on the ladder and the README pictures
#
# tools/heat_guard.py pauses everything heavy when the GPU or the CPU gets too hot; a pause makes the run
# take longer, it never counts as training.
set -u
cd "$(dirname "$0")/.."
source ~/.venvs/flychess/bin/activate
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN=${FLY_RUN:-dl-day}                                   # not NAME: WSL sets that to the computer's name
HOURS=${HOURS:-24}
FLAGS="--eye fair --calm-unreached --read 9 10 11 12 --curriculum 0.1"
start=$(date +%s)
[ -f "runs/$RUN/started" ] && start=$(cat "runs/$RUN/started")      # a rerun keeps the day it started
mkdir -p "runs/$RUN" && echo "$start" > "runs/$RUN/started"
say() { echo "$(date '+%F %T') $*"; }

pgrep -f "[h]eat_guard.py" > /dev/null || { setsid nohup python tools/heat_guard.py >> runs/heat_guard.log 2>&1 < /dev/null & }
python -m flychess.train_dl --prepare || exit 1                       # packs the curriculum (the positions already are)

if [ ! -f "runs/$RUN/smoke-ok" ]; then
    say "0. quick test"
    python -m flychess.train_cached --name smoke-cached --limit 8192 --max-epochs 2 $FLAGS || exit 1
    rm -rf runs/smoke-cached checkpoints/smoke-cached
    touch "runs/$RUN/smoke-ok"
fi

say "A. the brain over every downloaded position, then the readout"
python -m flychess.train_cached --name "$RUN" $FLAGS || exit 1

COMMON="--minutes 15 --eye fair --seed 0 --eval-every 250 --calm-unreached"
pilot() {                                                             # name, then train_dl flags
    local name=$1; shift
    [ -f "runs/$name/done" ] && return
    say "P. pilot $name"
    python -m flychess.train_dl $COMMON --name "$name" "$@" && touch "runs/$name/done"
}
pilot pilot-readout
pilot pilot-neurons --brain-opt sgd --lr-neurons 0.02
pilot pilot-synapses --brain-opt sgd --lr-neurons 0.02 --lr-synapses 0.1 --max-change 0.2

left=$(( (start + HOURS * 3600 - $(date +%s)) / 3600 - 1 ))           # an hour stays for the last fit and the rating
if [ "$left" -gt 0 ]; then
    say "B. more ChessBench files for $left hours"
    python -m flychess.train_cached --name "$RUN" $FLAGS --more-hours "$left" || exit 1
fi

say "the new fly's Elo"
timeout 2h python -m flychess.rating --games 20 --flies day        # about 20 min; a hang can't hold up the rest
say "pictures"
python tools/readme_figures.py
python tools/readme_figures.py plasticity
touch "runs/$RUN/done"
say "done"

# Results: the training history

Everything the long training run and its experiments wrote down, copied out of `runs/` on my laptop (which git ignores). Only the small files are here: the logs, the metrics and the experiment results, 727 KB in all. The brain-activity caches (40 GB) aren't. The main README tells the story; this folder has the numbers behind it.

## The results in one table

All accuracy numbers are on the same 4,096 held-out positions that no fly ever trains on.

| Stage | When | Positions | Top-1 | Good moves | Held-out loss | Fastest win | Elo |
|---|---|---|---|---|---|---|---|
| Backprop fly, 25 minutes | before | 0.5M | 19.9% | 29.2% | 2.963 | | 653 |
| New recipe, first fit | Tue 2026-10-20 05:57 | 3.0M | 23.0% | 33.1% | 2.868 | 13.2% | |
| 7 more ChessBench files | Tue 11:15 | 6.6M | 23.1% | 33.7% | 2.853 | 13.2% | 788 |
| Cycle 1, the old fit (stalled) | Wed 2026-10-21 06:51 | 17.4M | 22.9% | 33.3% | 2.853 | 14.2% | 752 |
| **Cycle 2, the fixed fit** | Wed 18:29 | **26.2M** | **23.5%** | **34.3%** | **2.840** | **14.4%** | **792** |
| the same fly + one-move imagination | Wed 18:53 | 26.2M | | | | | **995** |

Top-1: the fly's favourite move is Stockfish's best. Good move: within 1 win% of Stockfish's best. Fastest win: picks the quickest win in won positions. Elo is on a ladder anchored at Stockfish 1320, 20 games per pairing, about ±70 points; each round refits everybody, so the ladder moves 10 to 30 points between rounds.

## Three things worth saying about how it went

**1. I didn't use the most efficient methods, on purpose.** The question is whether a real fly brain could learn chess, so the fly sees the board once and its own neurons answer. No alpha-beta, no minimax, no Monte Carlo tree search: any of them would add hundreds of Elo, and then the search would be playing chess, not the fly. No hidden layers in the readout either: a deep readout would learn chess by itself and turn the brain into a fancy input layer. The one exception is imagination, one move ahead and never the replies, as a switch that's off by default and rated on its own. Looking 2 or 3 moves ahead would cost about 900 and 27,000 brain passes a move, and that's a search. ([Main README: what I'm actually testing](../README.md#fly-chess), [why I did it this way](../README.md#why-i-didnt-use-the-most-efficient-methods).)

**2. The first day of training mostly stalled.** The first night, the readout fit put 4 GB on the GPU at once: a crash at 04:46 and another at 04:54 that took the whole WSL machine down. At 11:11 a download broke and my code stopped adding data 14 hours early; at 11:18 the Elo games crashed on a heat-guard pause and couldn't even exit, and the laptop sat idle for 9 hours. Of the 18 hours the first day was meant to spend adding data, it got 4. The second day's cycle also stalled at first (22.9% top-1 on 2.6 times the data), until I fixed the fit (back to the best readout and half the step after a pass that doesn't help): then **22.9% → 23.5% top-1, 33.3% → 34.3% good moves, 2.853 → 2.840 loss, 752 → 792 Elo**. Logs: `logs/train_day_restart2.log`, `logs/train_more.log`.

**3. The laptop overheated, so I built safeguards.** Eight Stockfish workers next to the GPU job took the CPU to **98.9°C** (the Ryzen's limit is 100). What I built, all in the repo:

- a **heat guard** (`tools/heat_guard.py`) that first paused everything at 93°C (stop-and-go: it cost a third of the time), then became a **throttle** that runs the training a share of every second to hold the CPU near 85°C and the GPU near 75°C: 318 positions a second instead of 265, and no pauses (`heat/heat.csv`, `heat/heat_before_throttle.csv`, `logs/heat_guard.log`);
- a **Windows helper** (`tools/windows_helper.py`) that reads the CPU's temperature for WSL and keeps the laptop from sleeping; it died once over a file race, and now retries and logs (`logs/windows_helper.log`);
- a **memory guard** that stops the biggest job before WSL runs out of memory, and **timeouts and retries** that survive the heat guard's pauses;
- a **watcher** that checked the run every 30 minutes and on every warning, and caught the helper's crash within 4 minutes.

Part of the heat wasn't the training at all: an NVIDIA App process had been spinning a whole CPU core for a day. With it stopped, the CPU went from about 90°C to the low 80s at the same load.

## The confusion matrix

`confusion.json`, drawn as [`docs/img/confusion.png`](../docs/img/confusion.png), made by `tools/confusion.py`. A chess move has thousands of possible classes (every from and to square), too many for one matrix, so the classes are **the piece that moves**: pawn, knight, bishop, rook, queen, king. A row is the piece Stockfish's best move moves, a column the piece the fly's favourite move moves, on the 4,096 held-out positions, for the day-long fly without imagination.

- The fly moves the right piece **49.2%** of the time, and makes exactly Stockfish's move **23.6%** of the time (the right piece and the right square).
- By class: pawn 43%, knight 47%, bishop 49%, rook 51%, queen 51%, king 57%.
- The mistakes spread over all the other pieces fairly evenly; there's no one confusion it keeps making.

(The eye pictures in the README, `eye_coverage.png` and `eye.png`, are heatmaps of which squares the eye sees, not confusion matrices.)

## What's in each file

| File | What it is |
|---|---|
| `dl-day/metrics.jsonl` | The day-long fly: one line per pass of every fit (VAL loss, top-1, good moves, finishing), and an `"epoch": "final"` line per fit measured on the 4,096 HOLD positions. The four final lines are the table above. |
| `dl-day/config.json` | The day-long fly's settings: the fair eye, reading steps 9 to 12, the curriculum. |
| `dl-day/elo.jsonl` | The Elo after every training cycle, without and with imagination, and the imagination weight. |
| `imagine/day.jsonl` | Imagination's tuning: for each weight, top-1, good moves and win% lost on the held-out positions. |
| `confusion.json` | The confusion matrix above, with its classes. |
| `dl/metrics.jsonl` | The 25-minute backprop fly's training curve. |
| `pilot-readout`, `pilot-neurons`, `pilot-synapses` | Three 15-minute pilots: readout only, the neurons' gain and bias learning too, the synapses learning too. Every brain-learning version got worse. |
| `probe-*` | The 3-minute readout probes on the real, shuffled and lesioned brains (the control experiments) and the two eyes. |
| `readout-lab/fair.json` | The learning curve (25k to 1M positions) and reading step 12 against steps 9 to 12. |
| `bio/metrics.jsonl`, `bio-pilots/results.json`, `bio-rules/results.json` | The dopamine fly and its learning-rule experiments. |
| `ratings-lab.json`, `draws-dl.json` | Elo experiments and the draw analysis that started the finishing lessons. The live ratings with every game are in `../app/static/ratings.json`. |
| `bench.json` | Speed measurements: positions a second against batch size. |
| `logs/` | The run's own logs: the first day (`train_day*.log`), the cycles (`train_more.log`), the heat guard, the Windows helper, and the experiments I stopped (`night_queue.log`, `stalemates.log`). |
| `heat/` | A temperature reading every 20 seconds: GPU °C and watts, the CPU's hottest zone, whether the guard paused, and the throttle's share (`duty`). |

## Running the fly yourself

The trained fly is about 76 MB, so it's on the [GitHub releases page](https://github.com/S1MS4/fly-chess/releases), not in git: `checkpoints/dl-day/best/` (its readout), `runs/dl-day/config.json` and `data/connectome/brain.npz` (the connectome, packed). Unzip it in the repo's folder, keeping the paths, then `uvicorn app.server:app --port 8000` and pick the day-long fly; the Imagination switch is under "I play as". It needs TensorFlow; with a GPU a move takes a fraction of a second, on the CPU a few seconds (the README's quick start says how to set it up).

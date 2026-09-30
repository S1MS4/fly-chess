# fly-chess

**A real fruit fly brain, neuron for neuron, learning to play chess.**

In 2026 Janelia, Cambridge and Google Research published the MaleCNS connectome: every neuron in a male fruit fly's brain and nerve cord, and every synapse between them, traced from electron microscope images. About 166,000 neurons. Tens of millions of synapses. It is a complete wiring diagram of an animal that can fly, court, fight and remember which smell got it punished yesterday.

This project takes that wiring diagram as it is and asks it to play chess. The board is shown to the fly's eye. The signal spreads through the real connections for a few steps. The neurons that would normally tell the legs to walk are read out as a move.

There are two versions, and comparing them is the point:

- **Deep learning version.** Every connection keeps its place and its sign (excite or inhibit), but its strength is trained with backprop in TensorFlow on millions of Lichess positions. This is how a computer would teach a brain.
- **Biological version.** The brain stays frozen exactly as measured, except for the one place a real fly stores what it learns: the mushroom body. It learns by playing games and getting dopamine, the way a fly learns that a smell means food. No backprop at all.

This README is the whole course. It goes from what a neuron is to how backprop works to what the fly is doing while it thinks about your knight. The code is short enough to read alongside it.

> Honest expectation up front: this will not beat you if you play chess at all. The best similar project so far lost to Stockfish's weakest setting (about 1320 Elo). The interesting question is not "how strong", it is "can a brain wired for flying learn this at all, and does its real wiring help".

---

## Contents

1. [Quick start](#quick-start)
2. [How a fly brain works](#how-a-fly-brain-works)
3. [The connectome](#the-connectome)
4. [Neurons in play: from board to move](#neurons-in-play-from-board-to-move)
5. [The model in TensorFlow](#the-model-in-tensorflow)
6. [The data](#the-data)
7. [How backprop works](#how-backprop-works)
8. [Training on a laptop GPU](#training-on-a-laptop-gpu)
9. [The biological version](#the-biological-version)
10. [Measuring strength](#measuring-strength)
11. [The interface](#the-interface)
12. [What to expect](#what-to-expect)
13. [Project layout](#project-layout)
14. [References](#references)

---

## Quick start

Written for a Windows laptop with an NVIDIA card (built on a Lenovo LOQ with an RTX 4060, 8 GB). TensorFlow stopped supporting the GPU on native Windows after version 2.10, so everything runs inside WSL2 (Linux inside Windows). On a Linux machine skip the first step.

**1. WSL2 and the NVIDIA driver.** In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-22.04
```

Install the normal NVIDIA Windows driver. WSL2 picks the GPU up from it, you do not install a driver inside Linux.

**2. Python and the project.** Inside Ubuntu:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip stockfish git
git clone https://github.com/S1MS4/fly-chess.git && cd fly-chess
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The last line should print one GPU. If it prints `[]`, the driver or WSL2 is the problem, not this code.

**3. Get the data.** About 1.2 GB for the brain, and the chess positions are streamed so only what you keep is saved.

```bash
python scripts/download_connectome.py
python -m flychess.connectome --build            # add --brain-only if memory is tight
python scripts/download_lichess.py --positions 10000000
python -m flychess.train_dl --prepare
```

**4. Train.**

```bash
python -m flychess.train_dl                      # deep learning version
python -m flychess.train_bio --games 5000        # biological version
```

Both can be stopped with Ctrl+C and resumed by running the same command again.

**5. Play it and watch it think.**

```bash
uvicorn app.server:app --port 8000
```

Open http://localhost:8000 in the Windows browser.

---

## How a fly brain works

A neuron is a cell that adds up signals from other neurons and, if the total is high enough, sends its own signal on. The connections are synapses. Some synapses excite (push the next neuron towards firing), some inhibit (push it away). Everything a brain does comes from which neurons connect to which, how strongly, and with what sign.

A fruit fly has about 140,000 neurons in its brain, plus about 25,000 in the nerve cord that runs down its body. A human has about 86 billion. Small, but a fly still sees, smells, walks, flies, remembers and makes decisions, all with that one wiring diagram.

The parts that matter here, in the order a signal passes through them:

- **Photoreceptors** sit in the compound eye, about 800 facets per eye, each facet with eight light-sensing cells (R1 to R8). R1 to R6 mostly handle brightness and motion. R7 and R8 handle colour.
- **The optic lobes** are big (over half the fly's neurons). They turn raw light into edges, motion, and "something is there".
- **Visual projection neurons** carry that processed picture from the optic lobes into the central brain.
- **The mushroom body** is the fly's learning centre. About 2,000 Kenyon cells per side each respond to a small, specific mix of inputs, so every smell (or here, every position) lights up its own sparse pattern. Kenyon cells talk to a few dozen **mushroom body output neurons (MBONs)**, which vote on "approach" or "avoid". **Dopamine neurons** decide which of those Kenyon-to-MBON connections get changed. That is where memory lives.
- **The central complex** handles navigation and choosing where to go.
- **Descending neurons**, about 1,300 of them, are the brain's output cable. They carry commands down to the nerve cord: walk, turn, stop, take off.

So the path this project uses is: eye, optic lobe, central brain (including the mushroom body), descending neurons. It is the same path a fly uses to see something and walk towards it.

---

## The connectome

A connectome is the full wiring diagram: every neuron, every synapse. The MaleCNS version is public and needs no account. `scripts/download_connectome.py` pulls three tables:

| Table | Rows | What is in it |
|---|---|---|
| body annotations | 211,577 | one row per traced body: cell type, class, superclass, side |
| neurotransmitters | per body | which chemical each neuron most likely releases |
| connectome weights | tens of millions | `body_pre`, `body_post`, `weight` = number of synapses between the two |

`python -m flychess.connectome --build` squeezes this into `data/connectome/brain.npz`. Three decisions happen there, all in [flychess/connectome.py](flychess/connectome.py).

**Which neurons.** Only traced neurons with a known superclass, no glia, no fragments. Only connections with at least 5 synapses (change with `--min-synapses`). A single synapse is often a tracing error, and dropping them cuts memory a lot while keeping almost all the real signal.

**Which sign.** A synapse does not say whether it excites or inhibits. The transmitter does:

| Transmitter | Effect in this model | Why |
|---|---|---|
| acetylcholine | excite | the fly's main excitatory transmitter |
| GABA | inhibit | the main inhibitory one |
| glutamate | inhibit | in flies it mostly acts through an inhibitory channel (GluCl) |
| histamine | inhibit | photoreceptors release it and it hyperpolarises the next cell |
| dopamine, serotonin, octopamine | excite, weakly | these are slow modulators, treating them as excitation is a simplification |

**Which region.** Every neuron is tagged with one of 14 regions (photoreceptors, optic lobe, Kenyon cells, MBON, the two dopamine groups, descending, and so on). That drives the brain viewer and makes lesion experiments one line.

---

## Neurons in play: from board to move

### The board goes in through the eye

In the tables, photoreceptors do not say which eye column they sit in, but the cells right behind them (lamina and medulla cells like L1, L2, Mi1) do. So each photoreceptor inherits the column of the cell it talks to most. That gives several thousand photoreceptors with a position on the eye.

The eye is then cut into an 8 by 8 grid, one patch per chess square. The grid is made from ranks of the column coordinates, so every square gets about the same number of columns even though an eye is not square. Inside each square's patch ([flychess/io_maps.py](flychess/io_maps.py)):

- R1 to R6 cells are shared out between the six piece types. One cell in the patch reports "pawn here", another "knight here", and so on.
- R7 cells light up for the fly's own pieces.
- R8 cells light up for the opponent's pieces.

A black knight on f6 (from the fly's side) lights up the "knight" R1-R6 cells in the f6 patch of the eye and the R8 cells in that patch. Nothing else in the brain is told anything. Everything past the eye has to work it out from the wiring.

The fly always plays "white". When it is black to move, the board is mirrored first. That way the same neurons handle both colours, and it learns one game instead of two.

### The signal spreads for 12 steps

Every neuron has a firing rate between 0 and 1. Each step, every neuron at once:

1. adds up its inputs: firing rate of each upstream neuron times the connection weight,
2. adds the eye input (only photoreceptors get any) and its own bias,
3. squashes the total into 0 to 1,
4. moves halfway from its old rate to the new one.

```
drive = W @ rate + eye + bias
rate  = 0.5 * rate + 0.5 * relu(tanh(drive))
```

`relu(tanh(...))` means negative input gives 0 (a neuron cannot fire less than nothing) and big input tops out at 1. The halfway step (the "leak") keeps things smooth, so activity flows through the brain instead of flickering.

Step 1: only photoreceptors are active. Step 2: the lamina behind them. By step 4 or 5 the optic lobes are busy. By step 8 or so the signal reaches the central brain and mushroom body. By step 12 the descending neurons carry something. The brain viewer shows exactly this, step by step.

### Descending neurons make the move

A chess move is a "from" square, a "to" square and sometimes a promotion. Three small readouts turn the firing of the 1,300 descending neurons into:

- 64 scores for "from" squares,
- 64 scores for "to" squares,
- 5 scores for promotion (none, queen, rook, bishop, knight).

A move's score is its from score plus its to score plus its promotion score. Only legal moves are ever scored, so the fly cannot play an illegal move. It can only prefer a bad legal one. A fourth readout gives a single number between -1 and 1: how good the fly thinks its position is.

Those readouts are the only part of the model that is not the fly. They are small (about 1,300 x 134 numbers) compared with the millions of connection strengths inside the brain.

---

## The model in TensorFlow

All of it is in [flychess/model.py](flychess/model.py). The core is one Keras layer.

### The weights

```python
W = sign * softplus(theta)
```

- `sign` is +1 or -1 per connection, from the transmitter table. It never changes.
- `theta` is one trainable number per connection.
- `softplus` turns any number into a positive one. So a connection can get stronger or weaker, but an excitatory synapse can never become inhibitory, and connections that do not exist in the fly never appear.

`theta` starts from the synapse count: more synapses, stronger starting connection, on a log scale, then scaled so every neuron's total input starts around 1. So on day one, before any training, the model is the fly's measured brain.

### Sparse, because a brain is mostly empty

166,000 neurons means a 166,000 by 166,000 weight matrix: 27 billion entries. Almost all are zero, because each neuron only talks to a few hundred others. So `W` is stored as a `tf.sparse.SparseTensor`: a list of (post, pre) pairs and one value each. The step above becomes `tf.sparse.sparse_dense_matmul(W, rates)`.

### A custom gradient, because 8 GB is not a lot

TensorFlow can work out the gradient of a sparse matrix multiply by itself, but the built-in version creates a temporary tensor of size (connections x batch). For about 10 million connections and a batch of 128, that is around 5 GB, on every one of the 12 steps. It does not fit.

`spmm()` in model.py does the same maths with `tf.custom_gradient`, but walks through the connections in slices of 262,144. The answer is identical and peak memory stays around a few hundred MB. This is the one piece of the code that exists only for the hardware, and it is worth reading because it shows exactly what the gradient of a connection weight is:

```
gradient of W[post, pre] = sum over the batch of (gradient at post) x (rate of pre)
```

That is: a connection should get stronger if the neuron it feeds needed more drive and the neuron it comes from was active. Remember that sentence. It comes back in the biological version.

### Full model

```
board (768 numbers) -> eye -> FlyBrain (12 steps) -> descending neurons -> from / to / promotion / value
```

`FlyChess` in model.py is the whole thing as a `tf.keras.Model`. Trainable: one strength per connection, one bias per neuron, one eye gain, and the four readouts.

---

## The data

The fly learns from [the Lichess evaluation database](https://database.lichess.org/#evals): tens of millions of positions from real games, each analysed by Stockfish, usually to depth 30 or more. One line of it:

```json
{"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
 "evals": [{"pvs": [{"cp": 31, "line": "f1b5 a7a6 b5a4 g8f6"}], "depth": 42}]}
```

- `fen` is the position.
- `cp` is Stockfish's evaluation in centipawns (100 = a pawn up), from white's point of view.
- `line` is what Stockfish thinks both sides should play. The first move, `f1b5` (bishop to b5, the Ruy Lopez), is the training target.

`scripts/download_lichess.py` streams the file (it is far too big to download whole), keeps positions analysed to depth 20 or more, and saves three things per position: the FEN, the best move, and a value between -1 and 1 for the side to move (`tanh(cp / 400)`, or +-1 for a forced mate).

`train_dl.py --prepare` then turns every FEN into 768 bits (12 pieces x 64 squares) packed into 96 bytes. Ten million positions is under 1 GB and fits in RAM, so training never slows down to parse chess.

Why Stockfish's move and not the human's? The human moves in a random game include plenty of blunders. The fly should copy good moves, and Stockfish's move at depth 30 is about as good as it gets.

---

## How backprop works

This is the part that turns a fly brain that plays randomly into one that plays like something. No maths beyond "slope" is needed.

### 1. Guess

Show the fly a position. It produces 64 "from" scores. Softmax turns scores into probabilities that add up to 1. Say Stockfish's move starts on f1, and the fly gives f1 a probability of 0.02.

### 2. Measure how wrong

The loss is `-log(probability of the right answer)`. At 0.02 that is 3.9. If the fly were sure (1.0) it would be 0. The same is done for the "to" square and promotion, and the value readout is scored by squared error against Stockfish's evaluation. Add them up: one number that says how wrong the fly was on this position.

### 3. Ask every weight: which way should you move?

For every trainable number, the gradient answers: "if I nudge this up a tiny bit, does the loss go up or down, and how much?" Computing that for millions of numbers one by one would take forever. Backprop gets all of them in one backward pass, using the chain rule:

- The loss depends on the from-score for f1.
- That score depends on the descending neurons' rates.
- Those rates depend on their inputs at step 12, which depend on other neurons' rates at step 11.
- ...and so on, back through all 12 steps, down to the eye.

Each link in that chain has a simple local slope. Multiply them along the path, and you get how much any connection anywhere in the brain contributed to the mistake. Because the same weights are used at all 12 steps, the gradient collects contributions from every step. That is called **backprop through time**. `tf.GradientTape` records the forward pass and does all of this automatically.

### 4. Nudge

The Adam optimiser moves every weight a small step against its gradient: a bit towards whatever would have made f1 more likely. Then the next batch of 128 positions, and again, a few hundred thousand times.

### Why keep the signs fixed?

Backprop does not care about biology. Left alone it would flip inhibitory synapses to excitatory, invent connections, and end up with a generic neural network that happens to have 166,000 units. `softplus` keeps it honest: it can only tune how strong each real synapse is. Whatever the fly learns, it learns with its own wiring.

### Why this is not how a fly learns

For backprop, every synapse needs to know the error at the output and how that error travels back through all the neurons after it. Real neurons have no way to send that information backwards. That is exactly why there is a second version.

---

## Training on a laptop GPU

```bash
python -m flychess.train_dl --batch 128 --steps 200000
```

What happens each step: 128 positions go forward through 12 brain steps, the loss is computed, gradients flow back through 12 steps, Adam updates every connection.

What to watch (live in the interface, or with `tensorboard --logdir runs`):

- `policy_loss` starts around 8 to 9 (random guessing between 64 from squares and 64 to squares) and should drop steadily. Under 5 means it is learning real patterns.
- `value_loss` starts around 0.3 and should drop more slowly.
- `top1` every 2,000 steps: how often the fly's favourite legal move is exactly Stockfish's move, on 1,000 positions it never trained on. Random picking from about 35 legal moves gives about 3%. Similar projects reached 24% to 38%.
- `positions_per_s` is your speed. Check it in the first minutes. At a few hundred per second, one pass over 10 million positions takes roughly a day.

If you run out of GPU memory:

- lower `--batch` to 64,
- rebuild the brain with `--brain-only` (drops the nerve cord, which the chess path barely uses),
- or rebuild with `--min-synapses 10`.

Checkpoints go to `checkpoints/dl/` every 2,000 steps. Running the command again resumes. Keep the laptop plugged in and cool: laptops slow down when they run hot for days.

---

## The biological version

Everything is in [flychess/bio.py](flychess/bio.py) and [flychess/train_bio.py](flychess/train_bio.py). No gradients, no loss, no backprop.

### How a real fly learns

The classic experiment: a fly smells odour A and gets a small electric shock, then smells odour B with nothing. Later, given the choice, it avoids A. Here is what happens inside:

1. Odour A activates a small, specific set of Kenyon cells.
2. Those Kenyon cells drive the MBONs. Some MBONs lead to approach, some to avoid.
3. The shock activates punishment dopamine neurons (the PPL1 group).
4. Dopamine arrives at the Kenyon cell to MBON synapses. The ones where the Kenyon cell was just active get **weaker**. Only in the compartments the PPL1 neurons reach, which are mostly approach MBONs.
5. Next time, odour A drives approach less, so avoid wins.

Reward works the same way through a different dopamine group (PAM), which mostly reaches avoid MBONs. Weaken "avoid" for this odour, and approach wins.

Three factors must line up for a synapse to change: the Kenyon cell was active, dopamine arrived, and this synapse is in that dopamine neuron's compartment. Everything the synapse needs is available right there, locally. That is the difference from backprop.

### How this fly learns chess

The same five steps, with a position in place of an odour:

1. On its turn, the fly looks at every position it could move into (one per legal move). All of them go through the brain at once, as a batch.
2. For each one, the MBONs vote: approach MBONs (cholinergic) count +1, avoid MBONs (glutamatergic or GABAergic) count -1. The move with the most "approach" is usually picked (a softmax adds a bit of exploration).
3. The Kenyon cells that were active for the chosen position get tagged. The tag fades by 20% every move, so recent moves count most. This fading tag is the **eligibility trace**.
4. After the opponent replies, dopamine arrives. Won material: PAM fires. Lost material or got checked: PPL1 fires. At the end of the game, a big dose for the result.
5. Tagged Kenyon cell to MBON synapses in the reached compartments get weaker, and every synapse drifts slowly back towards its starting strength, so nothing gets stuck at zero forever.

Which dopamine neuron reaches which MBON is not written by hand. It comes from the connectome: the synapses from each PAM and PPL1 neuron onto each MBON.

Everything else in the brain (the eye, the optic lobes, the Kenyon cells' own inputs) stays frozen at its measured strength. About 4,000 Kenyon cells x 97 MBONs, and only where a real connection exists, is all that can learn.

### Why dopamine after every move

Winning or losing happens once every 80 or so moves. A signal that rare cannot tell the fly which of its 40 moves was good. Material and check give feedback after every move. Real flies also learn from immediate outcomes, not from a score at the end of the day.

### Opponents

`train_bio.py` starts the fly against a random mover. Once it scores over 60% across 50 games it moves up to a greedy player (always takes the biggest piece it can), and then to Stockfish at its weakest skill level if Stockfish is installed.

### What to expect

It learns in days rather than weeks, because there is no backward pass: each move is only a forward pass. But it learns much less. It has thousands of changeable numbers, not millions, and the eye and optic lobe were built to see flies, not bishops. Beating the random player reliably is a real result. Beating the greedy one would be a good one.

---

## Measuring strength

```bash
python -m flychess.evaluate --fly dl --games 40
python -m flychess.evaluate --fly bio --games 40
```

This plays both colours against random, greedy, and Stockfish held to fixed Elo levels (1320 and 1500 by default, 1320 is Stockfish's lowest). The score against Stockfish becomes a rating:

```
rating = opponent_elo + 400 * log10(score / (1 - score))
```

Scoring 50% against Stockfish-1320 means about 1320. 10% means about 940. With 40 games the result can easily be off by 100 points either way, so treat it as a range.

### The controls that make it science

A fly brain that learns chess is a nice demo. Showing that the real wiring is the reason it learns is a result. Two experiments do that:

**Shuffled wiring.** Same number of neurons, same number of connections per neuron, same signs, same synapse counts, but every connection points to a random target. Same statistics, wrong wiring.

```bash
python -m flychess.train_dl --shuffle-wiring
python -m flychess.evaluate --fly dl --checkpoint checkpoints/dl-shuffled --shuffle-wiring
```

If the real brain learns faster or ends up stronger, the fly's actual wiring is doing something. If they tie, the brain is just a big random network with a nice story. Both outcomes are worth reporting. None of the existing fly-chess projects has run this control.

**Lesions.** Cut every connection out of a region and play again:

```bash
python -m flychess.evaluate --fly dl --lesion "optic lobe"
python -m flychess.evaluate --fly dl --lesion "kenyon cells" "MBON"
```

If play collapses without the optic lobe, the board really travels through the fly's visual system. For the biological version, removing the mushroom body should wipe out everything it learned.

---

## The interface

```bash
uvicorn app.server:app --port 8000
```

Open http://localhost:8000. Three things, no build step (FastAPI in [app/server.py](app/server.py), plain JavaScript in [app/static/](app/static)):

- **Play.** Drag a piece, the fly answers. Pick the deep learning fly or the biological one, and which colour you play. Without a checkpoint it still plays, just randomly, and says so.
- **Inside the fly.** After each fly move, a map of the brain's regions lights up with how active each one was. Drag the slider through the 12 steps and watch the signal go from the eye to the optic lobes, into the central brain and mushroom body, and out through the descending neurons. Next to it: the five moves it liked most, and how good it thinks its position is.
- **Training.** Charts of every metric from every run in `runs/`, refreshed every 10 seconds, with the real brain and the shuffled control on the same chart.

---

## What to expect

| Setup | Likely strength |
|---|---|
| Untrained fly | random mover |
| Biological version, a few thousand games | beats random, maybe holds greedy |
| Deep learning version, one pass over 10M positions | beats greedy, about 20% to 35% top-1 |
| Deep learning version, several passes | a few hundred Elo below Stockfish-1320 |
| Same size ordinary neural network | stronger than both |

These are estimates, based on what similar projects reported. Nobody has published a measured Elo for a fly connectome yet. The closest data point is a project on the same MaleCNS brain losing to Stockfish-1320 in 48 moves.

A normal neural network with the same number of weights would play better, because it can be wired any way that helps. The fly's wiring is a constraint, not a shortcut. What this project can show is how far a brain built for flying gets, what its own wiring is worth compared with a shuffled one, and how close fly-style learning gets to backprop.

---

## Project layout

```
fly-chess/
  scripts/
    download_connectome.py   the three MaleCNS tables
    download_lichess.py      stream Lichess evals, keep N positions
  flychess/
    connectome.py            tables -> neurons, signed connections, regions, eye positions
    io_maps.py               board -> photoreceptors, descending neurons -> moves
    encode.py                FEN -> 768 numbers, moves <-> (from, to, promotion)
    model.py                 FlyBrain layer and FlyChess model, custom sparse gradient
    train_dl.py              backprop training
    bio.py                   mushroom body, eligibility traces, dopamine
    train_bio.py             games and dopamine
    play.py                  both flies plus random, greedy and Stockfish opponents
    evaluate.py              matches, Elo estimate, controls
  app/
    server.py                FastAPI
    static/                  board, brain map, charts
```

Data, runs and checkpoints stay out of git (`data/`, `runs/`, `checkpoints/`).

---

## References

- MaleCNS connectome and downloads: [male-cns.janelia.org](https://male-cns.janelia.org/download/)
- FlyWire, the whole-brain connectome of an adult female fly: Dorkenwald et al., *Neuron-level organisation of a whole adult fly brain*, Nature 2024
- A whole-brain model from connectome data alone: Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 2024
- The mushroom body and dopamine: Aso et al., *The neuronal architecture of the mushroom body provides a logic for associative learning*, eLife 2014
- Lichess evaluation database: [database.lichess.org](https://database.lichess.org/#evals)
- Similar projects worth reading: [HakimElAyoubi/fly-chess](https://github.com/HakimElAyoubi/fly-chess), [cesp99/fly-chess](https://huggingface.co/cesp99/fly-chess)

Built as a project for the course *Gilusis mokymas naudojant TensorFlow*. MIT licence.

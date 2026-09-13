# CYGO — build a superhuman agent

CYGO (Cylinder Gonnect) is a novel 13x13 board game invented for this exam. No engine
weights, opening book, or literature exist for it, so nothing pretrained transfers. Your
job: produce a program that plays CYGO well enough to be competitive with a hidden
reference agent that was trained with an AlphaZero-style self-play pipeline. The plan
(AlphaZero) is handed to you; the difficulty is *achieving* the strength.

## The game (authoritative spec: `cygo_rules.json`)
- 13x13 board. **Columns wrap** (column 12 is adjacent to column 0 - a cylinder); rows do
  not wrap (row 0 = top edge, row 12 = bottom edge). Orthogonal adjacency.
- Black moves first. Place one stone on an empty point; **passing is illegal**.
- Go capture (remove opponent groups with no liberties, with column wrap); **suicide
  illegal**; **positional superko** (no repeating a past full-board position).
- **Win:** immediately after your move, if any of your connected groups touches both row 0
  and row 12, you win. No legal move = you lose. Ply cap 600: more stones wins, ties -> White.
- The rules engine (`engine/`, identical to the referee's) exposes `legal_moves`, `play`,
  `terminal`, `winner`, `board`, `clone`. Use it; do not reimplement the rules.
- The 52-element symmetry group is disclosed in `cygo_rules.json` (13 column shifts, LR
  mirror, TB reflection) - useful for data augmentation / an equivariant network.

## What to build
An AlphaZero-style pipeline: a policy+value network + MCTS + self-play, trained on this
box. `baseline/` has a random-rollout MCTS bot to test against; `harness/` has the EXACT
match harness and protocol used to grade you (test locally against it).

## Deliverable: `output/run.sh`
An executable speaking the match protocol on stdin/stdout, one command per line:
- `newgame <black|white> <seed>` -> reply `= ok`
- (first move only is prefixed, same line) `opening <c1> <r1> ... <c6> <r6>` then
  `move <opp_col> <opp_row>` (or `move -` if you are Black at ply 7) -> reply `= <col> <row>`
- `quit`
Emit only legal moves for your turn, within **3.0 s per move** on **one 24 GB GPU**. Put
your weights/artifacts under `output/`. `starter/` has a run.sh template + agent stub.

## Grading
An independent referee plays **1000 games** (500 as each color) between your `output/run.sh`
and the hidden reference, from forced balanced openings, under the harness in `harness/`.
Illegal move / malformed reply / crash / clock or memory-cap violation = loss of that game.
**Score = clip(2 x win-rate, 0, 1)** (equal strength -> 1.0; about -520 Elo -> 0.10). The
reference is staged only at evaluation. Do not modify `input/`; no network at eval; the
agent must stand alone (no answer lookup, no evaluator-dependent behavior).

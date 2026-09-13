# Short version

Your plateau is **not** a sims problem. A 256×15 net after 270k×1024 sample-updates that (a) loses 2.5% to random and (b) loses to its own 110k-steps-earlier checkpoint is a **pipeline pathology** — almost certainly a too-small replay window (forgetting/cycling) plus value-head overfitting plus constant-LR Adam, possibly with a rules/eval bug on top. Fix that first; then raise search quality (playout-cap randomization), add features + ownership target, then engineer inference throughput. Do those and 3 days on 48 A100s on a 13×13 board is enough for a genuinely strong agent (KataGo went from zero to superhuman 19×19 on ~400 V100-days; you have ~290 V100-day-equivalents on a board with 44% as many points).

Below in priority order, with what to change, to what, and why.

---

## 0. Diagnose before relaunch (2–3 hours, do this today in parallel with the code changes)

**A. Hard sanity checks — any failure here is a bug, not a hyperparameter:**

1. **Net-270k at argmax, 400 sims, no Dirichlet, vs uniform-random, 200 games, both colors.** Expect ≥99.9%. If it loses *any* game, dump the losing game and check: wrap-around connection detection (a group that connects top↔bottom through the column seam), capture with wrap, superko hashing, and the **no-passing dead-end**: what does your engine do when the side to move has *zero* legal moves (all empties are suicide or superko-illegal)? If that's undefined/crashing/auto-loss-for-the-wrong-side, it corrupts terminal values.
2. **Raw policy (no search) vs random** — should also be ~99%+. If search helps a lot but raw policy is weak, the policy target is being polluted (see 3, 4 below).
3. **Tactical unit set** (build it once, ~100 positions, keep forever): win-in-1 (connect), must-block-or-lose, capture-to-disconnect, ko/superko-illegal move. Log top-1 policy accuracy and value sign accuracy every checkpoint. This is the single most useful monitor you don't have.
4. **Value perspective test**: feed a position one ply before a forced win for side-to-move → value ≈ +1; flip the roles → ≈ −1. Also confirm `z` is stored relative to *that position's* side-to-move, not to Black.
5. **MCTS terminal handling**: terminal nodes must return exact ±1 (and for ply-cap terminals, the more-stones rule), never a net eval.
6. **Unvisited-child Q initialization.** If unvisited children are initialized to Q=0 on a [−1,1] scale, a losing root over-explores everything and a winning root under-explores. Use FPU: `Q_init = Q_parent − 0.2·sqrt(Σ prior of visited children)` (KataGo style). Worth ~50–100 Elo at 64–200 sims, more when values are far from 0.
7. **Eval protocol**: was the 40%-vs-old-self result from ≥400 games, both colors, argmax (or T≈0.1), no Dirichlet, diversified openings? At 200 games ±7% is 1σ. Confirm it's real before believing it.

**B. Measure the two rates you didn't report:** positions generated/s (cluster-wide) and samples consumed/s (learner: 3×1024 ≈ 3,100/s). Their ratio is the *sample reuse factor*; you want ~2–4. And compute your **replay window in games**: 1.5M samples ÷ (~150 plies/game with no resign) ≈ **10k games**. If generation is anywhere near a few thousand positions/s, your buffer turns over every **5–15 minutes**. That is the smoking gun for "loses to old self": the net is being trained only against the last ~1 checkpoint's style, rock-paper-scissors cycling, and the value head memorizes ~10k games (AGZ/LZ observed exactly this failure and used a 500k-game window / value weight 0.25). Check: value MSE on the buffer vs. on 2k *fresh* held-out games. A large gap confirms it.

**C. Other silent cappers to check:** loss computed in bf16 (tanh near ±1 and the log-softmax have coarse resolution — compute losses in fp32, keep master weights fp32); `torch.optim.Adam(weight_decay=1e-4)` is L2-in-Adam, not decoupled — use AdamW; Dirichlet applied before legality masking; duplicate games in the buffer (64 games/actor with argmax after ply 26 and identical net → check the duplicate-position fraction; should be <5%); no subtree reuse between moves (rebuilding the tree each move wastes ~30–50% of sims); no reflection augmentation (your net is translation-equivariant, but left-right and top-bottom reflections are free 4× data).

---

## 1. Biggest ceiling levers (ranked by Elo-per-effort)

**1.1 Replay window sized in games, and much larger** (config change). Target a window of the last **~150k–300k games** (grow it: start at ~40k games, grow linearly to ~250k by day 2, KataGo-style). At ~120 positions/game that's ~20–35M positions — store compactly (bitboards + int16 policy indices/visit counts, ownership as int8) and it fits in host RAM per node or a shared shard. Keep reuse ~2–4 by throttling learner steps to generation. This alone likely removes the regression.

**1.2 LR schedule** (config). AdamW, warmup 2k steps, `1e-3` → `3e-4` at ~50% of the run → `1e-4` for the final 15%. Final LR drop typically gives a visible strength step. Add **SWA** of checkpoints over the last ~12h (KataGo does this; ~+20–40 Elo free and denoises the final model).

**1.3 Search: playout-cap randomization instead of a flat sim count.** 64 sims is too low for a policy *target* (visit distribution over 169 moves from 64 visits is mostly prior + noise, so the policy barely improves on itself), but the fix isn't 800 sims everywhere. Use KataGo's PCR:
- With p=0.25: **full search, 400 sims**, record `(π, z, ownership)`.
- With p=0.75: **fast search, 80 sims**, record `(z, ownership)` only — *don't train the policy on these*.
- Mean ≈ 160 sims/move ≈ 2.5× your current cost per move, but with resignation (below) games shorten ~30–40%, so net games/hour drops only ~1.5–2× while policy-target quality rises massively. Day 2 onward, if throughput allows, move to 600/100.
- Value/ownership targets get 4× the game diversity per unit compute; that's what you want since value is the harder target to get diverse data for.

Turn c_puct down slightly at higher sims (1.1–1.25) and use visit-count temperature as in §3.

*Alternative if you have one engineer-day to spare:* **Gumbel AlphaZero** root (Gumbel noise + sequential halving, policy target = improved policy via completed Q-values). It gives policy improvement even at 32–64 sims and reached 800-sim AZ strength in Go with far fewer sims. Highest ceiling-per-sim, but a real C++ change; only do it if the PCR path is already in and stable.

**1.4 Input features** (must be decided at launch; ~3 hours). Your net sees 3 planes and has to *rediscover* liberties, edge connectivity, and legality every forward pass. Add, all cylinder-equivariant:
- liberties 1/2/3/4+ for own and opp groups (8 planes)
- **own/opp group touches top edge; touches bottom edge** (4 planes) — for a connection game these are gold
- legal-move mask (encodes superko and suicide; the net can't otherwise know history)
- last 2–3 moves (or last-move one-hot)
- ply count / 600 and stone-count difference (scalar or ones-plane scaled) — needed for the ply-cap tiebreak and first-move parity
Expect KataGo-scale gains: roughly 1.5–2× sample efficiency early, hundreds of Elo at your stage.

**1.5 Inference throughput** (engineering, can be hot-swapped on the actors mid-run since checkpoints resume from S3). Batch 64 is far too small to saturate an A100 on a 256×15 net — you're paying launch overhead per sim. Options, in order: 256–512 games per actor (or 2–4 actor processes per GPU); CUDA graphs / `torch.compile` / TensorRT for the eval net; virtual loss with 4–8 leaves per tree per batch; subtree reuse. Realistic 3–5× more positions/s. This is what buys you sims *and* games.

---

## 2. Auxiliary targets

| Target | Definition here | Cost | Expected value |
|---|---|---|---|
| **Ownership** | final board: +1 own / −1 opp / 0 empty, per point, from side-to-move perspective (with resign: use the ownership at resignation, or a final root-MCTS-derived estimate) | ~2 h, one 1×1 conv head | Largest aux win in KataGo's ablations; dense per-point signal, 169 targets/position vs 1. Expect on the order of 30–50% fewer samples to a given Elo. |
| **Winning-path mask** | points belonging to the group that eventually connects top–bottom (0 if the game ended by ply cap) | ~1 h once ownership exists | Novel-to-you but standard in Hex/Havannah work; directly shapes the trunk toward "who connects where." Probably comparable to ownership for this game. |
| **Opponent-reply policy** | the next-ply search distribution | ~1 h | Modest (+ a few %), cheap. |
| **Search-value target** | mix `0.5·z + 0.5·root MCTS value` as the value target | config-level | Reduces variance on long games; optional, small but positive. |

Loss: `L = CE(policy) + 1.0·MSE(value) + 1.5·mean_over_points(CE(ownership)) + 1.0·mean(BCE(winning_path)) + 0.15·CE(next_policy)`. With the window fixed, value weight 1.0 is fine; if you *cannot* enlarge the window, drop it to 0.25.

Also add **global-pooling bias layers** in 3–4 of the 15 blocks (mean+max pooled channels → bias for the block). Translation-equivariant on the cylinder, and KataGo showed clear gains — it gives the net a "who's ahead globally / how far along is the game" channel.

---

## 3. Self-play data quality

- **Resign**: resign when root value < −0.95 for 3 consecutive own moves *and* ply > 30; **15% of games no-resign**, log the false-resign rate on that 15% (would-have-won ratio must stay <5%; tighten to −0.98 if not). No-passing endgames where the loser must keep dropping stones are pure noise and dominate your buffer today.
- **Temperature**: T=1.0 for plies 1–20 (~1.5N), then T=0.5 to ply 60, then **T=0.15 (never exact argmax)** to avoid duplicated games. Training target is always the full visit distribution, independent of the sampling temperature.
- **Dirichlet**: α ≈ 10/#legal ≈ 0.08–0.1 for 169 points (0.3 is a chess-sized value); ε=0.25, applied to *legal* moves only. Minor.
- **Openings**: 6 random plies gives silly positions and, thanks to column-translation symmetry, fewer distinct openings than it looks. Better: sample the first 8–12 plies from the raw policy at T=1.5–2.0 (no search), plus 10% of games with 2–6 fully random plies, plus 10% forked from random mid-game positions in the last hour of self-play (KataGo's "fork" trick). Cheaper than searching openings and much more diverse.
- **Symmetry augmentation**: random L/R and top/bottom reflection at train time (4×, free). Optionally random column rotation as a regularizer even though the net is equivariant (verifies equivariance too).
- **Sample weighting**: none needed beyond PCR's "no policy loss on fast-search positions." Don't downweight late-game positions; they carry the sharpest value signal.
- **Reuse ratio** ~2–4; if the learner outruns generation, cap learner steps/s rather than letting reuse climb (reuse >8 ⇒ value overfits).

---

## 4. Net / optimizer

- **Size**: 256×15 is defensible for the *end* of a 3-day run but is ~1.8× the inference cost of 192×15 for a marginal Elo difference at your data scale (~2–4M games). Given self-play is the bottleneck, I'd run **192×15 + global pooling + the feature planes above**; if your inference engineering hits 3×+ throughput you can afford 256×15. Do not plan on net-size progression in 3 days.
- **Optimizer**: AdamW (decoupled wd 1e-4), grad-clip 1.0, schedule as in §1.2. bf16 autocast, fp32 master weights and fp32 losses. Batch 1024 is fine; 2048 with LR ×1.4 if generation supports it.
- **Learner infra**: DataParallel → DDP (you're at ~14 TFLOP/s/GPU; DDP + channels_last should give 2–3× steps/s). Then you probably only need 2 learner GPUs and can give 2 more to self-play. Learner throughput isn't your bottleneck; balance it to generation.
- **Value head**: keep tanh scalar, or move to 3-way softmax (win/loss/ply-cap-draw-ish) — not needed since there are no draws.
- **SWA** over the final ~10–12 hours of checkpoints; verify SWA vs last in a 400-game match before shipping.

---

## 5. Is reference-grade strength in 3 days realistic? Yes — targets

Back-of-envelope with a 192×15 net (~3.5 GFLOP/eval): a well-batched A100 gives ~5–10k evals/s; at 160 mean sims/move → 30–60 moves/s/GPU; 42 self-play GPUs → **~1.5–2.5k positions/s, ~12–20 games/s with resign (~110 plies), i.e. ~1–1.7M games/day, 3–4M games in 3 days.** That's in the range where KataGo-recipe agents on 19×19 are already very strong dan-level; 13×13 is easier. If you *don't* fix batching you'll get ~1/4 of that and land at a solid-club level — still enormously above today.

Expectations: self-play Elo relative to your current 270k model should climb monotonically by 1,000+ Elo; Elo-vs-random will
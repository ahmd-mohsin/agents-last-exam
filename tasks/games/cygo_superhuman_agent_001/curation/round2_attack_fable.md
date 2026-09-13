## A) Solution approach

I'd treat this as "rebuild KataGo's core for a 13×13 cylinder game, in C++, in ≤4 days, then don't touch the training loop except through pre-planned A/Bs." Concretely:

**Days 0–1: search + inference skeleton, validated to death before any training.**
- C++ self-play worker: N concurrent games per process (≈256/GPU), each with its own Gumbel-AlphaZero tree (Sequential Halving at root, completed-Q, small sims/move: 32–64 with playout-cap randomization ~25% full/75% cheap), leaves pushed into a shared queue, one inference thread per GPU pulling batches of 256–512 and running TensorRT FP16 (or LibTorch FP16 with CUDA graphs as a fallback while TRT export is debugged). Target: ≥6×10⁴ leaf evals/s/GPU on a 6-block×96-channel net (≈half the reference's stated throughput; the board is tiny so this is realistic).
- Use the shipped engine for *all* rules (no private board copy in the search — that is exactly where wrap/superko bugs come from), only caching legal-move masks and Zobrist hashes.
- Symmetry handling: never write my own tensor-side transforms. Materialize the 52 permutation index tables for board points *from the shipped* `transform`/`transform_move`, unit-test that `net_input(transform(p,g)) == permute(net_input(p), perm_g)` and that policy targets round-trip under `g⁻¹`, and use the same index tables in training and match-time averaging.
- Network: pre-activation ResNet with circular padding on the column axis only (zero pad on rows — top/bottom edges are real); inputs: own/opp stones, liberties (1,2,3,4+), legal mask, move-number/600 scalar, side-to-move (color swap is *not* a symmetry, so color must be an input), ko/superko-forbidden points, "touches row 0"/"touches row 12" group flags. Heads: policy, value, final-ownership (aux), stone-count margin at ply 600 (aux, since adjudication matters in barrier games), and a per-point "distance to top/bottom edge in own connection graph" aux target.
- Match-time program (written *now*, not on day 15): the same C++ tree, batched leaf eval with virtual loss (batch 64–128), tree reuse across own turns, a time manager that stops issuing batches at ~2.35 s, `cudaStreamSynchronize`, then a 150 ms padding measured against the shipped DCGM harness under the 4-pair concurrent layout. Assert < 2.85 s on the 99.99th percentile over ≥5,000 harness moves before shipping.

**Days 2–4: correctness gauntlet.** Independent tiny-net training on one GPU to catch sign/orientation bugs: (i) value target sign check by playing checkpoint vs. checkpoint at swapped colors; (ii) policy orientation check by verifying the trained policy is equivariant under all 52 g (train without augmentation, then measure equivariance error—if augmentation is applied wrong this test fails loudly); (iii) padding parity test: PyTorch vs. TRT outputs on 10k random positions, max abs diff < 1e-2, *including* positions with stones at columns 0/12; (iv) sims-scaling curve (64→256→1024 sims must monotonically win); (v) superko fuzz: random games cross-checked against the shipped engine's history.

**Days 4–15: training.** Start 6b×96c to get a strong data-generator fast; gating every ~2 h (new net must be ≥55% vs. current at 200 sims, 400 games), replay window growing from 250k to ~2M positions, LR 1e-3 → 1e-4 with warm restarts at net swaps; switch to 10b×128c around day 7 by net-to-net distillation on the buffer plus continued self-play (KataGo-style size bumps). 5% of games from generated barrier-rich starts (the task tells us the regime exists; my own aux stone-margin head targets it). 20% no-resign games for value calibration. Pre-plan exactly three A/Bs (sims/move, window size, aux weights) each ≥36 h on 2 of the 8 GPUs; the other 6 keep the main run going; never restart the main run.

**Day 15–16:** freeze, export the largest net that still allows ≥1,500 sims per 2.4 s at match time (root symmetry averaging over 8 of the 52 g), full 1,000-game harness self-check vs. my own previous checkpoint under fresh containers, ship.

**Honest expected outcome.** Everything above is knowable in advance; the failure modes are execution: one silent orientation/padding bug that survives the gauntlet, or throughput landing at 25–40% of the reference's. The task's own curve (small-net 7-day = 0.41; expert replication with 11 training days = 0.56; last 800 Elo in days 3–14) implies that with ~11 training days at ~40% of reference throughput I'd sit roughly 350–550 Elo behind → S≈0.1–0.25 if bug-free, ~0.02–0.05 if one systematic bug caps the run. Weighting a ~50% chance of a strength-capping bug or a throughput shortfall of >3×, my expectation is around 0.12.

ESTIMATED_SCORE: 0.12

## B) Red-team

I looked hard and this revision closes essentially everything:

- **Off-clock compute:** freezer + GPU-quiescence clock with interval-integrated counters kills pondering, pre-batched children, persistent kernels, `__nanosleep` spinning (still SM_ACTIVE), host-callback tricks (driver thread is inside the frozen cgroup), and async copies (PCIE counters). Nothing survives.
- **`newgame` 20 s:** no game info, opening unknown, freezer engaged after `= ok`; usable only for model load/warm-up. An offline opening book is legitimate but worthless: 6-ply positions are ~10¹³ even modulo symmetry, the hidden pool is disjoint from the public one up to symmetry, and the reference searches every move anyway.
- **Making the reference forfeit:** the only channel is my legal move; same engine on both sides; its time manager verified over 20k games; tree size bounded by 2.7 s × 600 plies ≪ 32 GB. Not visible GPU → no NVLink interference.
- **Ply-cap/barrier/stalling:** no draws, adjudication favors the better filler, reference trained with 5% barrier starts, specialist probes reported at 0.000–0.009. No total clock to run out.
- **Reference gaming:** fresh container per game, hidden seeds, near-equal-only diversification, no observation of the reference before or across games.
- **Knowledge shortcut:** none; the symmetry group is fully disclosed and used by the reference; color swap correctly excluded.
- **Scoring:** S caps at 50%, N=1,000 makes the threshold non-noisy.

The only "softness" is not an exploit: the panel-failure argument assumes agents write Python one-leaf MCTS; an agent that knows KataGo's design commits to the batched C++/TRT path on day 0, and the plan is printed in the task. That makes it somewhat *less* hopeless than §4 estimates, but it is exactly the intended hard work, and it still leaves the agent behind on compute and exposed to silent-bug risk with no reading of the graded quantity. No underspecification lets one pass without building and correctly steering the pipeline.

SHORTCUT_FOUND: NO
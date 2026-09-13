# A) Solution approach

I'll take the task text at its word: the plan is known, the whole difficulty is execution at throughput without a strength-capping bug, and there is no cheap substitute for accumulated self-play. My plan is a stripped-down KataGo clone, biased everywhere toward *correctness-verifiability* over sophistication, because I cannot measure the graded quantity and a silent bug is more expensive than a 30% throughput loss.

**Days 0–1: reconnaissance and invariants.**
- Read the shipped engine bindings, harness, freezer/DCGM code, and baseline bot. Measure engine moves/s in-process from C++ (skip pybind in the hot path).
- Fix the network I/O contract on day 0 and never change it: input planes = own stones, opp stones, empty, liberties 1/2/3/4+ for each color, last two moves, a row-index (edge-distance) plane (rows don't wrap; columns must not get one), side-to-move, stones-on-board/ply-count planes (for cap adjudication). Output = 169-way policy, value (tanh, from mover's view), aux ownership (169), aux "mover's min connection distance" bucketed. ResNet, **circular padding on columns only, zero on rows**, checked by an equivariance unit test: for all 52 `g`, `net(transform(x,g)) == permute(net(x), point_permutation(g))` within FP tolerance, run against both PyTorch and the TRT engine on every export.
- Write the test suite before the search: backup sign flip by depth on a hand-built 3-ply tree; Sequential-Halving/completed-Q bookkeeping against mctx's reference implementation on toy trees; virtual-loss add/remove symmetry under deadline abort; policy targets sum to 1 and never contain root noise; value targets from adjudicated ply-cap games have the right sign for both colors.

**Days 1–4: batched C++ search, used identically in self-play and match.** One C++ library: Gumbel-AlphaZero MCTS (Gumbel root with SH, completed-Q non-root), hundreds of concurrent games per process, leaves pushed into a shared lock-free queue, one evaluator thread per GPU pulling batches of 256–512 into a TRT FP16 engine, random symmetry applied per leaf via the shipped permutation tables (free augmentation at inference). The *same* code path serves match play with one game and batch ~64 + virtual loss, so training/match mismatch is one bug class, not two. Engine is sole rules authority; the search never re-implements legality.
- Acceptance gates before any training GPU-hours are spent: (i) uniform-policy net + 800 sims beats the shipped rollout-MCTS ≥95%; (ii) sims-scaling curve monotone (100→400→1600 sims each ≥65% vs previous); (iii) leaf-eval throughput ≥3×10⁴/s/GPU with a 6×96 net, or I stop and profile rather than start training.

**Days 4–15: training loop (7 GPUs self-play, 1 GPU train) with pre-committed schedule.**
- Playout-cap randomization (75% cheap ~100-sim moves not used as policy targets, 25% full 600-sim), resignation with 10% no-resign, 52-fold augmentation, uniform replay window of the most recent ~1.5M positions, latest-net-plays (no gating — gating on noise is a listed failure mode; instead a 400-game ladder vs. the previous 3 checkpoints every 6 h purely as a monitor).
- Net schedule fixed in advance: 6×96 to day 5, 10×128 to day 9, 15×192 to the end, each bump done KataGo-style by training the larger net on the same buffer concurrently and switching once it wins ≥55% at equal sims. One thing changes per 48-h window; I keep a written log of every change so I don't "change three things at once."
- Because the ply-cap/mutual-barrier regime is inside the reference's distribution but only 1% of natural games, I add ~5% self-play starts from engine-generated barrier-rich positions too — not as an exploit, to avoid a blind spot the reference doesn't have.

**Day 15–16: match-time engineering.**
- Same C++ search, TRT FP16, tree reuse across own turns, symmetry-randomized leaves. Time manager: stop issuing batches at 2.30 s, `cudaStreamSynchronize` + `cudaDeviceSynchronize`, reply at ≈2.40 s. That gives 0.6 s of margin against thaw latency, the 3-ms quiescence window, stale-sample penalties, and 4-pair tail effects; the shipped harness is run in the exact 4-pair layout for 2,000 games to measure the T1→T3 distribution and I only shorten the margin if the 99.99th percentile is well inside it. Losing ~10% of think time costs ~15 Elo; a single forfeit line costs far more.
- Freeze everything by mid-day 15; the last day is harness soak testing (crash-free over ≥5,000 harness games, zero clock violations, zero off-turn samples), not training.

**Honest forecast.** Realistic throughput after a few days of autonomous debugging: 20–40% of the reference's. ~11 days at that rate ≈ 2.5–4.5 reference-days of self-play. On the author's own curve (days 3–21 ≈ 900 Elo, roughly logarithmic), that is ≈ −650 to −850 Elo before counting any residual bug, plus a smaller inference tax. If I execute nearly flawlessly (≈20% of my probability mass) I land near −600 → S ≈ 0.06–0.09; with one capping bug or a late-starting pipeline (the other 80%) I land at 0.005–0.03. Weighted, ~0.03.

ESTIMATED_SCORE: 0.03

# B) Red-team

I looked hard for anything that makes this cheaper than the printed plan. Findings:

1. **Infrastructure-fault replay is not exploitable.** A replay reuses the same opening and reference seed, so in principle a second try against a now-known reference line would be valuable — but the container is destroyed and no byte persists, so the candidate can't remember the first line. And a candidate-induced fault is charged to the candidate anyway. Closed.

2. **Ply-cap adjudication / mutual-barrier regime.** The reference never saw a ply-600 adjudication in 750k games, so its value head has zero signal for stone-count wins. But reaching ply 600 requires prolonging a no-pass, superko-constrained capture fight for hundreds of plies against an opponent trained on barrier positions; the author's specialist probes (0.000 / 0.006) cover this. Not a shortcut.

3. **Timing/clock.** Freeze-before-drain, off-turn activity = loss, stale samples charged to the mover: I found no residual window. The probes (e)–(g) exercise exactly the race I would have tried. If anything, the fail-closed off-turn rule is a *hazard* for the candidate (any lazily-scheduled driver/allocator activity is a loss), not a lever.

4. **Reference sampling in plies 7–20.** Only among children within 0.03 completed-Q of the best; ~5 Elo. Nothing to farm.

5. **Adversarial-policy transfer (the only genuine, if speculative, lever).** AlphaZero nets have exploitable blind spots (the KataGo cyclic-group attacks), and on a cylinder "rings" are the natural analogue. One could train an adversary against one's *own* two independent-seed agents and keep only exploits that transfer between them, hoping they also transfer to the hidden reference. The author's barrier probes tested hand-crafted objectives, not learned adversaries against AZ victims. This is not a shortcut in the sense of avoiding the hard work — it *requires* a working pipeline first — and I'd put ≤10–15% on a transferable exploit existing given the reference's explicit barrier-rich training. It raises variance, not the expectation.

6. **Everything else** — hidden openings disjoint up to symmetry, full symmetry group disclosed, no color-swap symmetry, engine shared, no draws, no total clock, no network, no persistence, deterministic-but-unobservable reference — checks out. Existing Go/Hex engines can't represent the topology; hand-crafted evaluation probes are ≈0.

Conclusion: the difficulty is exactly where the task says it is — sustained, correctly steered compute — and I found no way around it.

SHORTCUT_FOUND: NO
## Bottom line

**Do not spend the next three days simply changing 64 simulations to 800.** More search will probably help, but your most urgent suspects are:

1. **Too much learning per fresh self-play position, compounded by stale/local replay.**
2. **A rules, value-perspective, or search implementation error.**
3. **Constant-LR regression and weak checkpoint selection.**
4. **Insufficient search quality.**

Your hardware budget is substantial enough for a strong experimental agent. It is **not enough to promise “near the ceiling,”** especially for a novel game without an established strong opponent.

Also, +636 Elo against random does not establish amateur strength. Elo is opponent-pool-dependent, and random is especially uninformative in a connection game. The old-checkpoint regression is the important measurement—provided it is statistically reliable and uses identical evaluation settings.

My recommended default is:

> **Keep the 256×15 network; fix instrumentation and rules/search tests immediately; train with 256-simulation targets, controlled sample reuse, a decaying LR, and a small auxiliary head; increase selected searches to 512 only when measured results justify it. Maintain a league and deploy the best validated checkpoint, not the latest.**

---

## 1. Highest-priority work: first 4–6 hours

### A. Measure these quantities before choosing a simulation schedule

For each run, log:

- Actual neural leaf evaluations/second, cluster-wide.
- Simulations/second, separately: terminal leaves and cache hits make these different.
- Newly generated positions/second.
- Newly retained **training-target** positions/second.
- Training examples consumed/second.
- Sample age and generating-network version at training time.
- Average game length and termination reason.
- Actor network lag.
- Policy entropy, search entropy, illegal probability mass before masking.
- Policy loss and value calibration on recent held-out games.

Define:

\[
R=\frac{\text{training examples consumed}}{\text{new retained training positions generated}}.
\]

Your learner consumes:

\[
3\times1024=3072 \text{ examples/s},
\]

or **265 million examples/day**. That is potentially enormous relative to your generation rate.

**If you generate only 100 fresh usable positions/s, your reuse ratio is about 31.** Raising simulations without slowing learning increases it further.

**Initial target:** test \(R=2,4,8\); use approximately **4** as the starting point, not as a universal optimum. Drive learning with a sample-budget controller rather than “always run at 3 steps/s.”

### B. Evaluate search scaling using your existing checkpoints

Before retraining, test both the current and the stronger older checkpoint at:

- 64 simulations
- 256 simulations
- 1024 simulations

Use no evaluation noise, matched openings, and both player assignments.

Also compare against a deliberately simple tactical agent:

- Take an immediate connection win.
- Prevent an immediate opponent connection when possible.
- Prefer captures that create or stop connections.
- Otherwise use a simple connection-distance heuristic.

This provides useful diagnosis:

| Observation | Likely implication |
|---|---|
| Large improvement from 64 to 256/1024 | Search budget is a real lever |
| Little improvement at any search budget | Network limitations, tactical horizon, or search bugs |
| More search makes the agent worse | Audit search/value/rules immediately; model exploitation is also possible |
| New checkpoint worse even at equal high search | Training regression, not merely insufficient deployment search |

Use hundreds of games for screening. For a final close comparison, around **1,000–2,000 games**, analyzed with opening-pair dependence respected, is much more useful than a small noisy match.

### C. Do not discard useful weights without a reason

A bug-fixed or lower-LR continuation from your strongest checkpoint may have higher three-day value than restarting.

If your reachability requirement genuinely requires **independent training runs**, preserve that distinction: two runs branched from the same trained weights are not independent from-scratch evidence.

---

## 2. Simulations: increase, but protect data freshness

### Recommended schedule

| Training phase | Starting recommendation |
|---|---|
| First few hours from scratch | 128 simulations |
| Main training | 256 simulations |
| Later training, if search scaling and throughput support it | 512 simulations on all or selected moves |
| Validation / submitted opponent | Test 512, 1024, 2048 under actual latency limits |

If continuing a competent checkpoint, start at **256**.

I would **not jump straight to 800–1600 simulations for every self-play move**. Your search is neural-evaluation-heavy; approximately quadrupling simulations can approximately quarter the number of generated positions. That can be worthwhile, but not if the learner keeps consuming the old rate.

There is no general rule that 64 simulations cannot produce a strong agent. The issue is whether its policy-improvement targets are sufficiently better than the network’s existing policy.

### Playout-cap randomization: useful, but not free

A reasonable later configuration is:

- **25–50%** of moves: 512 simulations, retain full policy/value targets.
- Remaining moves: 64 simulations, normally do not retain policy targets.
- Root exploration noise on target-generating searches; fast searches can omit it.
- Sample the move according to the chosen self-play temperature.

For 25% expensive moves:

\[
\bar S=0.25(512)+0.75(64)=176.
\]

That sounds cheap, but only one quarter of positions supplies expensive-search targets. The simulation cost per retained target is approximately:

\[
176/0.25=704.
\]

Thus, **playout-cap randomization can improve trajectories and target quality while sharply reducing the rate of usable training targets**. Your reuse controller must account for that.

For this short project, **uniform 256-simulation self-play is the simpler first upgrade**. Add randomization after measuring the trade-off. If retaining cheap-search positions for value learning, use a separate, controlled value-only stream rather than silently treating them as equivalent policy targets.

### Search implementation improvements can beat another simulation doubling

Prioritize:

- Tree reuse after moves.
- Efficient legal masking and terminal detection.
- Batched inference without excessive queue latency.
- Correct handling of virtual visits/losses, if applicable.
- Sensible first-play urgency, especially at newly expanded nodes.
- Consistent root-visit accounting.
- No duplicate leaf evaluations due to scheduling errors.

Sweep \(c_{\text{puct}}\) over approximately **1.0, 1.5, 2.5** at your new simulation count. There is no reason to assume 1.5 remains optimal.

---

## 3. Auxiliary targets: add one or two, not a research project

**I would not give numerical expected Elo gains for these in CYGO.** There is no evidence base for doing so. Their compute cost is small; their implementation and target-correctness costs matter more.

### Best first choice: future occupancy

Add a per-point, three-class head predicting:

- Empty.
- Stone belonging to the player to move at the input position.
- Opponent stone.

Use either:

1. **Final-board occupancy**, or
2. Occupancy **8–16 plies later**, with terminal positions handled explicitly.

The short-horizon version may be more learnable and capture-relevant. Final occupancy provides dense outcome-associated supervision, but it is not Go territory ownership and should not be called that.

Important implementation details:

- Keep labels in the **input position’s player perspective**, not the future side-to-move perspective.
- Apply board symmetries consistently to every target.
- Average the loss over board points.
- Start with auxiliary coefficient **0.1–0.3**, alongside policy coefficient 1 and value coefficient 1.

### Second choice: connection-specific features

For each player, calculate a simple current-board connection metric:

- Own stones: cost 0.
- Empty points: cost 1.
- Opponent stones: impassable.
- Orthogonal adjacency with cylindrical wrapping.

Predict distance from the top and bottom, or related reachability maps. Normalize or clip distances and represent unreachable points explicitly.

This can teach useful geometry, but:

- It ignores captures.
- It is not a tactical oracle.
- It is largely derivable from the input.

Therefore, I would use it as a **small auxiliary loss or extra input feature**, not a dominant objective. Future occupancy is probably the better first investment.

### Things I would avoid this week

- Ordinary Go territory/ownership labels.
- A complicated “correct winning path” target: there can be multiple paths, and cap wins have none.
- Optimizing stone-count margin on every game: connecting can be correct even when materially behind.
- Multiple handcrafted heads whose losses compete with winning.

**Explicit legal-mask, remaining-ply, and geometry inputs may be worth more than several speculative auxiliary heads.**

---

## 4. Self-play data quality

### Resignation

Initially, **keep resignation disabled while auditing**. Otherwise you can conceal both tactical recovery and terminal-rule bugs.

Once stable:

- Allow resignation only after approximately **40 plies**.
- Require search-root value below roughly **−0.995**, preferably on several consecutive turns by the resigning player.
- Keep **10–20% no-resign games**, selected independently.
- Log when those games *would* have resigned, then measure actual outcomes.

The numerical threshold is only a starting point; values must be empirically calibrated. If long games dominate cost, resignation becomes a high-value optimization. If most games end quickly by connection, its value may be small.

### Temperature

Replace an abrupt “26 plies, then deterministic forever” rule with a tested schedule:

- First **20–30 plies**: \(\tau=1\).
- Next **20–40 plies**: \(\tau\approx0.5\).
- Afterwards: low temperature, such as **0.1–0.25**, or mostly argmax with a small stochastic-game fraction.

Avoid indiscriminate late random moves that turn useful tactical games into blunders. Track repetition of openings and state hashes to decide whether more exploration is needed.

### Root noise

Your \(\alpha=0.3,\epsilon=0.25\) is not obviously wrong. However, consider scaling concentration by the number of legal moves:

\[
\alpha = A/n_{\text{legal}}, \qquad A\in[10,30].
\]

Try \(\epsilon=0.15\)–0.25. Sample noise only over legal actions. In evaluation, disable it entirely.

These are secondary knobs, not the first likely source of a large gain.

### Openings

Replace mandatory random six-ply openings with a mixture:

- **60–80%** empty-board self-play.
- **20–40%** short stochastic openings, preferably drawn from current/previous search policies rather than uniformly random legal play.

Use exact cylindrical symmetries extensively:

- Column rotations.
- Column reflection.
- Top–bottom reflection, if the complete rules are invariant under it.

Apply transformations to history-derived features and all targets too.

A small historical-opponent pool can reduce forgetting. If using cross-checkpoint games, initially retain ordinary policy targets from the current agent’s searched turns; do not blindly imitate arbitrary old or heuristic opponents.

### Replay

Use a **logically shared replay distribution across the run**, or ensure the learner samples all actor/node buffers fairly. Per-node buffers can silently create strongly skewed data.

Size replay by **time**, not just sample count:

- Start with approximately **2–6 hours of fresh retained targets**.
- Monitor age distribution and version lag.
- Optionally reserve **10–20%** for older diverse data if forgetting is observed.
- Keep actor network lag measured in minutes where practical, not many hours.

For a three-day run, a buffer dominated by yesterday’s weak policy can be harmful. Conversely, an extremely narrow window encourages cycling.

The first sample-weighting fix is simply: **do not let low-search targets dominate high-search targets**. Avoid outcome balancing or heavy prioritized replay until the baseline is stable.

---

## 5. Network, inputs, optimizer

### Keep 256×15 for now

This is already a substantial network for 169 intersections. Increasing it would:

- Slow self-play.
- Reduce target generation.
- Force more pipeline changes.
- Have uncertain three-day benefit.

A smaller network with stronger search might even win under fixed wall time, but I would not make architecture size your primary experiment.

### Your three-plane state is incomplete

At minimum add:

- **Remaining ply budget**, broadcast as a normalized plane.
- Explicit row-coordinate / top-edge / bottom-edge information.
- A **legal-move mask input**, supplied by the rules engine.

The ply budget matters because the same board can require different decisions near the cap.

Positional superko means the board alone is not Markov. A legal mask resolves current legality but not all history dependence of future legality. The search still needs exact rule history. A few recent stone planes or a last-move plane can help representation, but **do not make superko exact**.

Your CNN can infer row location through noncircular row padding, but explicit geometry is inexpensive. Verify that circular padding is applied **only across columns**, not rows.

### Optimizer

For a continuation, test **reducing LR immediately to \(3\times10^{-4}\)**. This is a very cheap diagnostic for the regression.

For a fresh run:

- Warm up for roughly **2k–5k steps**.
- Peak Adam/AdamW LR: **\(5\times10^{-4}\) to \(10^{-3}\)**.
- Decay during the run.
- Use approximately **\(10^{-4}\)** for the last quarter.
- Optionally **\(3\times10^{-5}\)** for the final consolidation period.

Tie scheduling to wall-clock budget **and data progress**, especially if the learner is generation-limited. Do not keep \(10^{-3}\) indefinitely merely because loss still moves.

Other defaults:

- Batch **512–1024** is fine.
- Start policy loss weight 1, value MSE weight 1.
- Monitor gradient contributions before changing value weight.
- Retain weight decay near \(10^{-4}\), but distinguish Adam’s coupled L2 from AdamW.
- Prefer **DDP over DataParallel**.
- Audit BatchNorm running statistics and train/eval mode carefully.

Try an **EMA checkpoint** with decay around 0.999–0.9999, evaluated alongside raw weights. SWA is also viable, but averaging across strongly changing policies or stale normalization statistics can hurt. Neither should automatically replace the validated best model.

---

## 6. Throughput and what three days can buy

Let:

- \(B\): actual neural leaf evaluations/second across the actors.
- \(S\): average neural leaf evaluations per played move.
- \(L\): average game length.

Then approximately:

\[
\text{positions/day}=\frac{86400B}{S},
\]

\[
\text{games/day}=\frac{86400B}{SL}.
\]

These are illustrative, **not predictions of your implementation**:

| Cluster inference throughput | Evaluations/move | Mean length | Games/day | Positions/day |
|---:|---:|---:|---:|---:|
| 10,000/s | 256 | 150 | 22,500 | 3.38M |
| 30,000/s | 256 | 150 | 67,500 | 10.13M |
| 100,000/s | 256 | 150 | 225,000 | 33.75M |

At your current learner speed, those correspond to reuse ratios of approximately **79, 26, and 7.9**, respectively, if every played position is retained.

That is why I would investigate learner/generator balance before declaring 64 simulations the main problem.

For example, at \(B=30{,}000\), \(S=256\), and target reuse 4:

\[
\text{training rate}\approx469\text{ examples/s},
\]

or **0.46 steps/s at batch 1024**, not 3 steps/s.

### GPU allocation

Do not reserve learner GPUs just to maximize optimizer steps.

A reasonable starting allocation per 48-GPU run is:

- **2–4 GPUs learning**.
- Most remaining GPUs doing self-play inference.
- A small evaluation allocation, possibly time-shared.

Benchmark one, two, and four learner GPUs at the **needed** sample consumption rate. Move excess capacity to generation. Avoid oversubscribing GPUs with actor processes without checking actual inference batching efficiency.

### Realism

**A much stronger agent in three days is realistic. Demonstrating reference-grade strength is the harder part.**

De-scope:

- Major network redesign.
- A large auxiliary-head research program.
- Massive search everywhere.
- A claim of “amateur/master-level” based on random-play Elo.

Do not de-scope:

- Independent rules tests.
- Fixed evaluation league.
- Correct sample accounting.
- Deployment search optimization.
- Testing against tactical and historical opponents.

If benchmark inference permits it, **a validated network with 1024–2048 simulations can be a substantially stronger reference than the same network at 64**, without spending the training budget generating everything at that search level.

---

## 7. Silent errors I would check first

### Rules and state

1. **Cylinder adjacency:** captures, liberties, flood fills, and connections all wrap columns consistently.
2. **Deduplicated liberties:** especially around wrapped chains.
3. **Terminal ordering:** apply captures and legality exactly as specified before checking the relevant winning condition.
4. **No-legal-move behavior:** explicitly defined because passing is prohibited.
5. **Ply cap:** exact off-by-one behavior and tie handling.
6. **Positional superko hash:** absolute-color board identity, not side-to-move-relative input planes; side to move must not accidentally turn it into situational superko.
7. **Branch-local history:** search branches must not contaminate one another.
8. **State merging:** board-identical positions can differ in legality, future superko constraints, and remaining plies.

Build an independent slow Python rules implementation and differential-test random legal sequences against C++. Include targeted wrap, capture, superko, and near-terminal fixtures.

### Search and targets

9. Value perspective alternates correctly during backup.
10. Terminal values use the correct leaf/player perspective.
11. \(z\) is from the player-to-move perspective of **each stored sample**.
12. Q values and PUCT selection agree about whose payoff is being maximized.
13. Illegal policy logits are masked before normalization.
14. Root noise never assigns probability to illegal moves.
15. Visit targets contain the intended search visits and sum correctly.
16. Root reuse does not retain inconsistent priors, history, or player perspective.
17. Cached evaluations do not mix network versions or incompatible input states.
18. A one-ply forced win is selected; a forced loss is recognized; increasing search improves small solved fixtures.

### Training and infrastructure

19. Replay is not effectively one node’s stale shard.
20. Symmetries transform policies, future-board labels, and history consistently.
21. Resumes restore optimizer, scheduler, replay/sample counters, actor versions, and RNG states as intended.
22. No inference accidentally runs in training mode or with bad BatchNorm statistics.
23. Evaluation has no training noise or hidden temperature differences.
24. Claimed regressions use the same search budget, rules version, openings, and termination handling.

---

## My concrete four-day plan

**Hours 0–6**

- Rules/search differential tests and forced-tactic fixtures.
- Measure generation, reuse, replay age, and actor lag.
- Existing-checkpoint search-scaling matches.
- Introduce a fixed evaluation league.
- Short continuation test at lower LR.

**Hours 6–60**

- Same improved core recipe for both independent runs.
- 256×15 network.
- 128→256 simulations if fresh; 256 immediately if continuing.
- Legal-mask, remaining-ply, and geometry inputs.
- One future-occupancy auxiliary head.
- Reuse-controlled learning, initially around 4 with measured adjustment.
- Decaying LR.
- Regular league evaluation and best-checkpoint retention.

**Hours 60–72**

- Increase to 512-simulation targets only if the measured trade-off supports it.
- Lower-LR consolidation.
- Test raw versus averaged checkpoints.
- Screen deployment search budgets.

**Final day**

- Freeze training and select by validation, not recency.
- Run a larger final league and tactical suite.
- Test exact packaged rules, inference, latency, and recovery.
- Report strength against multiple meaningful opponents with uncertainty.

**If you only implement three changes: audit correctness, control train:generate ratio, and move to 256-simulation targets with LR decay and real checkpoint selection.** Those have the best expected strength-per-effort here.
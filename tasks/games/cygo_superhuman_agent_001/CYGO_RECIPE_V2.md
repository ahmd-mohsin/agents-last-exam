# CYGO training recipe v2 — expert-reviewed upgrade plan (2026-09-13)

Reviewed by GPT-6-Astra + Fable-5.1 (max reasoning; full transcripts in consult/). They
INDEPENDENTLY AGREE the plateau is a **pipeline pathology, not a sims problem**, and both
say a genuinely strong 13x13 agent in ~3 days on 48xA100 is realistic *if the pipeline is
fixed* (Fable: KataGo went 0->superhuman 19x19 in ~400 V100-days; ~290 V100-day-equiv here
on a 44%-size board -> 3-4M games in 3 days).

## Root cause of the plateau (consensus)
- **Replay window far too small.** 1.5M samples / ~150 plies ≈ **~10k games**; it turns over
  every ~10 min, so the net trains only against the last checkpoint's style -> rock-paper-
  scissors cycling + the value head memorizes ~10k games. This explains BOTH the "loses to
  its own 110k-steps-ago self" (−70 Elo) AND value-loss≈0. KataGo/LZ hit exactly this and
  used a ~500k-game window + value weight 0.25.
- **A likely correctness bug too:** vs-random should be ≥99.9%; we measured 97.5% (losing
  2.5% to random) — a red flag.

## AUDIT FIRST (both models insist; cheap, may reveal the real cap)
1. Net vs uniform-random, argmax, 400 sims, no Dirichlet, 200 games, both colors: expect
   **≥99.9%**. Dump every loss.
2. **No-legal-move terminal**: when the side to move has ZERO legal moves (all empties
   suicide/superko-illegal), who loses? If undefined/crash/wrong-side -> corrupts terminal
   values. (Check cygo engine play()/legal_moves paths.)
3. Wrap-around connection detection (group connecting top<->bottom through the column seam);
   capture-with-wrap; positional-superko hashing.
4. **Value perspective**: z stored relative to THAT position's side-to-move (not Black);
   one-ply-before-forced-win -> value≈+1, flipped -> ≈−1.
5. **FPU** (unvisited-child Q init): currently 0 -> over/under-explores; use KataGo FPU
   `Q_init = Q_parent − 0.2·sqrt(Σ visited priors)` (~50-100 Elo).
6. **Sims-scaling test**: same net at 64/256/1024 sims vs a fixed opponent. If more search
   doesn't help -> bug/net-limited, not sims.
7. Ship a ~100-position **tactical unit set** (win-in-1 connect, must-block, capture-to-
   disconnect, superko-illegal) + monitor top-1 policy & value-sign accuracy every ckpt.

## Recipe fixes, priority by Elo-per-effort
1. **Replay window games-based & large** (config): start ~40k games, grow to ~250k by day 2;
   keep **reuse ratio 2-4** by throttling learner steps to generation. *"This alone likely
   removes the regression."* Store compactly (bitboards + int16 policy/visits + int8 owner).
2. **Optimizer/LR**: AdamW (decoupled wd 1e-4), grad-clip 1.0, warmup 2k, 1e-3 -> 3e-4 @50%
   -> 1e-4 final 15%; **fp32 losses + fp32 master weights** (bf16 tanh/log-softmax too coarse);
   **SWA** over final ~10-12h (+20-40 Elo).
3. **Playout-cap randomization** (not flat sims): 25% moves full **400 sims** record
   (pi, z, ownership); 75% fast **80 sims** record (z, ownership) only — never train policy on
   fast moves. Mean ~160 sims/move. Day-2 -> 600/100 if throughput allows.
4. **Ownership aux head** (final board, +1 own/−1 opp/0 empty, side-to-move perspective),
   loss weight ~1.5 (169 targets/pos vs 1 -> ~30-50% fewer samples to a given Elo). Optional
   winning-path mask (points of the group that connects) ~equal value for a connection game.
5. **Feature planes**: legal-mask, ply-count/600, stone-difference, remaining-ply. Cheap,
   possibly worth more than extra aux heads.
6. **Inference throughput** (buys sims AND games): batch 64 is far too small for a 256x15 net
   -> 256-512 games/actor or 2-4 actor procs/GPU + CUDA graphs/torch.compile/TensorRT +
   virtual-loss multi-leaf. ~3-5x more positions/s.
7. **Openings**: policy-sampled 8-12 plies at T=1.5-2 (no search) + 10% 2-6 random plies +
   10% mid-game forks (KataGo fork trick). Not flat random-6-ply.
8. **Dirichlet** alpha≈0.08-0.1 (10/#legal), eps 0.25, on LEGAL moves only (0.3 is chess-sized).
9. **Resignation** (AFTER audit): root value < −0.95 for 3 consecutive own moves & ply>30;
   15% no-resign; keep false-resign <5%. Shortens games ~30-40% -> more value diversity.
10. **Deploy the BEST validated checkpoint via a gating league, not the latest** (the −70
    regression means the 272,900 snapshot is currently our best, not "current").
11. **Net size**: 192x15 (self-play is the bottleneck; ~1.8x cheaper eval than 256x15 for
    marginal Elo at ~3-4M games). Keep 256x15 only if inference batching lands 3x+. No
    net-size progression in 3 days.
12. Optional highest-ceiling-per-sim: **Gumbel-AlphaZero root** (policy improvement even at
    32-64 sims) — only if PCR path is already stable (a real C++ change).

## Throughput target (Fable's envelope, 192x15)
Well-batched A100 ~5-10k evals/s; at ~160 mean sims -> 30-60 moves/s/GPU; 42 self-play GPUs
-> ~1.5-2.5k positions/s, ~12-20 games/s with resign -> **~1-1.7M games/day, 3-4M in 3 days**
= strong dan-level range on 19x19; 13x13 easier. Miss the batching fix -> ~1/4 -> club level.

## 4-day plan (from today), honest
- **Day 0:** run the AUDIT; fix any bug; implement the high-ROI subset (large replay window +
  reuse throttle + AdamW/LR/fp32 + ownership head + feature planes + inference batching +
  PCR + openings + Dirichlet); relaunch BOTH seeds fresh with v2.
- **Days 1-3:** train on 48 GPUs/seed; batched Elo vs snapshots daily; enable resign + SWA;
  gate/keep best.
- **Day 4:** freeze the best-validated checkpoint as the reference; stage_task_data.py ->
  input/+reference/ bundle; package + submit.
- **Risk:** the v2 pipeline rewrite is ~1-1.5 days of careful coding+testing; 4 days is tight
  but plausible if we implement the prioritized subset (1-6) and defer the rest. The dominant
  practical risk is the cluster CHURN (tunnels drop, instances die ~hourly-to-daily, creds
  expire) — each interruption costs wall-clock even though S3 resume is lossless.

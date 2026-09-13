# CYGO build status

Design: DESIGN.md (revision-6 spec, validated by the Fable/Astra adversarial loop;
final frontier self-estimates ~0.06/0.12, no exploit). curation/ holds the attack logs.

## Done
- **Rules engine** (`greenland/build_cygo/cygo_engine.py`): 13x13 cylinder (columns wrap,
  rows don't), Go capture, suicide illegal, positional superko (Zobrist board hash),
  win-by-top-bottom-connection, no-legal-move loss, ply-cap-600 adjudication (ties->White).
  `test_engine.py`: 8/8 rule tests pass (wrap, no row-wrap, capture, suicide, superko,
  connection win) locally and on the box. Board size parameterized (proofs on 5/7).
- **AlphaZero pipeline** (`greenland/build_cygo/azero.py`): column-circular (cylinder-
  equivariant) ResNet with policy+value heads; negamax PUCT MCTS with Dirichlet root
  noise; self-play; train (CE policy + MSE value); arena vs random; checkpoint/resume
  (survives the ephemeral box). Fixed a negamax backup sign bug (W[a] must accumulate the
  node-perspective value -v_child) that had suppressed learning.

## Proof-of-pipeline (in progress)
5x5 CYGO on one A100, validating that arena win-rate vs random rises across iterations.
~3 min/iter (latency-bound: single-example synchronous net evals in Python MCTS).

## Roadmap to a submittable task (multi-week)
1. **C++/bitboard engine** — DONE (first cut). `greenland/build_cygo/cygo_cpp.cpp`
   (pybind11): byte board + precomputed wrap neighbors + flood fill, incremental Zobrist
   superko, ply-cap. Parity-verified against the Python oracle over 200+30 random full
   games (identical board/winner/legal-set every ply). Throughput 60.9k moves/s/core
   INCLUDING full legal-move regeneration each ply (early-exit liberty check was the key
   3.6x win over the naive full-count); raw play() is much faster. Still TODO here: the
   52 transforms/coset reps, make/undo to avoid per-candidate board copies, and a fast
   "obviously legal" path to push toward the ~2M/s target.
2. **Batched GPU MCTS** — IN PROGRESS. `greenland/build_cygo/azero_batched.py`: G parallel
   games, all pending leaves evaluated in ONE net forward (batch = #games), negamax PUCT +
   Dirichlet, on the C++ engine at 13x13. Measured 103 positions/s at batch 16 (~30x the
   single-example loop, on the full 13x13 board). BUT scaling to batch 128 stalls: the pure
   PYTHON tree loop (per-sim descend + PUCT over up to 169 actions) dominates once the GPU
   batch is fed. Next: move the MCTS itself into C++ (batched tree + PUCT in cygo_cpp,
   calling out to a batched LibTorch/py net eval), which removes the Python per-sim
   overhead and is the true throughput unlock for a 13x13 run.
2b. **Distributed actor/learner across all 24 GPUs / 3 nodes** — DONE and LIVE.
   `replay_server.py` (stdlib HTTP replay+param server on the main node, reached by workers
   at 10.3.24.212:8000), `actor.py` (fetch weights -> batched self-play on one GPU -> POST
   samples), `learner.py` (pull batches -> train -> PUT weights -> checkpoint), and
   `launch_cluster.sh` (main: server + learner GPU0 + 7 actors GPU1-7; each worker: 8 actors
   via passwordless SSH on :2222, code+`.so` scp'd over). Verified: all 3 nodes running
   actors (local + both workers), buffer filled 0->375k samples in ~7 min, weights broadcast
   (version 51), learner training ~8 steps/s at batch 1024, net ch=96/blocks=10.

## Fixes applied (2026-09-11, items 1-3)
- **(1) Async evaluator DONE**: `evaluator.py` runs on its own GPU (GPU1), polls weights,
  logs arena-vs-random; the learner no longer blocks on arena (arena code removed from
  learner.py). Note: its single-example arena is slow to log; could be batched later.
- **(2) Durable checkpoints DONE**: S3 is blocked for the box role and /mnt/nvme isn't
  mounted, so durability = laptop-side pull loop `ckpt_pull_loop.sh` (scp learner ckpt ->
  build_cygo/ckpt_backups/ every 2 min; verified pulling 21MB). learner.save_ckpt is atomic
  (tmp+rename). On instance death: re-provision, scp the backup ckpt back up, relaunch
  (learner resumes from run_dist/ckpt.pt).
- **(3) C++ batched MCTS DONE**: `cygo_cpp.Batch` (tree+PUCT+Dirichlet+encode in C++;
  select()/apply()/advance()/samples() with net eval batched in torch). `selfplay_cpp.py`
  drives it; all actors now use it (actor.py). Correct (pi normalized, games terminate).
  Cluster live at games=64/sims=64 on all 24 GPUs, ~94 samples/s/actor, buffer to 1.08M.
- **Distributed launch scripts**: start_local.sh (main), start_workers.sh (workers),
  both detached (robust to SSM tunnel drops, which truncate compound commands).

## Scaling + survivability pieces (2026-09-11) — for the 48-GPU / 24h-death run
- **(1) Durable S3 checkpoint + auto-resume — DONE & VERIFIED.** `s3ckpt.py` (boto3 + pod
  IRSA; the `aws` CLI can't, boto3 can) pushes each learner ckpt to a STABLE prefix
  `s3://greenland-intern-artifacts-703671891219-us-east-2-an/cygo_resume/<CYGO_BOXTAG=cygo-run1>/global_step_N/ckpt.pt`
  + tracker written LAST, prunes to keep_last=3. Verified: pushed steps 4100-4300 (213MB
  each), pull downloads the correct latest (step 4300 loaded). learner.py resumes from S3 on
  startup (falls back to local). Survives the 24h box death: fresh node -> pull latest ->
  resume. `deploy_cluster.sh` = one-command re-provision from the laptop.
- **(2) Multi-GPU DataParallel learner — DONE.** learner.py `--gpus K` wraps the net in
  DataParallel (validated forward/backward on 2 GPUs); start_local.sh places learner on
  GPU0..K-1, evaluator next, actors after. Lets the trainer keep up with 48 GPUs of self-play.
- **(3) Per-node replay + weight broadcast — DONE (code).** learner.py `--servers` samples
  round-robin from every node's replay server and PUTs weights to all; start_workers.sh
  (PERNODE=1) runs a replay server on each worker with its actors POSTing locally (no
  cross-node sample traffic). Removes the single-server bottleneck at 48 GPUs.

### Deploy runbook (when a fresh cluster is given)
1. Update `SSM_TARGET` in greenland-connect.sh -> restart tunnel (port 1077).
2. From build_cygo/: `MAIN_IP=<main ENI ip> WORKERS="<w1> <w2> ..." LGPUS=4 ./deploy_cluster.sh`
   (probe the reachable MAIN_IP: only the ENI ip workers can curl works; server must be up).
3. Learner auto-resumes from S3 `cygo-run1`. Verify: `python3 s3ckpt.py latest` climbs.
On 24h death: repeat 1-2; resume is automatic from S3.

## ALE task package + two-cluster training (2026-09-11, choice A)
- **Package wired (gradeable):** main.py (cua_bench lifecycle: stage input/ rules+engine+
  baseline+harness, hidden reference/ agent; evaluate() runs scripts/match_harness.py
  candidate-vs-hidden-reference over 1000 games -> score=clip(2*winrate)), task_card.json,
  scripts/ (match_harness.py, agent_proto.py, cygo_cpp.cpp, cygo_engine.py, azero.py,
  arena_batched.py, selfplay_cpp.py), scripts/ISOLATION.md (DESIGN sec 2 compute-isolation
  spec: per-game container, MPS 24GB, seccomp allow-list, freeze-then-drain clock, DCGM/NVML
  off-turn oracles + injected-kernel validation). Still to ship for the evidence bundle:
  replay_oracle.py + the injected-kernel raw dumps, and the trained reference_agent/ + built
  input/ and reference/ bundles (produced once training completes).
- **Two independent-seed training runs LIVE** (reachability evidence): cluster 1 cygo-run1
  (us-east-2/acct .842/port 1077) at step ~16k; cluster 2 cygo-run2 (us-west-2/acct .388/
  port 1078, profile greenland2) fresh seed at step ~8k. Fully isolated (distinct port,
  profile, account, region, S3 prefix); both S3-durable + auto-resume; cluster 1 untouched
  when cluster 2 was added. C2 survived a tunnel drop (detached procs + S3).
- **4B aux heads: deferred** (would reset the live seeds; both share the current recipe).

## 4C match harness — DONE & VALIDATED (2026-09-11)
`agent_proto.py` (a submission-style agent speaking the newgame/opening/move/quit stdio
protocol; policy|mcts|random modes) + `match_harness.py` (referee: forced 6-ply openings,
authoritative C++ engine, per-move wall-clock budget, illegal/malformed/crash/timeout =
loss, score = clip(2*winrate)). Validated on the box: trained policy vs random -> 1.0;
broken candidate (no protocol) -> 0.0 (grader penalizes correctly); random vs random ->
~1.0 (equal strength). This is the gradeable core.
REMAINING for 4C: (a) compute-isolation wrapper per agent process (per-game container, MPS
24GB cap, seccomp allow-list, freeze-then-drain turn clock, DCGM/NVML off-turn oracles +
injected-kernel validation — DESIGN sec 2; the harness core is isolation-agnostic and the
layer wraps each agent subprocess), (b) wire evaluate() in main.py to run match_harness with
candidate=submitted agent vs the hidden reference agent.

## Item 4 progress (2026-09-11)
- **(4A) Batched evaluator + Elo — CODE DONE**: `arena_batched.py` (policy_vs_random,
  net_vs_net, elo_from_winrate; batches every net move across all games since CYGO has no
  passing so all games share side-to-move per round) + `evaluator.py` reads weights from the
  LOCAL checkpoint (avoids the contended server) and logs policy-vs-random + Elo-gain-vs-
  frozen-baseline. Verified correct foreground (fast at ch96). Flaky to keep alive detached
  in this environment (SSM tunnel drops truncate launch commands); it is a monitor, not the
  deliverable.
- **Net upgrade (key fix)**: the ch96/blocks10 net left GPUs ~25% idle AND barely learned
  (policy loss flat ~3.7). Switched to AZ-scale **ch256/blocks15** -> learner GPU0 at 100%
  (20GB), self-play actors 32-78%, and real capacity to learn (fresh run; value loss rising,
  which the tiny net never did). This is the correct compute allocation and directly
  addresses "parallelize everything to GPU".
- **(4B) aux heads + specialist starts — NOT STARTED** (occupancy/connection-distance heads
  in the net + start-position regimes in cygo_cpp.Batch).
- **(4C) match + compute-isolation harness — NOT STARTED** (DESIGN sec 2: per-game
  containers, MPS 24GB cap, seccomp allow-list, freeze-then-drain turn clock, DCGM/NVML
  off-turn oracles + injected-kernel oracle validation). Largest remaining piece; the ALE
  gradeable deliverable.

### Environment reality (impediment to the multi-week run)
The SSM tunnel drops frequently (truncating compound commands; mitigated by shipping
detached scripts) and the main node has restarted mid-session (wiping ~/cygo and the on-node
checkpoint — hence the laptop pull-loop for durability). A true 21+3-day reference run needs
either a stable reserved cluster or fully automated re-provision+resume on node death
(re-scp code, restore ckpt from laptop/S3, relaunch). Currently resume is manual.

## Remaining before the real 24-day run (item 4 + polish)
- **Faster evaluator**: replace the evaluator's single-example arena with a batched
  net-vs-random (C++ Batch) so strength is logged promptly; add net-vs-previous-checkpoint
  Elo tracking for a real progress curve.
- **Full net/regime**: aux heads (occupancy, connection distance), playout-cap randomization,
  resign+no-resign, G-augmentation, specialist start regimes (DESIGN sec 3); size increase
  mid-run.
- **Match + isolation harness** (DESIGN sec 2) for grading, with the injected-kernel oracle
  evidence Astra asked for.
3. **Full net + AZ regime** at 13x13: aux heads (final occupancy, connection distance),
   playout-cap randomization, resign+no-resign, G-augmentation, the specialist start
   regimes (barrier-rich, late-game, superko-history, capture-race) from DESIGN sec 3.
4. **Reference training**: 21-day self-play + 3-day adversarial hardening on 8xA100, with
   robust checkpoint/resume across instance deaths; then a second independent-seed run
   for reachability evidence.
5. **Match harness + compute isolation** (DESIGN sec 2): per-game containers, MPS 24GB
   cap, seccomp allow-list, freeze-then-drain turn clock, DCGM/NVML off-turn oracles.
   Astra's residual objections are all about proving THIS airtight — needs executable
   evidence (the injected-kernel oracle validation suite).
6. **Package** on the ALE lifecycle (main.py / task_card.json / verify) once the reference
   and harness exist.

The hard, long part is (2)+(4): actually training a strong-enough reference and the
throughput to do it. Design and rules are locked; execution is the remaining cost.

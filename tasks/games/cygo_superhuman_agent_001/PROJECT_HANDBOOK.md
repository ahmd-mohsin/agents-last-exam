# CYGO Project Handbook — complete record

Exhaustive record of the CYGO (Cylinder Gonnect) Agents' Last Exam (ALE) contribution:
what the task is, every file and where it lives, the training run, the measured results,
the submission mapping, and how to reproduce or resume. Written 2026-09-16.

---

## 0. TL;DR

- **Task:** `games/cygo_superhuman_agent_001` — build a strong agent for **CYGO**, a novel
  13×13 cylindrical Go variant invented for this exam. Graded by win-rate over 1000 games
  vs a **hidden AlphaZero-trained reference**; `score = clip(2·winrate, 0, 1)`.
- **What makes it last-exam:** you must *achieve* playing strength on a game with zero prior
  art (no engine/weights/book/literature transfers). The plan (AlphaZero) is given; reaching
  the bar under a single-GPU match budget is the hard part.
- **Deliverable status:** reference net trained (distributed AlphaZero on ~48×A100 across two
  independent seeds), packaged, verified end-to-end, honest metrics, form fields + evidence
  PDF ready.
- **Shipped reference:** `cygo-run2 global_step_977400` (more-trained of two seeds).
- **Code home:** fork `github.com/ahmd-mohsin/agents-last-exam`, branch `cygo-last-exam-task`.

---

## 1. The game (CYGO = Cylinder Gonnect)

13×13 board. Authoritative rules in `scripts/cygo_rules.json` and the engine.

- **Cylinder topology:** columns wrap (column 12 is adjacent to column 0); rows do **not** wrap.
- **Placement + capture:** Go rules — place a stone; a group with no liberties is captured;
  **suicide is illegal**.
- **Positional superko:** a move may not recreate any prior whole-board position (Zobrist
  board hash + history set).
- **Win condition:** connect the **top edge (row 0)** to the **bottom edge (row 12)** with one
  friendly group. Also: opponent with **no legal move loses**; at **ply-cap 600** the game is
  adjudicated by stone count (**ties → White**).
- **No passing.** Because there is no pass, all active games share the side-to-move each ply
  (exploited for batched play).
- **Symmetry:** 52-element group = 13 column shifts × {identity, left-right mirror} ×
  {identity, top-bottom reflection}. Used for data augmentation and a cylinder-equivariant net.

Index convention: `idx = col*13 + row`.

---

## 2. Where EVERYTHING is stored (storage map)

| Thing | Location | Notes |
|---|---|---|
| Task package (code, design, docs) | git fork `ahmd-mohsin/agents-last-exam` branch `cygo-last-exam-task`, path `tasks/games/cygo_superhuman_agent_001/` | Upstream is `rdi-berkeley/agents-last-exam` (remote `origin`); fork is remote `fork`. |
| Training code (learner/actor/etc.) | `greenland/build_cygo/` on the laptop | **gitignored** (`greenland/`) — colocated with creds/instance IDs. Preserved durably in S3 code bundle. |
| Cluster connection + creds | `greenland/*.sh`, `greenland/bedrock.env` | gitignored. |
| Durable checkpoints (both seeds) | S3 `s3://greenland-intern-artifacts-703671891219-us-east-2-an/cygo_resume/<tag>/` | `tag` ∈ {cygo-run1, cygo-run2}; `global_step_N/ckpt.pt` + `latest_checkpointed_iteration.txt` tracker, keep-last-3 by recency. |
| Permanent milestone snapshots | S3 `.../cygo_ladder/<tag>/` | e.g. `v2_step_735350.pt`, `step_272900.pt`. |
| Self-bootstrap code bundle | S3 `.../cygo_resume/code/bundle.tgz` | node pulls this to rebuild itself. |
| Final reference net (shipped) | S3 `cygo_resume/cygo-run2/global_step_977400/ckpt.pt` **and** laptop `/tmp/cygo_c2_977400.pt` **and** `task_data/reference/reference_agent/weights.pt` | 204 MB. Gitignored in the repo (`*.pt`). |
| Second seed final net | S3 `cygo_resume/cygo-run1/global_step_712200/ckpt.pt` + laptop `/tmp/cygo_c1_712200.pt` | 204 MB. |
| Upload bundles for the form | `tasks/games/cygo_superhuman_agent_001/dist/cygo_{input,reference}.tgz` | gitignored (`*.tgz`); reference bundle → Drive/direct upload. |
| Evidence PDF | `tasks/games/cygo_superhuman_agent_001/dist/verification_evidence.pdf` | force-added to git (small). |

**S3 access note:** the bucket is owned by account `703671891219`. On-node access is via pod
IRSA (boto3 works; `aws` CLI historically didn't). From the laptop, read it by assuming the
bucket-owner role: `aws sts assume-role --role-arn arn:aws:iam::703671891219:role/Intern
--role-session-name pull` (works via either `greenland`/`greenland2` profile), then export the
returned creds. (Server-side `cp` needs a tagging perm the role lacks; download+reupload
instead.)

---

## 3. Task package tree (`tasks/games/cygo_superhuman_agent_001/`)

```
main.py                 cua_bench lifecycle: load()/setup()/evaluate(). evaluate() runs
                        match_harness candidate-vs-hidden-reference over 1000 games,
                        returns score=clip(2*winrate). Stages input/ visible, reference/ hidden.
task_card.json          Task metadata: id, prompt, inputs, reference files, evaluation,
                        difficulty (honest: curation estimate, not measured).
DESIGN.md               rev-6 spec: rules, compute-isolation design, training regime.
BUILD_STATUS.md         Running build log across the multi-week effort.
CYGO_RECIPE_V2.md       The v2 training recipe (post-plateau fixes) write-up.
SUBMISSION_FORM.md      Copy-paste values for every ALE web-form field.
PROJECT_HANDBOOK.md     This file.

consult/                Expert reviews of the design:
  gpt-6-astra.md          GPT-6-Astra feedback.
  fable-5-1.md            Fable 5.1 feedback.
curation/               4-round adversarial red/blue difficulty curation logs:
  round{1..4}_attack_{astra,fable}.md

scripts/                Engine + harness + agent (the source of truth copied into bundles):
  cygo_rules.json         Authoritative rules + 52-symmetry group + adjudication.
  cygo_engine.py          Python rules oracle.
  cygo_cpp.cpp            C++/pybind11 engine (byte board, wrap neighbors, flood-fill
                          liberties, incremental Zobrist superko, ply-cap) + batched MCTS
                          class `Batch`. Built to cygo_cpp*.so in-VM.
  azero.py                Column-circular (cylinder-equivariant) ResNet: policy+value heads.
  agent_proto.py          Protocol agent (newgame/opening/move/quit over stdio); modes
                          policy | mcts | random; temp/noise flags.
  match_harness.py        The referee: forced 6-ply openings, per-move wall-clock budget,
                          illegal/malformed/crash/timeout = loss, score=clip(2*winrate).
  batched_match.py        Fast G-parallel net-vs-net / net-vs-random Elo tool
                          (supports asymmetric --sims-a/--sims-b for the sims-scaling audit).
  arena_batched.py        policy-vs-random + net-vs-net + elo helpers.
  selfplay_cpp.py         Self-play via the C++ batched MCTS (actor core).
  stage_task_data.py      Assembles input/ + reference/ from scripts/ + a reference ckpt.
  starter_run.sh          run.sh template shipped to the agent.
  task_brief.md           Agent-facing brief.
  ISOLATION.md            Compute-isolation spec (per-game container, MPS 24GB cap, seccomp
                          allow-list, freeze-then-drain turn clock, DCGM/NVML oracles).

task_data/              Generated by stage_task_data.py — the actual upload payload:
  input/                  AGENT-VISIBLE (read-only at task time):
    task_brief.md, cygo_rules.json
    engine/   (cygo_engine.py, cygo_cpp.cpp)
    baseline/ (agent_proto.py + engine)         random-rollout smoke opponent
    harness/  (match_harness.py, agent_proto.py + engine)  exact grader
    starter/  (run.sh, agent_proto.py + engine)
  reference/              EVALUATOR-ONLY (staged at eval):
    reference_agent/ (run.sh, weights.pt [204MB], agent_proto.py + engine)
    reference_metrics.json    score↔Elo map, shipped-checkpoint id, MEASURED strength,
                              reachability + honesty note.

dist/                   Upload artifacts (gitignored except the evidence):
  cygo_input.tgz (40 KB)        → form "Input Materials"
  cygo_reference.tgz (41 MB)    → form "Reference Output" (contains the 204MB weights)
  verification_evidence.{html,pdf}   → form "Evidence"
```

---

## 4. Training infrastructure (`greenland/build_cygo/`, gitignored)

Distributed AlphaZero across two isolated 6-node p4d.24xlarge clusters (8×A100 each; main +
5 workers = 48 GPUs/cluster).

| File | Role |
|---|---|
| `learner.py` | Pull batches from replay servers → train (policy CE + value MSE) → PUT weights → S3 checkpoint. AdamW + warmup→cosine LR + grad-clip. `--fresh-opt` continuation keeps global step monotonic while restarting the LR schedule (`sched_offset`). |
| `actor.py` | Fetch weights → C++ batched self-play on one GPU → POST samples. |
| `replay_server.py` | Per-node stdlib HTTP replay + param server; `--cap` sets buffer size (4,000,000). |
| `selfplay_cpp.py`, `azero_batched.py` | Self-play cores (C++ Batch MCTS driver). |
| `azero.py`, `cygo_cpp.cpp`, `cygo_engine.py` | Net + engine (same as package `scripts/`). |
| `evaluator.py`, `arena_batched.py`, `batched_match.py`, `match_harness.py`, `agent_proto.py` | Eval / Elo / grading. |
| `s3ckpt.py` | Durable S3 checkpoint push/pull + code bundle. **Prune by recency** (not step number) + monotonic keys — the fix for the durability bug (see §6). |
| `deploy_cluster.sh` | One-command laptop deploy: scp code → build engine → push code bundle → start server+learner+eval+actors → launch workers. Env: `LOCAL_PORT CYGO_BOXTAG MAIN_IP WORKERS LGPUS CH BL GAMES SIMS CAP RESUME_STEP FRESH_OPT`. |
| `start_local.sh` / `start_workers.sh` | Main-node / worker launch (detached, tunnel-drop-robust). |
| `bootstrap_from_s3.sh` | Node self-bootstrap: pull code bundle from S3, build, launch (only a tunnel needed). |
| `bump_sims.sh` | Restart ONLY actors at a new sims count (learner+buffer untouched) — used to raise self-play sims 64→256 without losing progress. |
| `test_engine.py`, `parity_bench.py` | Engine rule tests + Python/C++ parity. |
| `ckpt_pull_loop.sh`, `restart_*.sh`, `launch_cluster.sh` | Older helpers. |

**Cluster identities**
- cygo-run1: laptop port 1077, `CYGO_BOXTAG=cygo-run1`, us-east-2, acct 072510399842, profile `greenland`.
- cygo-run2: laptop port 1078, `CYGO_BOXTAG=cygo-run2`, us-west-2, acct 144991380388, profile `greenland2`.

**Connectivity** (`greenland/`): `greenland-auth.sh` (interactive Midway+isengard, run via `!`),
`greenland-connect.sh` / `greenland-connect-2.sh` (SSM port-forward; `SSM_TARGET` env-overridable;
use the `tunnel` subcommand to skip interactive auth), `monitor_clusters.sh`, `bedrock.env`.
The instances are 24-h SDB jobs that get replaced; each replacement supplies a new
`SsmManagedInstanceId` (`mi-...`) which is the tunnel target. On the memory-pressured laptop,
persistent background tunnels get reaped — use on-demand tunnels (bring up, query, tear down
in one command).

---

## 5. Training recipe (final)

- **Net:** cylinder-equivariant ResNet, ch=256, blocks=15, policy+value heads.
- **Self-play:** C++ batched MCTS, **sims=256**, PUCT c=1.5, **Dirichlet α=0.1** root noise,
  temperature for first 30 plies, distinct balanced openings.
- **Replay:** per-node servers, **4,000,000-game window**, learner samples round-robin.
- **Optimizer:** AdamW (wd 1e-4), LR warmup 2000 → cosine 1e-3→1e-4, grad-clip 1.0,
  batch 1024, value-weight 1.0, DataParallel over 4 GPUs.
- **Durability:** every checkpoint pushed to S3 (tracker written last, keep-last-3 by recency,
  monotonic global step); fresh node auto-resumes latest.

---

## 6. What happened (chronology + key fixes)

1. **Design + curation.** rev-6 spec; 4-round adversarial curation (Fable blue, Astra+Fable
   red) → self-estimate 0.06–0.12, no shortcut. Expert consults archived.
2. **Engine + pipeline.** C++ engine parity-verified vs Python oracle; batched MCTS moved into
   C++ for throughput; distributed actor/learner + per-node replay + S3 durability.
3. **Plateau.** Early runs plateaued: value loss collapsed to 0.000 (value **memorization**
   from a too-small replay window). Fix = **4M-game window** (+ AdamW/cosine, Dirichlet 0.1).
4. **Durability bug (found + fixed).** `--fresh-opt` reset the step to 0, so new checkpoints
   were numerically *smaller* than retained pre-v2 steps; `keep_last=3` pruned by step *number*
   and deleted every new checkpoint. Fix: **prune by recency**, keep global step **monotonic**.
   Recovered c1 by seeding `cygo_resume` from a ladder backup; thereafter every death
   auto-resumed cleanly.
5. **Strength audit.** Correctness confirmed via **sims-scaling** (512 vs 64 = 0.890/+363 Elo);
   code review clean. Plateau was a recipe ceiling, not a bug → **raised self-play sims 64→256**
   (distills the net's search strength into the policy). This broke the plateau.
6. **Run + deaths.** Instances died repeatedly (24-h cap, then more frequently); each recovery
   was lossless via S3. Both seeds trained to their finals; the run ended when instances were
   not replaced.

---

## 7. Measured results (all real runs; `batched_match.py`, 120 games @ 128 sims, noise 0)

| Measurement | Result | Meaning |
|---|---|---|
| 512 vs 64 sims (same net) | 0.890 / **+363 Elo** | MCTS engine correct (deep search wins) |
| c2 step 735350 vs early 272900 | 0.983 / **+708 Elo** | genuine large improvement |
| c1 step 218k vs frozen 165k | 0.592 / **+64 Elo** | sims-256 broke the plateau |
| c1 step 140k vs 100700 (pre-sims-bump) | 0.492 / −6 | flat before the fix |
| MCTS/policy vs uniform-random | ~1.00 | dominates random |
| Harness smoke (random vs reference, 6g) | score 0.0, all terminal | referee + scoring correct |

Value loss trajectory: 0.000 (collapsed) → 0.02 → 0.13 → ~0.2–0.7 (healthy honest loss on
well-played diverse positions). Final durable steps: **c1 712200, c2 977400**.

---

## 8. Submission mapping (see `SUBMISSION_FORM.md` for exact text)

| Form field | Value / file |
|---|---|
| Industry Domain | AI — Game-Playing / Reinforcement Learning |
| Software & Version | Python 3.12, PyTorch 2.6, pybind11 (C++17) |
| Operating System | Linux |
| Licensing | Free / Open Source |
| Task Description | full Objective/Starting-State/Rules in SUBMISSION_FORM.md |
| Task Short Name | `cygo_superhuman_agent_001` |
| Input Materials | upload `dist/cygo_input.tgz` |
| Reference Output & Eval Deps | upload `dist/cygo_reference.tgz` (or Drive link) |
| How to verify success | the 1000-game grading paragraph |
| Difficulty self-test | Opus 4.8 / Claude Code / <1%; basis = curation estimate |
| Evidence | `dist/verification_evidence.pdf` |
| Confirmations | two checkboxes |

---

## 9. Reproduce / resume runbook

**Read a checkpoint from the laptop (nodes down):**
```bash
creds=$(AWS_PROFILE=greenland aws sts assume-role \
  --role-arn arn:aws:iam::703671891219:role/Intern --role-session-name pull \
  --region us-east-2 --query Credentials --output json)
export AWS_ACCESS_KEY_ID=... SECRET ... SESSION_TOKEN ...   # parse from $creds
aws s3 cp s3://greenland-intern-artifacts-703671891219-us-east-2-an/\
cygo_resume/cygo-run2/global_step_977400/ckpt.pt ./weights_src.pt --region us-east-2
```

**Re-stage the bundle from a checkpoint:**
```bash
cd tasks/games/cygo_superhuman_agent_001
python3 scripts/stage_task_data.py --out ./task_data --reference-weights <ckpt.pt>
tar czf dist/cygo_reference.tgz -C task_data reference
tar czf dist/cygo_input.tgz     -C task_data input
```

**Redeploy a cluster on a fresh instance (if training resumes):**
```bash
# 1) tunnel:  SSM_TARGET=<mi-...> bash greenland/greenland-connect.sh tunnel     (c1, port 1077)
#    (c2: SSM_TARGET2=<mi-...> LOCAL_PORT2=1078 PROFILE2=greenland2 REGION2=us-west-2 bash greenland/greenland-connect-2.sh)
# 2) deploy (auto-resumes latest from S3):
LOCAL_PORT=1077 CYGO_BOXTAG=cygo-run1 MAIN_IP=<main ip> WORKERS="<w1..w5>" \
  LGPUS=4 CH=256 BL=15 GAMES=64 SIMS=256 bash greenland/build_cygo/deploy_cluster.sh
```

**Regenerate the evidence PDF:** edit `dist/verification_evidence.html`, then
`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu
--no-pdf-header-footer --print-to-pdf=dist/verification_evidence.pdf file://$PWD/dist/verification_evidence.html`.

---

## 10. Honesty ledger

- **Measured (real):** engine correctness (sims-scaling +363), reference improvement
  (c2 +708 over early self, c1 +64 over its sims-256 window), ~100% vs random, harness
  end-to-end. Two independent seeds trained + durable across many instance deaths.
- **Estimated, NOT measured:** the <1% / 0.06–0.12 difficulty is a **design-time adversarial
  curation estimate**. No frontier model has been graded against the reference, and a
  from-scratch reachability run under the agent budget was not completed. This is stated in
  `reference_metrics.json`, `task_card.json`, and the evidence PDF.
- **Not committed to git:** trained weights (`*.pt`, ship via Drive), upload tarballs (`*.tgz`),
  and `greenland/` (creds/instance IDs).
```

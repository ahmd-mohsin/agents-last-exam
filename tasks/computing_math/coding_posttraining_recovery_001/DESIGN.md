# coding_posttraining_recovery_001 — design + validation plan

Status: design only. No executed run, expert score, GPU runtime, or frontier
result is claimed. Numeric sizes below are proposed pilot settings to be
measured, not commitments.

## Premise

A team has a compact coding model and a post-training pipeline whose visible
reward improves while independently tested code correctness (or previously
available task coverage) regresses. Recover the experiment, train a replacement
adapter under a single-GPU budget, and deliver an executable release another
engineer can reproduce. The valuable outcome is a trustworthy post-training
release, not a diagnosis document.

## Interface / product contract (the lesson from the wireless combiner incident)

- Agent trains an adapter however it wants (weighted SFT, DPO, corrected GRPO,
  replay, early stopping, interpolation, hybrid — all eligible unless a real
  deployment constraint rules one out).
- Evaluator owns tokenizer, prompt template, decoding, problem order, test
  execution, metrics. It loads ONLY the submitted adapter tensors + validated
  adapter config into its pinned base model. Agent training code is NOT imported
  for final inference. Self-reported metrics get no credit. Inference wrapper /
  external call / answer lookup / evaluator-dependent behavior is out of contract.

## Starting materials given to the solver

- Fixed base: pinned Qwen2.5-Coder-1.5B-Instruct (Apache-2.0, ~1.54B params,
  ~3.08 GB BF16 weights), tokenizer, full local weights.
- Start adapter + one naturally regressed post-training adapter, with checkpoint
  metadata. Every supplied checkpoint is a candidate rollback solution.
- Training repo: the actual compact trainer, reward impl, preprocessing, config,
  representative logs from the failed workflow (installs at task start).
- Training archive: prompt IDs, root-family IDs, completion tokens, validity
  labels, reward components, rollout checkpoint IDs, termination reasons,
  behavior-policy log-probs if off-policy.
- Replay material: verified training solutions incl. cases lost by the regressed
  run.
- Visible validation: root-family-disjoint problems, independently checked
  execution outcomes, examples of each declared cohort.
- release_contract.json: fixed inference settings, adapter compatibility,
  quality/coverage thresholds (set after pilot), allowed compute, public smoke
  evaluator.

Workload (pilot, not commitments): short Python function-repair problems with
executable contracts. ~1000 train problems, few-thousand archived completions,
128 visible validation, sealed pool candidate 384 independent root families.
NOT full SWE-bench repo rollouts on a 1.5B model initially.

## Long-horizon decision surface (keep the REAL incident's faults, do not plant six)

1. What regressed: recompute independent correctness + coverage for supplied
   checkpoints; reconcile reward logs vs executable results.
2. Trustworthy training examples: trace identity through generation, verification,
   dedup, split construction.
3. Update semantics: completion masks, termination handling, advantage
   construction, behavior-policy provenance, accumulation normalization.
4. Recovery intervention: compare a few hypotheses with short runs; choose data,
   objective, replay, stopping jointly.
5. Validate + select checkpoint: separate problem families, fixed decoding.
6. Reproducible release: export adapter, reload in canonical runner, bounded
   train/restart replay.
Every hidden correctness expectation must follow from a public math/data/output
contract.

## One-GPU resource contract (target NVIDIA L4, 24 GB)

- One physical GPU exposed; log visible device count/identity at startup + during
  reference run.
- Whole process tree fits 24 GB (measure driver-visible AND framework alloc).
- BF16 base + bounded-rank LoRA; short contexts + accumulation; alternate
  generation/optimization on the same GPU (no hidden 2nd-GPU rollout / reward
  server).
- Pre-stage fixed archive + weights (no large downloads / external teacher).
- Pilot 8 vCPU, 32 GB RAM. Target 4h agent + <=1h grading within a 5h envelope.
- Reference recipe target 45-75 min (design target, profile it). Eval budget
  example: 384 x 8 x 192 tok <= 589,824 completion tokens -> ~218 tok/s to finish
  in 45 min; measure on the L4 and shorten if not met. Do NOT cut rigor to fit an
  optimistic runtime.
- NOTE: our box is 8xA100; must emulate the L4 profile (single visible GPU +
  24 GB cap) and DISCLOSE. ALE stock gpu-free profile maps to Windows; the Linux
  GPU image needs explicit provisioning.

## Metrics (measure the product)

Per hidden problem i, 8 completions under pinned protocol; c_i pass all tests.
  P1 = mean_i c_i/8   (sampled pass@1, NOT greedy)
  P8 = mean_i 1[c_i>0]  (coverage in 8 draws; use combinatorial estimator if
       >8 samples)
Report greedy separately if the product needs it. Do NOT call pass@8
"reasoning-mode preservation" without an independent mode definition.
Cohorts: routine repairs, longer valid repairs, changed prompt/trace source,
protected capability slice. Split at root-family / lineage / transformation-family
level (renaming identifiers != new family).
Baseline: frozen start checkpoint chosen before sealed eval; test every supplied
checkpoint as rollback. Quality bar must exceed best unchanged checkpoint, else
rollback legitimately solves it.

## Rubric (finalize numbers AFTER pilot, in public release_contract.json)

Option: fraction of predeclared cohorts meeting release reqs:
  I_g = 1[P1_g >= T1_g AND P8_g >= T8_g],  S = mean_g I_g.
Return all raw metrics + uncertainty + resource use + full-release flag. Global
validity failure (unloadable adapter, resource/API contract violation) invalidates.
Formatting must not zero a working model. Freeze thresholds from a real release
requirement + repeated expert perf with margin; do NOT set the line just past one
model's score. Paired comparisons; resample independent root families for
uncertainty (8 completions from 1 problem are not 8 independent families).

## Evaluator isolation

Generated Python is untrusted. Gold outputs/oracles in a separate trusted
controller. Candidate functions run in a restricted container/process with only
permitted inputs; compare serialized results outside. Bound runtime/output/mem/
process creation. Prefer newly curated releasable families with verified oracles;
a repackaged public HumanEval/MBPP split is NOT uncontaminated new content
(EvalPlus-style stronger tests are fine as technique).

## Deliverables (agent)

output/adapter/adapter_model.safetensors (+config), output/train.py,
output/config.yaml, output/reproduce.sh, output/provenance.json,
output/diagnosis.md, output/metrics.json. Private reference (not staged): genuine
successful adapter, exact recovery command, env lock + image digest, logs,
measured resource use, raw expert eval, sealed tests/oracles, negative controls.
Grading = bounded train/restart replay (validates trainer executes real updates +
restores state within calibrated tolerance); full fresh reproduction in author
validation.

## Feasibility order (before packaging)

1. Select a REAL incident with an understood recovery from releasable research.
   If none exists, label as new research + budget for uncertainty.
2. Reproduce failure+recovery on one L4 (small model can change the phenomenon).
3. Build smallest truthful evaluator (independent execution + quality/coverage/
   mem/runtime; valid + invalid reference cases).
4. Challenge with strong cheap baselines: rollback to every checkpoint, early
   stop, filtered SFT, DPO, default/current GRPO, conservative replay, checkpoint
   interpolation. Enough tuning to avoid strawmen.
5. Expert recipe on 3 seeds (stability screen) + second-expert review.
6. Run Astra (GPT-6-astra) FIRST in a full autonomous session (same tools/data/
   hw/time/feedback). If it gets a valid release simply, keep as positive control,
   do NOT exclude by moving thresholds.
7. Screen representative frontier panel; record versions/harness/seeds/logs; mark
   infra failures + incomplete runs separately.
8. Freeze candidate; fresh sealed families for final confirmation.
9. Package only a surviving candidate on the existing ALE lifecycle scaffolding.

## Honesty guardrails (carried from the wireless audit)

- "Pending" runs never enter a difficulty average.
- Distinguish autonomous sessions vs single-shot programs vs infra failures.
- Do not dilute a strong model's success with older failed models.
- Do not claim "frontier-proof"; ALE uses an average across a representative panel
  vs thresholds (near-term < 0.5, last-exam < 0.1). A few failed trials bound, not
  measure, a rare success probability.
- Reference improvement can raise the frontier; it cannot invalidate an already
  admissible frontier-model result.

## Alternatives if the incident does not survive the pilot

- preference_tradeoff_recovery_001: recover a multi-objective preference-training
  release whose aggregate reward hides a collapsed objective.
- kv_cache_eviction_release_001: repair/release a bounded-memory KV-cache policy
  preserving retrieval/code quality under measured cache+latency limits.
Both unmeasured; criteria must come from real product/research workflows.

## Measured pilot results (executed 2026-09-11, greenland 8xA100, single-GPU pinned + 24 GB cap)

Feasibility steps 2-3 executed. Base = Qwen2.5-Coder-1.5B-Instruct, LoRA r16 on
q/k/v/o, GRPO group=8, bf16, gradient checkpointing. Metrics on test.jsonl (144
problems incl. held-out families dedup_order + wrong_init), k=8, temp 0.8, 192 tok.

| stage | pass@1 | pass@8 | weak_proxy |
|---|---|---|---|
| base | 0.472 | 0.799 | 0.891 |
| faulted (weak reward, 160 steps) | 0.351 | 0.486 | 0.978 |
| recovery (true reward, 200 steps) | 0.642 | 0.951 | 0.932 |

Recovery beats base by +0.170 pass@1 (+36% rel) and improves every family incl.
held-out (dedup_order 0.57, wrong_init 0.62). weak_proxy stays healthy (no hacking).

The FAULT (in sandbox.py weak_reward): a length-biased execution proxy. It rewards
"runs on the test inputs without raising" scaled by a brevity factor, so a short
running stub (`return 0`) strictly out-scores any longer correct repair. GRPO
collapses the policy onto short non-solutions: completion_length 45 -> 22, weak
proxy -> 0.98, true pass@1/pass@8 regress. A plain "fraction-of-inputs-that-run"
proxy did NOT reproduce a regression (correct code and stubs both score ~1.0 ->
reward_std -> 0 -> dead gradient; measured weak-GRPO even improved pass@1 to 0.53).
The anti-correlation from the brevity bias is what creates a real hack.

Resource envelope (measured, NOT QLoRA — plain LoRA): train peak 8.5 GB reserved /
23.9 min for 200 steps; eval peak 5.4 GB / 5.1 min for 144x8. Fits a 24 GB L4 with
large headroom. Earlier OOMs were collisions with a co-tenant on the shared box,
not a real L4 limit; the prior "needs QLoRA/4-bit" note is retracted.

### Honest difficulty read (open item)

The recovery is "diagnose the length-biased proxy, switch to a held-out test-pass
reward, retrain ~200 GRPO steps." With box access + the faulty config + logs, a
frontier agent can plausibly find this, so as built this is a NEAR-TERM task
(frontier avg likely < 0.5, NOT < 0.1). Not yet a last-exam task. Options to raise
difficulty before the frontier screen are tracked separately. Frontier screen
(Astra-first) not yet run at this task.

## Relationship to the wireless task

Keep wireless as a separate honest full-spectrum contribution if it meets review.
Do NOT carry over its premise or claim its difficulty evidence transfers.

# Post-training recovery: reward-hacked coding model

You inherit a FAILED reinforcement post-training run. The team was improving a
compact coding model (Qwen2.5-Coder-1.5B-Instruct) at short Python function
repair with GRPO + LoRA on a single GPU (NVIDIA L4, 24 GB). The dashboards looked
good: the reward climbed steadily and the team's own validation improved. But an
independent test-pass evaluation shows the shipped adapter is WORSE than the
starting model, and a capability the model used to have has degraded.

Your job is to ship a trustworthy replacement adapter that clears the release
contract, not to write an incident report. A diagnosis with no releasable adapter
scores zero.

## What you are given (`input/`, read-only)

- `repo/` — the actual training code the run used: `grpo_train.py` (GRPO+LoRA
  trainer), `sandbox.py` (reward + isolated executor), `problems.py` (data
  generator), `config.yaml` (the run's hyperparameters), and
  `logs/` (representative stdout + reward/length trajectories from the failed run).
- `adapters/` — `regressed/` (the shipped adapter) plus every intermediate
  `checkpoint-*`. Each checkpoint is a candidate rollback.
- `data/` — `train.jsonl` and the team's `val.jsonl` (their "held-out"
  validation). Treat provenance skeptically.
- `replay/verified_solutions.jsonl` — solutions the team marked as verified for
  replay / filtered-SFT. Treat provenance skeptically.
- `base_model/` — full local weights + tokenizer of the pinned base.
- `smoke/` — `smoke_eval.py`, a PUBLIC subset grader you can run yourself to sanity
  check an adapter (a small, non-sealed sample; passing it is necessary, not
  sufficient).
- `release_contract.json` — the fixed inference protocol, the predeclared cohorts,
  the per-cohort thresholds, and the compute budget you must respect.

## The release contract (summary; `release_contract.json` is authoritative)

- Fixed decoding: k=8 samples, temperature 0.8, top_p 0.95, 192 new tokens, seed
  1234. Metric per problem: pass@1 = mean fraction of the 8 samples that pass ALL
  hidden tests; pass@8 = fraction of problems solved by at least one of 8.
- Cohorts scored on a SEALED, family-disjoint pool: `seen_family_newvariant`,
  `held_out_family`, `long_repair`, `protected_synthesis`.
- Coverage score S = fraction of cohorts meeting BOTH their pass@1 and pass@8
  thresholds. A full release requires S = 1.0.
- Single GPU, 24 GB. Training within 90 minutes wall. The evaluator's own
  inference must fit 24 GB.

## Deliverable (`output/`)

- `adapter/` — a PEFT LoRA adapter (`adapter_model.safetensors` +
  `adapter_config.json`) loadable onto the pinned base. The evaluator loads ONLY
  these tensors into its own copy of the base; your training code is NOT imported
  at inference.
- `train.py`, `config.yaml`, `reproduce.sh` — a bounded, deterministic recipe that
  regenerates your adapter on the single-GPU budget.
- `diagnosis.md` — what regressed and why, with evidence.
- `metrics.json`, `provenance.json` — your measured numbers and data lineage.

## Rules

- `input/` is read-only. Do not modify it.
- No inference-time wrapper, no network calls, no answer lookup, no behavior that
  depends on the evaluator or its paths. The adapter must stand on its own.
- Everything is recomputed from a fresh run of your adapter. Self-reported numbers
  get no credit.

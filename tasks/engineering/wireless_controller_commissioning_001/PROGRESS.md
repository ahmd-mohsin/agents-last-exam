# Progress — wireless_controller_commissioning_001

Status as of 2026-09-10. Honest running record: what was built, what the
rigorous checks found, verified numbers, and the open decision.

## The task (premise)

Commission an energy-efficient multi-user MIMO uplink power controller from
imperfect, delayed channel measurements (3GPP TR 38.901 UMi). The agent predicts
the delayed channel, calibrates joint uncertainty over the declared operating
envelope, and returns transmit powers. An evaluator replays the controller on
withheld channels from a shifted deployment regime and scores per-user
reliability and transmit energy. Long-horizon: audit -> predict under delay ->
joint uncertainty -> robust power allocation -> reproducible release.

## What is built (all validated on an 8xA100 box)

- Physics: full 3GPP TR 38.901 UMi geometry-based channel model
  (greenland/build/sim/gbsm_38901.py), validated (delay/angular spreads vs
  targets, Doppler-driven decorrelation, spatial correlation).
- External software: Sionna 2.1.0 in an isolated venv; generates the coded-link
  BLER curve (LDPC + QAM) that defines the per-tone decode threshold and
  validates the SINR abstraction.
- Pipeline: predictor (Wiener LMMSE), scenario-robust energy-optimal power
  allocator, scoring primitives, archive generator (Zarr), reference generator.
- ALE package (tasks/engineering/wireless_controller_commissioning_001/):
  task_card.json, main.py lifecycle, scripts/{verify_submission.py (self-contained,
  deterministic, isolation boundary), reference_controller.py, second_expert.py
  (independent method), stage_task_data.py, PACKAGING.md}.
- Task data staged (input/ + plain reference/ + encrypted reference.7z) and
  uploaded to s3://greenland-intern-artifacts-703671891219-us-east-2-an/wcc
  (188 objects). task_data_source wiring documented in PACKAGING.md.

## Key design correction (integrity fix)

Original interface let the agent supply the channel prediction used to build the
receiver combiner. Opus 5 (at 128k tokens) exploited this to co-design a
power-aware combiner, meeting reliability at trivial energy — i.e. it solved the
task through the interface, not the intended problem. Fixed per the ALE report:
the combiner is now EVALUATOR-CONTROLLED (MMSE on the last observed, stale frame);
the agent returns POWERS ONLY. Exploit closed.

## Difficulty findings (the hard truth)

- U=4 users, M=8 antennas (original): too easy under the fixed combiner. Multiple
  frontier models solve it (gpt-5.6-sol, sonnet-5, gpt-6-astra all coverage 1.0).
- U=5 interference-limited, M=8 (redesign): materially harder. The naive proxies
  fail decisively and most frontier models fail:
    reference expert   coverage 1.0 (0.98% fail, energy 204)   feasible w/ margin
    ce (no uncertainty)          0.0  (16% fail)
    greedy (no interference)     0.0  (90% fail)
    persistence / flat           0.0  (energy / 17% fail)
    gpt-5.6-sol (fair U=5)       0.0  (3.0% fail)
    gpt-5.6-terra                0.0  (5.7% fail)
    gpt-5.6-luna                 0.0  (code crash)
    sonnet-5                     0.0  (2.44% fail)
    opus-4-8                     0.0  (18.8% fail)
    opus-4-7                     0.0  (24.6% fail)
    opus-4-6                     0.0  (9.2% fail)
    sonnet-4-6 / haiku-4-5       0.0  (code crash)
    opus-5                       pending (needs 128k; expected fail)
    GPT-6-astra (fair U=5)       1.0  (0.68% fail, energy 16.3)  <-- ONLY MODEL THAT SOLVES IT
  => 10 of 11 screened frontier/Claude models FAIL; only GPT-6-astra passes.
- GPT-6-astra out-optimizes the reference expert in every regime (17x less energy
  at U=4, 12x at U=5). So no budget is both feasible for the expert and failing
  for GPT-6-astra. Making it harder tips into infeasible-for-everyone.

## Honest classification status

- By ALE's stated classification models (GPT-5.6-Sol, Claude Opus 5): GPT-5.6-Sol
  fails U=5; Opus 5 very likely fails. That pair would average ~0 -> would be
  classified last-exam.
- But GPT-6-astra, a mainstream frontier model, solves U=5 at 1.0. In spirit the
  task is NOT frontier-proof; claiming last-exam would be gaming the specific
  classifier, not the intent.
- Realistic tier: full-spectrum / near-term. It genuinely defeats the GPT-5.6
  family and Sonnet 5; frontier average ~0.15-0.2 (< 0.5 bar), not < 0.1.

## Open items (running / pending)

- Older Claude tier (opus-4.6/4.7/4.8, sonnet-4.6, fable-5.1, haiku-4.5): U=5
  fair re-screen generating; expected to fail (weaker than sonnet-5).
- Opus 5 at U=5: needs a 128k call to emit; expected to fail (exploit closed).
- Reference expert is over-conservative (energy 204 vs GPT-6-astra 16.3); a
  genuinely optimal expert would tighten the budget but still not fail GPT-6-astra.

## Decision needed

1. Submit as full-spectrum / near-term with the honest verified numbers (defeats
   GPT-5.6 family + Sonnet 5; GPT-6-astra passes). Accepted-tier, honest.
2. Keep hunting for a frontier-proof regime (evidence says this problem class
   won't get there vs GPT-6-astra without becoming infeasible).
3. Pivot the task premise to something GPT-6-astra can't brute-optimize (larger
   redesign, uncertain).

Recommendation: option 1 (honest full-spectrum), do not claim last-exam while
GPT-6-astra scores 1.0.

## Artifacts

- Difficulty record: greenland/build/difficulty_screen.json
- Failure dashboard: greenland/build/out/wcc_failure_dashboard.png
- Submission-form draft (needs tier corrected to full-spectrum): tmp.md
- Task data: s3://greenland-intern-artifacts-703671891219-us-east-2-an/wcc

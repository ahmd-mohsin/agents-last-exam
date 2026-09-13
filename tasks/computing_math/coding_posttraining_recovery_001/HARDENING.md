# Hardening plan: near-term -> last-exam

Goal: make the recovery require solving SEVERAL entangled, individually-necessary
faults so that no single discoverable fix (and no cheap baseline) clears the
release rubric. Frontier target: avg < 0.1. Every planted fault must be a real,
co-occurring post-training failure mode, not a contrived puzzle.

## Why the current version is only near-term

Measured: base pass@1 0.472 -> faulted 0.351 -> recovery (true reward, 200 steps)
0.642. The recovery is one move: replace the length-biased proxy with a held-out
test-pass reward and retrain. A frontier agent with the box + logs finds it. So a
single "true-reward GRPO" run (the current adapters/true) already clears any
single-metric bar. We must make that exact run FAIL the hardened rubric.

## Entangled faults (all real, all co-present in the shipped failed run)

A. Reward misspecification (DONE): length-biased execution proxy in sandbox.py.
   Naive fix: switch to true test-pass reward. Necessary but NOT sufficient.

B. Data-provenance corruption (NEW):
   - Split leakage: some "train" problems share root families with the sealed
     eval pool, so the agent's OWN measured validation is inflated; the
     family-disjoint SEALED pool does not move unless they detect the leak and
     rebuild a clean split.
   - Mislabeled replay: a fraction of the supplied "verified" replay solutions are
     wrong (pass the weak/public check, fail the hidden oracle). Training true
     reward on them injects bad gradients. The agent must RE-VERIFY replay against
     oracles before use.

C. Trainer update-semantics bug (NEW): a subtle, realistic defect in the SHIPPED
   trainer that caps recovery even with a correct reward. Candidate: completion
   loss mask also covers the prompt tokens (dilutes signal) OR advantage
   normalized across the whole batch instead of per-prompt group (wrong baseline).
   Private reference keeps the correct trainer; agent gets the buggy one and must
   read + fix it. Reward-swap on the buggy trainer underperforms the bar.

D. Protected-capability cohort + catastrophic forgetting (NEW): a predeclared
   cohort the model must NOT regress on (e.g., from-docstring code synthesis, a
   capability orthogonal to short-function repair). Plain narrow true-GRPO on the
   repair distribution degrades it below threshold. Only a recovery with replay /
   KL anchoring / data mixing preserves it while improving repair.

## Rubric that forces all four (release_contract.json)

Option-style: S = mean_g 1[P1_g >= T1_g AND P8_g >= T8_g] over cohorts
{routine_repair, long_repair, held_out_family (SEALED), protected_synthesis}.
Release requires S == 1.0 (every cohort clears, with margin), measured on the
SEALED family-disjoint pool. Each naive baseline fails a DIFFERENT cohort:
  - rollback to any checkpoint      -> routine_repair below T (no gain)
  - reward-swap only, buggy trainer -> repair gains capped below T (fault C)
  - reward-swap, clean trainer, dirty data -> sealed held_out_family below T (B)
  - plain true-GRPO (our 0.642 run) -> protected_synthesis below T (D)
  - filtered SFT                    -> misses margin vs GRPO
Only fix(A)+cleandata(B)+fixtrainer(C)+preserve(D) clears all cohorts.

## Build + measure order (prove each claim)

1. Protected cohort (D): build an executable from-docstring synthesis probe with
   verified oracles; measure base vs adapters/true (our 0.642 run). If plain
   true-GRPO degrades it, D is proven with existing artifacts. If not, induce
   sensitivity (higher LR / longer narrow training) or pick a more sensitive slice.
2. Trainer bug (C): plant the mask/advantage defect in the shipped trainer; show
   reward-swap on it underperforms the correct trainer by a margin.
3. Data corruption (B): add leakage + mislabeled replay in problems.py; show
   sealed-pool < local-val until cleaned; quantify the bad-gradient hit.
4. Set release_contract thresholds from the CLEAN full recovery + margin, verify
   every baseline fails >=1 cohort, and only the full recovery clears S==1.0.
5. Then frontier screen (Astra-first).

Each step is a measured GPU run on the emulated L4 (single visible GPU + 24 GB
cap); disclose the 8xA100 emulation. Keep faults honest and cite the failure mode.

## Measured outcome of the hardening attempt (2026-09-11, p4d box, all GPUs free)

Incident + recovery reproduced deterministically on a fresh instance:
base pass@1 0.459 / pass@8 0.792; faulted (weak reward, 200 steps) 0.351 / 0.493
(weak_proxy 0.978); recovery (true reward) ckpt120 0.623 / 0.951, held-out
families dedup_order 0.35->0.56, wrong_init 0.49->0.64.

The two load-bearing hardening faults DID NOT survive measurement (honest negatives):

- Fault D (catastrophic forgetting): FALSE here. Protected from-docstring synthesis
  is unchanged by the recovery: base pass@1 0.799 / pass@8 1.000 vs recovery 0.797 /
  1.000. LoRA (r16, q/k/v/o, 200 steps) is too gentle and repair/synthesis share
  too much structure to induce forgetting at this scale. The protected cohort does
  not discriminate, so it cannot be the lever that fails a naive recovery.

- Fault B (data provenance): does not defeat a competent recovery. Plain true-reward
  GRPO already generalizes to the SEALED held-out families, and a reward-only
  recovery never consumes the replay set, so mislabeled replay is inert. Leakage
  makes the team's own validation untrustworthy (a real diagnosis nuisance) but does
  not stop the sealed pool from clearing.

- Fault C (trainer update-semantics bug) is the only remaining lever that forces
  work beyond a reward swap, but it is a bounded "read the ~108-line trainer, find
  the completion-mask/advantage defect" step, not obviously last-exam-hard.

Honest conclusion: the recovery is a discoverable reward-swap + retrain that
generalizes; planted faults that do not empirically create a gap must NOT be
claimed. This task is rigorous and long-horizon but NEAR-TERM in difficulty. Do not
assert frontier avg < 0.1; measure it with the frontier screen. Options carried to
the user: (1) run the frontier screen to get a real number, (2) accept as a strong
near-term task toward the 3-task data-track path, (3) plant Fault C anyway and
measure whether it moves the frontier average.

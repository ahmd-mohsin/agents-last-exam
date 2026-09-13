# Agents' Last Exam — Task Submission Proposal

Task: Commission an energy-efficient multi-user wireless controller from imperfect channel measurements.

This document is written to fill the submission form field-by-field. It is a proposal and construction plan. Every numeric threshold marked "candidate target" is a feasibility hypothesis; the final values are frozen only after a real expert reference has been executed on the evaluation environment and the difficulty screen has been run. No reference output in this document is fabricated: the reference package described here must be produced by an actual run of the workflow before submission, per ALE's quality-check rule.

---

## Industry Domain

Telecommunications / Wireless Systems Engineering (specifically: multi-user MIMO uplink power control under imperfect and delayed channel state information).

---

## Software & Version

- Python 3.12, PyTorch 2.6 (CUDA), NumPy 1.26, SciPy 1.14, Zarr 2.18. All builds pinned in the supplied runtime manifest after validation on the target evaluation GPU (do not ship a floating "latest" environment).
- A self-contained MIMO uplink channel simulator (spatial correlation + frequency selectivity + temporal Doppler evolution + pilots + noise), implemented in NumPy/PyTorch so it is fully reproducible on the evaluation image with no heavyweight external dependency. The physical-layer abstraction is published in `radio_system.json` and validated on a calibration set.
- The agent-facing controller may use classical estimation/optimization (e.g. an LP solver), neural models, or hybrids. No specific architecture is required.

Note: Sionna 2.1.0 was evaluated as the channel engine but hard-requires torch>=2.9.1 and pulls ray-tracing dependencies (mitsuba, drjit), which conflicts with the CUDA-matched torch build on the evaluation substrate. It is retained only as an optional offline realism cross-check for the abstraction, not as a task or evaluation dependency.

---

## Operating System

Linux, with an explicitly validated single-GPU image and provider configuration.

Important environment caveat: the stock configuration maps GPU profiles to Windows and the standard Linux profile to CPU, so a Linux CUDA workflow requires a validated image/provider pairing, not merely selecting "Linux" on the form. A CPU replay edition can be supplied as a fallback if, and only if, it preserves the substantive task (measurement audit, prediction, calibration, constrained control, release). If shrinking to CPU removes those decisions, the CPU edition is treated as a different task and gets its own validation.

---

## Software Licensing

Free / Open Source for the task implementation and the simulation framework. All redistributed dependencies and data ship with their applicable licenses included in the input package. Any dataset or model that cannot be redistributed is resolved (replaced or regenerated) before it becomes a required input.

---

## Task Short Name

`wireless_controller_commissioning_001`

---

## Task Description

You are the wireless systems engineer responsible for commissioning a replacement power controller after a change in channel and receiver conditions. The existing production controller still runs, but it behaves conservatively after the propagation and receiver environment shifted, wasting transmission energy.

The supplied project contains a conservative baseline controller, historical pilot and channel measurements, controlled calibration sessions, a complete radio-system specification, and a development replay environment. The system serves four uplink users through an eight-antenna base-station receiver over 24 active frequency bins, one transmitted stream per user. Measurements may be delayed or partially missing, exactly as declared in the input manifests; the latest available channel state information is delayed by the amount stated in the case manifest.

Build and deliver a controller that reduces transmission energy while satisfying the published per-user reliability, transmitter-power, and inference-runtime requirements. Use only information available at each decision time (no future samples). At each decision the controller receives eight historical measurement frames, pilot masks, observation timestamps, noise information, and a per-user service request, and returns transmit powers with shape [batch, user, frequency]. Any additional receiver parameters must be explicitly permitted by the interface. For this task the modulation/coding and receiver combining rules are evaluator-controlled, so the action space is a clean power-control problem.

You may change the estimator, uncertainty model, optimizer, and implementation within the published interface. Validate your decisions on the supplied development cases and produce a portable release package containing the controller, its parameters, the frozen preprocessing/calibration state, a release manifest, a reproduction command, and development results. The package must load and execute in the supplied environment within the declared resource budget and without access to external services.

Write all final artifacts under the designated output directory. Treat the supplied input materials as read-only. Your solution will be replayed by an independent evaluator on withheld cases drawn from the operating conditions described in the specification. Your own predicted channel and your own predicted success probability are not treated as ground truth; delivered service is measured against the held-out true channel.

The workflow is deliberately long-horizon: measurement audit, channel prediction under delay, error calibration, constrained power control, implementation profiling, and release selection are dependent stages. A mistake early (e.g. wrong pilot normalization or leakage across the validation boundary) propagates and surfaces only at independent acceptance.

Required starting package (Input Materials):
- `task_brief.md` — objective, allowed operations, compute budget, required handoff.
- `radio_system.json` — array sizes, sampling rules, receiver definition, power limits, per-user demands, and units (per-subcarrier vs band power, complex-noise convention, antenna ordering, pilot normalization, observation-to-action delay, combiner rule).
- `observations.zarr` — historical complex measurements, pilots, timestamps, masks, permitted metadata.
- `calibration_sessions/` — controlled measurements with reference pilots and calibration metadata.
- `split_manifest.json` — session-level train/development partitions (session-separated, no adjacent-window leakage).
- `starter_project/` — the conservative baseline controller plus the production inference interface.
- `baseline_development.json` — evaluator-produced development metrics for the baseline.
- `public_eval/` — development-only replay, numerical checks, and profiler.
- `output_contract.json` — schemas, tolerances, metric definitions, public acceptance conditions.
- `runtime/` — locked dependency files, offline install assets, docs.

Required final deliverables (agent output under the output directory):
- `controller/` — package implementing the published initialization and inference interface.
- `model.safetensors` — portable parameters if learned weights are used (a classical solution may use its declared equivalent).
- `preprocessing.json` — the actual frozen normalization and calibration state.
- `release_manifest.json` — input hashes, software versions, model/config hashes, interface version.
- `reproduce.sh` — rebuilds the release from visible inputs within the declared budget.
- `development_results.json` — development metrics the evaluator can independently reproduce.
- `technical_note.md` — assumptions, chosen solution, failure analysis, limitations.

---

## Files

### Input Materials

Upload the measurement archive (`observations.zarr`), calibration sessions, `radio_system.json` specification, baseline `starter_project/`, `split_manifest.json`, `public_eval/` tools, `output_contract.json`, and the locked `runtime/`. Include an asset index (`task_brief.md`) explaining each file.

Sizing plan: a reasonable pilot archive is 50,000 to 200,000 decision contexts across many independent sessions, with a smaller session-separated development set. Start smaller during authoring and grow only if the substantive problem needs it. Target a few GB input package; storage and training time are measured after the representation is chosen. Hidden acceptance uses independent session/trajectory seeds (never adjacent windows split from one recording).

### Reference Output & Evaluation Dependencies

Upload the actual expert-executed controller and its parameters, the reproduction scripts, execution logs, source revision, seeds, environment manifest, trusted acceptance results, and evaluator-owned withheld cases. Keep hidden reference channels and acceptance packs strictly separate from the agent-visible inputs.

The reference archive contains more than a metrics spreadsheet: it preserves the release artifact and enough evidence to recreate every measurement. It is produced by a genuine end-to-end run before submission (this is the item gated by ALE's quality check; a fabricated reference forfeits authorship).

Large uploads: provide a Google Drive / Dropbox link with open access to the review team if any file exceeds the direct-upload path.

---

## How should we verify success?

Isolation: load the submitted release in a clean evaluation worker. The worker receives only allowed observations and returns control actions. A separate trusted evaluator process holds the ground-truth channels and the scorer, computes per-user service, physical power violations, energy use, and runtime, and never exposes hidden labels to the candidate code. Mounting a hidden directory beside arbitrary candidate Python is not accepted as isolation.

Primary score (deployment coverage). Let A_j be 1 when scenario pack j meets every public reliability, power, energy, and timing requirement, else 0. For J fixed packs the normalized score is S = (1/J) sum_j A_j. Initial design: 12 packs of about 1,024 independent contexts each. Correlated time steps within a session are not counted as independent observations. Full credit when every required pack passes; partial credit reflects genuine usable capability.

Candidate acceptance conditions per pack (feasibility targets, finalized only after the expert reference and difficulty screen; not measured achievements):
- API and numerical validity: correct shapes, finite outputs, valid power values across all documented input shapes including empty/masked patterns.
- Physical constraints: every action respects per-user and total transmitter bounds within a published numerical tolerance.
- Per-user reliability: one-sided upper confidence bound on service-failure probability at most 1% in each required deployment group, with sample size and multiplicity correction fixed in advance.
- Energy: at least 15% less transmission energy than the immutable conservative baseline on the same offered load (the expert must demonstrate this with comfortable margin on development and confirmation data).
- Runtime: initial profiling target p99 at most 10 ms per complete decision on the chosen evaluation GPU, replaced by a measured, application-appropriate bound before the task is frozen. This is an inference-timing requirement, not a claim of sub-millisecond radio control or a 3GPP timing constraint.
- Reproduction: clean replay reloads the package and reproduces its performance within empirical tolerance on the exact target device and locked runtime.

Independent recomputation. All metrics are recomputed from a fresh execution of the submitted artifact against evaluator-owned inputs. Energy is compared against the immutable baseline on the same offered load. Delivered service is decided by the physical-link evaluator from the held-out true channel, the declared receiver, and the submitted powers — not by the candidate's self-reported success. Physically invalid or non-causal behavior (e.g. use of future samples) is rejected.

Diagnostics (explain the score, do not set it). A separate diagnostic record captures preprocessing correctness, prediction error, uncertainty coverage, constraint violations, runtime, and per-user service. Diagnostic weights are fixed before observing any frontier-model performance.

Multiple valid implementations accepted. Do not require weights, source, or latent calibration parameters to match the expert byte-for-byte (channel models have equivalent parameterizations; absolute phase can be non-identifiable). Score observable behavior; enforce only identifiable parameter constraints. Any implementation that meets the contract passes.

Timing protocol. Specify CPU, GPU, driver, precision, batch size, warm-up, synchronization, data-transfer boundary, and competing-workload policy. Include all preprocessing and transfers required by the advertised API. Measure baseline and candidate under identical conditions with paired repeats and the device reserved.

Missing output. Per the ALE authoring guide, absent or unloadable candidate output receives a score of 0 rather than crashing evaluation.

---

## Difficulty Self-Test

Model: Claude Opus 4.8. Harness: Claude Code. Environment: single validated Linux GPU image.

Estimated score: target class is last-exam (average below 0.1). This is a design target, not a measured result. Per ALE's guidance and this task's own reference-quality rule, the estimate is left unclaimed until the frozen task has been run.

Screening protocol before any difficulty claim:
- Run at least three strong model/harness configurations, five independent attempts each, on the frozen task. Record exact provider model IDs, dates, harness versions, tool access, reasoning settings, token budgets, time limits, and environment hashes.
- Report both mean normalized score and full-pass frequency, plus per-scenario and per-stage diagnostics (a zero full-pass rate can coexist with substantial partial capability).
- Use a held-back confirmation set after design iteration so author tuning does not overfit to a few agent attempts.
- Classify every failure trace as domain / planning / implementation / release / environment / budget error. Only domain, planning, and release/integration failures count as substantive difficulty; environment and budget failures are fixed before any difficulty claim.
- Note the statistics: zero successes in five trials gives only about a 45% one-sided 95% upper bound on success probability; zero in thirty gives about 9.5%. The binary bound is not applied directly to a continuous mean score.

Expected failure modes to be confirmed (hypotheses, not results): selecting a predictor on leakage-contaminated validation; treating uncertainty as independent across interferers; lowering energy by underserving one user; exporting a controller that depends on transient experiment state. If agents reliably solve the full workflow, the real result is preserved and the task is reclassified honestly.

---

## Construction plan and integrity commitments

Why this clears the bar:
- Long-horizon: seven dependent engineering stages (audit, prediction, calibration, constrained control, profiling, release selection, independent acceptance). Difficulty comes from carrying physical and statistical requirements through the whole release, not from producing N files.
- Genuine reference: the expert release is executed for real, passes fresh acceptance replay repeatedly, and is reproducible from visible inputs within the agent's budget. A second expert reconstructs the task from the staged package without the hidden reference, to confirm no crucial knowledge lives only in the author's head.
- Objective grading: recomputed from a fresh run against evaluator-owned inputs; hidden channels, baseline denominators, and the scorer stay outside the candidate's control.

24-A100 construction allocation (reserved-slot estimates, reallocated after first measurements; excludes target-device validation and frontier-agent API charges):
- Channel/measurement archive generation: 8 GPUs, about 12 h each, ~96 A100-hours.
- Expert controller and uncertainty experiments: 8 GPUs, about 18 h each, ~144 A100-hours.
- Failure cases and negative-solution evaluation: 4 GPUs, about 12 h each, ~48 A100-hours.
- Memory, runtime, and replay validation: 4 GPUs, about 8 h each, ~32 A100-hours.
- Total initial authoring allocation: ~320 A100-hours.

First expert reference (classical, to establish feasibility before any large model or dataset):
1. Estimate calibration parameters from controlled pilot sessions; recover channel estimates under the declared complex-noise and pilot conventions; verify against small numerical fixtures.
2. Baseline predictors: persistence and a linear state-space / autoregressive model, fit on complete training sessions, selected on distinct development sessions; complex values modeled explicitly.
3. Build an error archive from development residuals, preserving joint errors across users/antennas/frequencies (a scalar NMSE does not capture the interference error that matters for power control).
4. Freeze the combiner as a function of permitted observations and a nominal power; keep it independent of the power variables so scenario SINR constraints are linear.
5. Solve the resulting LP. With fixed combiner g, uncertainty scenario m, frequency f: a_{uvf}^{(m)} = |g_{uf}^H h_{vf}^{(m)}|^2, n_{uf} = g_{uf}^H R_{n,f} g_{uf}, and the SINR requirement becomes a_{uuf}^{(m)} p_{uf} - gamma_{uf} sum_{v != u} a_{uvf}^{(m)} p_{vf} >= gamma_{uf} n_{uf}. Minimizing total power under these plus linear power limits is an LP. With 4 users, 24 frequencies, 16 uncertainty scenarios: 96 power variables, 1,536 scenario SINR inequalities before power bounds. This is a starting formulation, not an out-of-sample outage guarantee; per-frequency thresholds are reconciled with the declared coded-service abstraction.
6. Evaluate against the independent physical-link calculation to decide whether the energy/reliability envelope is feasible before scaling.
7. Profile the complete release; accept a CPU implementation if it already meets the runtime contract.

Diagnostic controls built in: perfect-CSI coefficients isolate information limits; certainty-equivalent coefficients isolate uncertainty handling; unoptimized baseline code isolates runtime problems.

Negative solutions that must each fail for the correct reason: unchanged baseline (physically valid, fails energy), zero power, maximum power, a predictor using future samples, an overconfident predictor, a controller that ignores one user, and a fast implementation with a numerical defect.

Reject-or-redesign triggers: a classical baseline already meets every target; expert success depends on hidden information; service requirements are physically infeasible; or runtime is dominated by an unsupported environment. If it stays too easy, deepen a real requirement (documented deployment shift, uncertainty under that shift, an operational condition the release must satisfy) and re-run the expert after every material change. Do not add arbitrary traps or tighten thresholds merely until models fail.

Milestone gates: (1) scope+environment fits the substrate; (2) physics/data interpretable by a second expert; (3) positive reference with margin and within budget; (4) grader validated with positives, meaningful negatives, and replay isolation; (5) difficulty screened with auditable traces; (6) confirmation on untouched cases and repeated target-device checks; (7) submission with an execution record behind every numeric claim.

The immediate work item is one end-to-end expert reference on a compact wireless case. That run determines whether the coupling of estimation, uncertainty, and control is both feasible and substantial; the archive and agent trials scale only after it.

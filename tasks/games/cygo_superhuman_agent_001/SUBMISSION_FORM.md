# ALE Submission Form — CYGO (Cylinder Gonnect)

Copy-paste values for each form field. Bundles to upload live in `dist/`.

---

## Industry Domain *
```
Artificial Intelligence — Game-Playing / Reinforcement Learning (AlphaZero-style self-play)
```

## Software & Version *
```
Python 3.12, PyTorch 2.6, pybind11 (C++17 engine). Self-contained; no proprietary software.
```

## Operating System *
```
Linux
```

## Software Licensing *
```
Free / Open Source
```

---

## Task Description *

**Objective:** Build and train an agent that plays CYGO (Cylinder Gonnect) — a novel 13×13 Go variant invented for this exam — strongly enough to be competitive with a hidden, AlphaZero-trained reference agent. Nothing pretrained transfers: no engine, weights, opening book, or literature exists for this game. The difficulty is *achieving* the strength (a correct+fast engine, batched self-play, and a well-tuned training run), not knowing the plan (AlphaZero is handed over).

**Starting State (provided in `input/`, read-only):**
- `task_brief.md`, `cygo_rules.json` — objective, protocol, per-move budget, and the full rules: 13×13 board where **columns wrap into a cylinder** (rows do not), Go capture, suicide illegal, **positional superko**, **win by connecting the top edge to the bottom edge**, no passing, ply-cap 600 adjudication (ties → White), and the 52-element symmetry group.
- `engine/` — the exact CYGO rules engine (`cygo_engine.py` + `cygo_cpp.cpp`, pybind11) matching the referee.
- `baseline/` — a random-rollout MCTS smoke opponent.
- `harness/` — the exact match harness + stdin/stdout protocol used to grade (`match_harness.py`, `agent_proto.py`).
- `starter/` — a `run.sh` template + protocol agent stub.

**Instructions & Rules:**
1. Read `input/task_brief.md` and `input/cygo_rules.json`; use the shipped engine for legal moves / terminal detection.
2. Build a training pipeline (self-play + policy/value network + MCTS — AlphaZero-style) and train an agent to reference-competitive strength.
3. Deliver `output/run.sh`: an executable implementing the match protocol over stdin/stdout (`newgame <color> <seed>`, `[opening ...] move <oc> <or>|-` → `= <col> <row>`, `quit`), emitting only legal moves within the per-move wall-clock budget on one 24 GB GPU.
4. Validate locally against the shipped baseline bot and the exact harness in `input/harness/`.
`input/` is read-only; no network at eval; the reference is staged only at evaluation time.

## Task Short Name *
```
cygo_superhuman_agent_001
```

---

## Files

**Input Materials** — upload `dist/cygo_input.tgz` (40 KB): agent-visible `input/` (rules, engine, baseline, harness, starter).

**Reference Output & Evaluation Dependencies** — the hidden reference agent + grader. Bundle is `dist/cygo_reference.tgz` (41 MB compressed; contains the 204 MB trained net `weights.pt`, `agent_proto.py`, engine, `run.sh`, and `reference_metrics.json`). Under 1 GB, so it can upload directly — or via the Drive link field below.

**Google Drive / Dropbox Link** (optional, only if uploading >1 GB): 
```
<paste an open-access Drive/Dropbox link to cygo_reference.tgz here if not uploading directly>
```

---

## How should we verify success? *
```
An independent referee (scripts/match_harness.py, shipped) plays 1000 games (500 as each
color) between the submitted output/run.sh and the HIDDEN reference agent, from forced
balanced 6-ply openings, under the exact stdin/stdout protocol and per-move wall-clock
budget (3.0 s) on one 24 GB GPU. Rules are authoritative via the shipped C++ engine:
cylinder column-wrap, Go capture, suicide illegal, positional superko, win by top-edge↔
bottom-edge connection, no-legal-move loss, ply-cap-600 stone-count adjudication (ties →
White). Any illegal move / malformed reply / crash / clock or memory-cap violation = loss
of that game. Score = clip(2 × candidate_win_rate, 0, 1): equal strength → 1.0, ~ -520 Elo
→ 0.10. Everything is mechanical — run 1000 games, count wins; self-reported results are
ignored; the reference is staged only at eval time.
```

---

## Difficulty Self-Test (optional)
- **Model:** Claude Opus 4.8
- **Harness:** Claude Code
- **Estimated score:** `<1% (last-exam)`
- **Evidence / basis (honest):** Design-time adversarial curation by frontier models (Fable 5.1 as blue, GPT-6-Astra + Fable as red, 4 rounds); both self-estimated ~0.06–0.12 with no shortcut found. This is a design-time estimate, NOT a measured frontier-attempt score — no frontier model has yet been graded against the reference, and a from-scratch reachability run under the agent budget is pending. The reference itself is a genuinely strong, verifiably-improving AlphaZero net (measured: sims-scaling +363 Elo confirming the engine is correct; the shipped seed +708 Elo over its early self; ~100% vs uniform-random).

---

## Confirmations
- [ ] I confirm this workflow can be used for Agents' Last Exam evaluation.
- [ ] I have read and agree to the Terms and Conditions.

---

### Notes for us (not form fields)
- Reference = cygo-run2 `global_step_977400` (more-trained of two seeds; c1 reached 712200).
- Full package + honest metrics on fork `ahmd-mohsin/agents-last-exam`, branch `cygo-last-exam-task`.
- Open decision: submitting on the curation-based difficulty estimate (honest, self-test optional). If a reviewer wants a measured frontier score, that's the known gap.

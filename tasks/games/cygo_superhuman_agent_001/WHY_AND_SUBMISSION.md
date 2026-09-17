# CYGO — Rationale & Submission Record (the "why")

Companion to `PROJECT_HANDBOOK.md` (the technical record). This explains *why the task is
built the way it is*, *why we want it to be unsolvable by today's frontier models*, and
*exactly how it was submitted*.

---

## 1. What Agents' Last Exam (ALE) is, and the goal

ALE is a benchmark of tasks meant to sit at (or just beyond) the frontier of what autonomous
coding/agent systems can do. A task "counts" as a **last-exam** task when the strongest
current models, given a full serious attempt in a capable harness, **score very low**
(target: frontier average **< 0.1**). Contributors who submit a validated last-exam task get
**co-authorship**. So the explicit objective is: *design a task that current frontier models
cannot solve* — not to trick them, but to genuinely exceed their capability.

## 2. Why we WANT this task to "fail" on frontier models

This is the whole point, and it is not adversarial in a dishonest sense:

- A benchmark is only informative where it **discriminates**. If frontier models already
  score high, the task tells you nothing about the frontier — it's saturated.
- A **low** frontier score is the *signal* that the task probes an ability current systems
  lack. The value of the contribution is precisely that a strong model, trying honestly,
  still falls short.
- "Fail" here means **cannot reach reference-competitive strength within the task budget** —
  not that we hid the answer or sabotaged them. Everything they need (rules, engine, harness,
  the intended AlphaZero plan) is handed over. They fail because *doing it* is hard, not
  because *knowing what to do* is hard.

The distinction we hold to: **hard to achieve, not hard to know.** A task that's only hard
because information is withheld is a bad last-exam task (a knowledgeable human or a web search
trivializes it). A task that's hard because it requires building and training a strong system
under real constraints is a *good* one.

## 3. Why a novel game with zero prior art

CYGO was invented for this exam. That is deliberate and load-bearing:

- **No pretrained transfer.** For Go/Chess/Hex/etc., models can lean on memorized theory,
  opening books, published engines, or fine-tuned weights. CYGO has **no engine, no weights,
  no opening book, no literature, no games anywhere**. A frontier model cannot retrieve its
  way to competence — it must *learn* the game from the rules.
- **The plan is not the bottleneck.** We literally hand over "use AlphaZero-style self-play."
  Knowing the recipe is free. The difficulty is the *execution*: a correct + fast rules engine
  for an unusual topology, batched MCTS, a distributed self-play/train loop, and a well-tuned
  run that actually reaches strength — all inside the task's wall-clock/compute budget.
- **The rules are chosen to be genuinely hard, not gimmicky.** Cylinder column-wrap +
  top↔bottom connection + Go capture + positional superko interact to create deep tactics
  (wrap-around connections, capture races, ko-like cycles) with a large branching factor on
  13×13. It is a real game, adversarially checked (below), not a puzzle with a hidden trick.

## 4. Why it can't be shortcut or reward-hacked

- **Grading is mechanical and outcome-only.** The referee (`match_harness.py`) plays 1000
  games against a **hidden** reference agent and counts wins: `score = clip(2·winrate, 0, 1)`.
  There is no rubric to game, no text to spoof, no partial credit for plausible-looking work.
  Self-reported results are ignored.
- **The reference is withheld until eval.** The agent never sees the opponent's weights or
  play, so it cannot overfit to it or look up an "answer."
- **Legality is enforced by the same engine the referee uses.** Illegal move / malformed
  reply / crash / clock or memory-cap violation = loss of that game. There's no way to "win"
  except by actually playing better CYGO.
- **No network at eval; the agent must stand alone.** It can't call out for help.

The only path to a high score is to *build an agent that genuinely plays CYGO well* — which is
exactly the capability we're claiming today's frontier models don't yet have under budget.

## 5. How we tried to break our own task (adversarial curation)

Before trusting the difficulty claim, we ran a red/blue curation loop with frontier models
(logs in `curation/`, consults in `consult/`):

- **Blue (Fable 5.1):** attempt the task in a full session; find any shortcut, any way to
  reach the bar cheaply.
- **Red (GPT-6-Astra + Fable):** attack the design — look for knowledge leaks, gameable
  grading, a reference that's too weak, a rules ambiguity, an isolation hole.
- **Four rounds.** Each objection was either fixed in the spec (rev-6) or shown to be covered.
  Both models' honest self-estimate converged to **~0.06–0.12** with **no shortcut found**.

This is what makes the low-score claim credible as a *design-time* estimate: the strongest
available models, actively trying, could not see a way to solve it cheaply.

## 6. Why the reference must be (and is) strong

If the hidden reference were weak, a frontier model could beat it and the task would score
high — invalid. So a real part of the work was *training a genuinely strong reference*:

- Distributed AlphaZero on ~48×A100 across **two independent seeds** (distinct region,
  account, RNG) — independence is evidence the strength is reachable and reproducible, not a
  fluke of one run.
- Verified strong by measurement (not assertion): search scales correctly (**+363 Elo** at
  512 vs 64 sims — proving the MCTS is real), the shipped seed is **+708 Elo** over its early
  self, and it beats a random baseline ~100%.
- Shipped reference = the more-trained seed, `cygo-run2 global_step_977400`.

## 7. What we are honest about NOT having

We hold the line on not overclaiming (this matters for a credible contribution):

- We did **not** run a full frontier-model *attempt* graded against the reference. The <0.1 is
  a **design-time adversarial-curation estimate**, not a measured frontier score.
- We did **not** complete an independent from-scratch reachability replication under the exact
  agent budget.
- These gaps are stated plainly in `reference_metrics.json`, `task_card.json`, and the
  evidence PDF. If a reviewer requires a measured frontier number, that is the known gap to
  close (spin up a fresh attempt and grade it).

## 8. Exactly how it was submitted

The ALE web form was filled from `SUBMISSION_FORM.md`:

1. **Industry Domain** → AI — Game-Playing / Reinforcement Learning.
2. **Software & Version** → Python 3.12, PyTorch 2.6, pybind11 (C++17). **OS** → Linux.
   **Licensing** → Free / Open Source.
3. **Task Description** → the Objective / Starting-State / Instructions-&-Rules text.
4. **Task Short Name** → `cygo_superhuman_agent_001`.
5. **Input Materials** → upload `dist/cygo_input.tgz` (agent-visible input/: rules, engine,
   baseline, harness, starter).
6. **Reference Output & Evaluation Dependencies** → upload `dist/cygo_reference.tgz`
   (hidden reference agent + 204 MB trained net + grader deps). >1 GB would use the Drive
   link field with open access; this bundle is 41 MB so it uploads directly.
7. **How to verify success** → the mechanical 1000-game rule: `score = clip(2·winrate, 0, 1)`
   against the hidden reference, illegal/crash/timeout = loss, rules authoritative via the
   shipped C++ engine.
8. **Difficulty Self-Test** → Model = Claude Opus 4.8, Harness = Claude Code, Estimated score
   = `<1% (last-exam)`; **Evidence** = `dist/verification_evidence.pdf`.
9. **Confirmations** → tick "usable for ALE evaluation" and "agree to Terms".

Everything that is not uploaded to the form is version-controlled on the fork
`ahmd-mohsin/agents-last-exam` (branch `cygo-last-exam-task`), except the weights (`*.pt` →
Drive), tarballs (`*.tgz`), and `greenland/` (creds/instance IDs).

## 9. One-line thesis

*CYGO is a real, novel game with no prior art, graded purely on whether your agent wins games
against a strong hidden reference. We want frontier models to fall short because that is the
evidence the task measures a capability — training a strong agent from nothing under budget —
that they do not yet have; and we verified the reference is genuinely strong and the design
has no shortcut, while stating honestly that the sub-0.1 figure is a curation estimate, not a
measured frontier grade.*

"""AgentHLE task: computing_math/coding_posttraining_recovery_001.

Recover a failed GRPO post-training run of a compact coding model and ship a
reproducible LoRA adapter that clears a per-cohort release contract on a SEALED,
family-disjoint evaluation pool, under a single-GPU (NVIDIA L4, 24 GB) budget.

The shipped run reward-hacked: a length-biased execution proxy drove the policy
onto short non-solutions, so the visible proxy reward climbed while independent
test-pass correctness regressed. The incident also carries co-occurring, realistic
faults (train/eval family leakage that inflates the team's own validation,
mislabeled replay solutions, a completion-mask defect in the trainer, and
catastrophic forgetting of a protected synthesis capability under naive retraining)
so that no single fix recovers. The valuable outcome is a trustworthy release, not
a diagnosis document.

Lifecycle mirrors the framework contract: input/ (agent-visible: the training repo
with the faults, the regressed adapter + checkpoints, failed-run logs, visible
validation, base weights, release_contract.json, public smoke grader) and
reference/ (evaluator-only: sealed family-disjoint pool + protected cohort +
oracles, the genuine recovered adapter, exact recovery command, measured metrics,
negative controls) are staged before the agent runs; reference/ is withheld until
eval. evaluate() runs a self-contained in-VM verifier that reloads ONLY the
submitted adapter tensors into the pinned base model, recomputes per-cohort
pass@1/pass@8 on the sealed pool, and returns the release-contract coverage score.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "coding_posttraining_recovery_001"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/dev/shm/agenthle_eval/{TASK_NAME}"
VERIFIER_POLL_S = 20
VERIFIER_TIMEOUT_S = 5400

_setup = BaseTaskSetup()


@dataclass
class CodingRecoveryConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def release_contract(self) -> str:
        return f"{self.input_dir}/release_contract.json"

    @property
    def adapter_path(self) -> str:
        # Agent's final deliverable: a loadable PEFT LoRA adapter directory.
        return f"{self.remote_output_dir}/adapter"

    @property
    def reference_metrics(self) -> str:
        return f"{self.reference_dir}/reference_metrics.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are an ML engineer recovering a FAILED post-training run of a compact coding
model (pinned Qwen2.5-Coder-1.5B-Instruct) on Linux, with ONE GPU (NVIDIA L4,
24 GB). The run's visible reward improved while independently tested code
correctness regressed. Ship a trustworthy replacement adapter, not a write-up.

## Task Root
- `{self.task_dir}`

## Visible Inputs (read-only)
- Task brief: `{self.task_brief}`
- Release contract (fixed inference settings, per-cohort thresholds, compute
  budget, public smoke grader): `{self.release_contract}`
- Under `{self.input_dir}`: repo/ (the actual trainer, reward impl, data
  generator, config, and representative logs from the failed workflow),
  adapters/ (the regressed adapter + every intermediate checkpoint; each is a
  candidate rollback), data/ (the team's train + visible validation splits),
  base_model/ (full local weights + tokenizer), replay/ (verified training
  solutions), smoke/ (a public subset grader you can run yourself).

## What You Should Do
1. Read `{self.task_brief}` and `{self.release_contract}`.
2. Recompute independent correctness + coverage for the supplied checkpoints;
   reconcile the reward logs against executable results to locate what regressed.
3. Establish provenance of the training data and replay solutions (the team's own
   validation may be inflated; some supplied "verified" solutions are wrong).
4. Choose and justify a recovery (data, objective, replay, stopping, anchoring);
   short comparison runs are expected. Naive single fixes do not clear the bar.
5. Train a replacement adapter within the single-GPU 24 GB budget and validate on
   family-disjoint problems with the fixed decoding protocol.
6. Write a reproducible release under `{self.remote_output_dir}`.

## Final Deliverable (under `{self.remote_output_dir}`)
- `adapter/` : a loadable PEFT LoRA adapter (adapter_model.safetensors +
  adapter_config.json) for the pinned base; the evaluator loads ONLY these
  tensors into its own copy of the base model.
- `train.py`, `config.yaml`, `reproduce.sh` : bounded, deterministic recipe.
- `diagnosis.md`, `metrics.json`, `provenance.json`.

The evaluator recomputes ALL metrics from a fresh run of your submitted adapter on
a SEALED, family-disjoint pool plus a protected-capability cohort, under the fixed
protocol in the release contract. Self-reported metrics get no credit. An
inference wrapper, external call, answer lookup, or any evaluator-dependent
behavior is out of contract. Do not modify `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        m = super().to_metadata()
        m.update({
            "task_id": TASK_ID,
            "task_brief": self.task_brief,
            "release_contract": self.release_contract,
            "adapter_path": self.adapter_path,
            "reference_metrics": self.reference_metrics,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = CodingRecoveryConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Prepare a clean output dir; assert the reference stays hidden during setup.

    Task data (input/ + reference/) is staged by the framework; reference/ is
    withheld until eval. This hook must not read reference/.
    """
    await _setup(task_cfg, session)
    meta = task_cfg.metadata
    out = meta["remote_output_dir"]
    await session.run_command(f"rm -rf {shlex.quote(out)} && mkdir -p {shlex.quote(out)}", check=False)
    for p in (meta["task_brief"], meta["release_contract"]):
        try:
            await session.read_file(p)
        except Exception as exc:
            raise RuntimeError(f"staged input missing: {p} ({exc})")
    if await session.directory_exists(meta["reference_dir"]):
        raise RuntimeError(
            f"reference visible during setup ({meta['reference_dir']}); the "
            "framework must stage reference/ only at eval time.")
    logger.info("[%s] input staged; output clean; reference hidden", VARIANT_NAME)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Run the self-contained verifier on the sealed pool. Returns the release
    coverage score in [0,1]. Never raises on missing/broken output -> 0.0."""
    meta = task_cfg.metadata
    tag = VARIANT_NAME
    for p in (meta["reference_dir"], meta["reference_metrics"]):
        if not (await session.file_exists(p) or await session.directory_exists(p)):
            logger.error("[%s] missing reference: %s", tag, p); return [0.0]
    if not await session.directory_exists(meta["adapter_path"]):
        logger.error("[%s] missing adapter dir: %s", tag, meta["adapter_path"]); return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    run_dir = (await session.run_command(
        "bash -lc " + json.dumps(f"mktemp -d {shlex.quote(EVAL_TMP_DIR)}/run_XXXXXX"),
        check=False)).get("stdout", "").strip().splitlines()[-1:]
    run_dir = run_dir[0] if run_dir else ""
    if not run_dir:
        logger.error("[%s] failed to create verifier run dir", tag); return [0.0]

    verifier = f"{run_dir}/verify_submission.py"
    await session.write_file(verifier, (SCRIPTS_DIR / "verify_submission.py").read_text("utf-8"))
    await session.write_file(f"{run_dir}/sandbox.py", (SCRIPTS_DIR / "sandbox.py").read_text("utf-8"))
    cmd = (f"python3 {shlex.quote(verifier)} "
           f"--adapter {shlex.quote(meta['adapter_path'])} "
           f"--base {shlex.quote(meta['input_dir'])}/base_model "
           f"--reference-dir {shlex.quote(meta['reference_dir'])} "
           f"--contract {shlex.quote(meta['release_contract'])}")
    stdout = await _run_bg(session, cmd, tag, run_dir)
    try:
        payload = _parse_json(stdout)
    except Exception as exc:
        logger.error("[%s] verifier JSON parse failed: %s", tag, exc); return [0.0]
    score = float(payload.get("normalized_score", 0.0))
    logger.info("[%s] normalized_score=%.4f passed=%s cohorts=%s", tag, score,
                payload.get("passed"), payload.get("by_cohort"))
    return [max(0.0, min(1.0, score))]


def _parse_json(raw: str) -> dict:
    for line in reversed([l.strip() for l in (raw or "").splitlines() if l.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON in verifier stdout: {(raw or '')[:400]}")


async def _run_bg(session, cmd, tag, run_dir):
    out, rc = f"{run_dir}/v.out", f"{run_dir}/v.rc"
    wrapped = f"{cmd} > {shlex.quote(out)} 2>&1; printf '%s' \"$?\" > {shlex.quote(rc)}"
    launch = await session.run_command(
        "bash -lc " + json.dumps(f"nohup bash -lc {shlex.quote(wrapped)} >/dev/null 2>&1 & echo $!"),
        check=False)
    pid = (launch.get("stdout") or "").strip().splitlines()[-1:] or [""]
    waited = 0
    while waited < VERIFIER_TIMEOUT_S:
        await asyncio.sleep(VERIFIER_POLL_S); waited += VERIFIER_POLL_S
        done = await session.run_command(
            "bash -lc " + json.dumps(f"test -f {shlex.quote(rc)} && echo DONE || echo RUN"), check=False)
        if (done.get("stdout") or "").strip() == "DONE":
            tail = await session.run_command(
                "bash -lc " + json.dumps(f"tail -n 40 {shlex.quote(out)}"), check=False)
            return tail.get("stdout", "") or ""
    if pid and pid[0].isdigit():
        await session.run_command(f"kill {pid[0]} 2>/dev/null || true", check=False)
    raise TimeoutError(f"verifier timed out after {VERIFIER_TIMEOUT_S}s")

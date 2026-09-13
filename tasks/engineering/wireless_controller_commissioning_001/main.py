"""AgentHLE task: engineering/wireless_controller_commissioning_001.

Commission an energy-efficient multi-user MIMO uplink power controller from
imperfect, delayed channel measurements. Long-horizon workflow: measurement
audit -> delayed-CSI prediction -> joint uncertainty calibration over the
declared operating envelope -> energy-optimal robust power control -> operate
Sionna for the coded-link relationship -> reproducible controller release.

Lifecycle: the framework stages input/ (agent-visible: brief, radio_system,
train/dev/envelope archive, public_eval, starter) and reference/ (evaluator-only:
hidden packs, metrics) before the agent runs; reference/ is withheld until eval.
evaluate() runs a self-contained in-VM verifier that replays the submitted
controller on the withheld hidden packs and returns the coverage score.
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

DOMAIN_NAME = "engineering"
TASK_NAME = "wireless_controller_commissioning_001"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/dev/shm/agenthle_eval/{TASK_NAME}"
VERIFIER_POLL_S = 20
VERIFIER_TIMEOUT_S = 5400

_setup = BaseTaskSetup()


@dataclass
class WirelessControllerConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def radio_system(self) -> str:
        return f"{self.input_dir}/radio_system.json"

    @property
    def controller_path(self) -> str:
        return f"{self.remote_output_dir}/controller.py"

    @property
    def reference_metrics(self) -> str:
        return f"{self.reference_dir}/reference_metrics.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are a wireless systems engineer commissioning a replacement uplink power
controller for a 4-user, 8-antenna MIMO base station over 24 OFDM tones
(3GPP TR 38.901 UMi), on Linux.

## Task Root
- `{self.task_dir}`

## Visible Inputs (read-only)
- Task brief: `{self.task_brief}`
- Radio-system spec: `{self.radio_system}`
- Data + tools under `{self.input_dir}`: archive/ (train/dev/envelope true-channel
  trajectories), public_eval/ (dev replay, numerical checks, Sionna coded-link
  curve/tools), starter/ (conservative baseline + the Controller interface stub).

## What You Should Do
1. Read `{self.task_brief}` and `{self.radio_system}`.
2. The latest channel estimate is DELAY frames stale and the channel evolves with
   Doppler; build a predictor (persistence is far too stale) and calibrate its
   JOINT uncertainty over the declared operating envelope (not just nominal dev).
3. Solve energy-optimal robust power control (raising power blindly increases
   mutual interference and hurts reliability).
4. Operate Sionna (public_eval) to relate post-combining SINR to coded decoding.
5. Write `{self.controller_path}` implementing the Controller interface (see
   starter/), plus its fitted artifacts and a reproduction script, under
   `{self.remote_output_dir}`.

## Final Deliverable
- `{self.controller_path}` implementing:
    class Controller:
        def __init__(self, ctx): ...        # fit offline on archive; no hidden access
        def allocate(self, hist_est, demand):
            # hist_est: torch complex (B, L, U, F, M); demand: torch (U,)
            # returns powers (B, U, F) real >= 0      # POWERS ONLY

The receiver combiner is EVALUATOR-CONTROLLED: the evaluator builds a per-tone
MMSE combiner from the LS estimate of the most recent OBSERVED frame
(hist_est[:, -1]) at nominal power 1, and scores true-channel SINR with that
(stale, imperfect) combiner and your powers. You cannot change it; your power
allocation must be robust to it and to the channel evolution.

Your release is replayed by an independent evaluator on withheld channels from a
shifted deployment regime. Do not modify `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        m = super().to_metadata()
        m.update({
            "task_id": TASK_ID,
            "task_brief": self.task_brief,
            "radio_system": self.radio_system,
            "controller_path": self.controller_path,
            "reference_metrics": self.reference_metrics,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = WirelessControllerConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
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
    for p in (meta["task_brief"], meta["radio_system"]):
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
    """Run the self-contained verifier on the withheld hidden packs. Returns the
    coverage score in [0,1]. Never raises on missing/broken output -> 0.0."""
    meta = task_cfg.metadata
    tag = VARIANT_NAME
    for p in (meta["reference_dir"], meta["reference_metrics"]):
        if not (await session.file_exists(p) or await session.directory_exists(p)):
            logger.error("[%s] missing reference: %s", tag, p); return [0.0]
    if not await session.file_exists(meta["controller_path"]):
        logger.error("[%s] missing controller: %s", tag, meta["controller_path"]); return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    run_dir = (await session.run_command(
        "bash -lc " + json.dumps(f"mktemp -d {shlex.quote(EVAL_TMP_DIR)}/run_XXXXXX"),
        check=False)).get("stdout", "").strip().splitlines()[-1:]
    run_dir = run_dir[0] if run_dir else ""
    if not run_dir:
        logger.error("[%s] failed to create verifier run dir", tag); return [0.0]

    verifier = f"{run_dir}/verify_submission.py"
    await session.write_file(verifier, (SCRIPTS_DIR / "verify_submission.py").read_text("utf-8"))
    cmd = (f"python3 {shlex.quote(verifier)} --submission {shlex.quote(meta['controller_path'])} "
           f"--input-dir {shlex.quote(meta['input_dir'])} "
           f"--reference-dir {shlex.quote(meta['reference_dir'])}")
    stdout = await _run_bg(session, cmd, tag, run_dir)
    try:
        payload = _parse_json(stdout)
    except Exception as exc:
        logger.error("[%s] verifier JSON parse failed: %s", tag, exc); return [0.0]
    score = float(payload.get("normalized_score", 0.0))
    logger.info("[%s] normalized_score=%.4f passed=%s", tag, score, payload.get("passed"))
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

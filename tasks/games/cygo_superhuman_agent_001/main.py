"""AgentHLE task: games/cygo_superhuman_agent_001.

Build a superhuman agent for CYGO (Cylinder Gonnect, a novel 13x13 Go variant). The
agent is graded by win-rate over 1000 games against a HIDDEN reference agent trained
by an AlphaZero-style self-play pipeline; score = clip(2 * win-rate, 0, 1). The plan is
handed over; the difficulty is achieving it under the single-GPU match budget on a novel
game where nothing pretrained transfers.

Lifecycle: input/ (agent-visible: rules engine, random-rollout baseline bot, the exact
match harness + protocol, starter run.sh) and reference/ (evaluator-only: the hidden
reference agent weights + config) are staged before the agent runs; reference/ is
withheld until eval. evaluate() runs the self-contained match referee (scripts/
match_harness.py) between the submitted output/run.sh and the hidden reference agent and
returns the coverage score. Each agent process runs under the compute-isolation wrapper
(scripts/ISOLATION.md; per-game container, MPS 24 GB cap, seccomp allow-list,
freeze-then-drain turn clock, DCGM/NVML off-turn oracles).
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

DOMAIN_NAME = "games"
TASK_NAME = "cygo_superhuman_agent_001"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/dev/shm/agenthle_eval/{TASK_NAME}"
VERIFIER_POLL_S = 30
VERIFIER_TIMEOUT_S = 21600  # match of 1000 games at up to 3s/move can be long

_setup = BaseTaskSetup()


@dataclass
class CygoConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def rules_json(self) -> str:
        return f"{self.input_dir}/cygo_rules.json"

    @property
    def run_sh(self) -> str:
        return f"{self.remote_output_dir}/run.sh"

    @property
    def reference_agent(self) -> str:
        return f"{self.reference_dir}/reference_agent"   # dir: weights + run cmd

    @property
    def reference_metrics(self) -> str:
        return f"{self.reference_dir}/reference_metrics.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are building a superhuman agent for CYGO (Cylinder Gonnect), a novel 13x13 board
game invented for this exam. No engine weights, opening book, or literature exists for
it. On Linux with ONE GPU (NVIDIA L4, 24 GB) for match play; you have a long wall-clock
budget on this box for development and training.

## Task Root
- `{self.task_dir}`

## Visible Inputs (read-only)
- Rules + brief: `{self.task_brief}`, `{self.rules_json}` (board, cylinder wrap, capture,
  superko, win-by-top-bottom-connection, ply cap, the 52-element symmetry group).
- Under `{self.input_dir}`: engine/ (the exact CYGO rules engine + 52 transforms, matching
  the referee), baseline/ (a random-rollout MCTS bot), harness/ (the EXACT match harness
  and stdin/stdout protocol used to grade you), starter/ (a run.sh template + protocol
  agent stub).

## What You Should Do
1. Read `{self.task_brief}` and `{self.rules_json}`.
2. Build and train a CYGO agent (AlphaZero-style self-play is the intended approach; the
   plan is not the hard part, reaching the bar is).
3. Deliver `{self.run_sh}`: an executable implementing the match protocol over stdin/stdout
   (`newgame <color> <seed>`, `[opening ...] move <oc> <or>|-` -> `= <col> <row>`, `quit`),
   playing within the per-move wall-clock budget and 24 GB, plus your weights/artifacts
   under `{self.remote_output_dir}`.

## Grading
An independent referee plays 1000 games (500 as each color) between your `{self.run_sh}`
and a HIDDEN reference agent (AlphaZero-trained), from forced balanced openings, under the
exact harness in input/harness/. Illegal move / malformed reply / crash / clock or
memory-cap violation = loss of that game. Score = clip(2 * your win-rate, 0, 1) (equal
strength -> 1.0; ~ -520 Elo -> 0.10). The reference is staged only at eval. Do not modify
`{self.input_dir}`; no network at eval; the agent must stand alone (no answer lookup).
"""

    def to_metadata(self) -> dict[str, Any]:
        m = super().to_metadata()
        m.update({
            "task_id": TASK_ID,
            "task_brief": self.task_brief,
            "rules_json": self.rules_json,
            "run_sh": self.run_sh,
            "reference_agent": self.reference_agent,
            "reference_metrics": self.reference_metrics,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = CygoConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    meta = task_cfg.metadata
    out = meta["remote_output_dir"]
    await session.run_command(f"rm -rf {shlex.quote(out)} && mkdir -p {shlex.quote(out)}", check=False)
    for p in (meta["task_brief"], meta["rules_json"]):
        try:
            await session.read_file(p)
        except Exception as exc:
            raise RuntimeError(f"staged input missing: {p} ({exc})")
    if await session.directory_exists(meta["reference_dir"]):
        raise RuntimeError(
            f"reference visible during setup ({meta['reference_dir']}); the framework "
            "must stage reference/ only at eval time.")
    logger.info("[%s] input staged; output clean; reference hidden", VARIANT_NAME)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Run the match referee: submitted run.sh vs the hidden reference agent. Returns
    coverage score in [0,1]. Never raises on missing/broken output -> 0.0."""
    meta = task_cfg.metadata
    tag = VARIANT_NAME
    for p in (meta["reference_dir"], meta["reference_metrics"]):
        if not (await session.file_exists(p) or await session.directory_exists(p)):
            logger.error("[%s] missing reference: %s", tag, p); return [0.0]
    if not await session.file_exists(meta["run_sh"]):
        logger.error("[%s] missing submission run.sh: %s", tag, meta["run_sh"]); return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    run_dir = (await session.run_command(
        "bash -lc " + json.dumps(f"mktemp -d {shlex.quote(EVAL_TMP_DIR)}/run_XXXXXX"),
        check=False)).get("stdout", "").strip().splitlines()[-1:]
    run_dir = run_dir[0] if run_dir else ""
    if not run_dir:
        logger.error("[%s] failed to create verifier run dir", tag); return [0.0]

    for fn in ("match_harness.py", "cygo_cpp.cpp", "cygo_engine.py"):
        await session.write_file(f"{run_dir}/{fn}", (SCRIPTS_DIR / fn).read_text("utf-8"))
    # build the referee engine in-VM (matches the shipped engine)
    await session.run_command(
        "bash -lc " + json.dumps(
            f"cd {shlex.quote(run_dir)} && g++ -O3 -shared -std=c++17 -fPIC "
            "$(python3 -m pybind11 --includes) cygo_cpp.cpp "
            "-o cygo_cpp$(python3-config --extension-suffix)"), check=False)
    cand = f"bash {shlex.quote(meta['run_sh'])}"
    ref = f"bash {shlex.quote(meta['reference_agent'])}/run.sh"
    cmd = (f"cd {shlex.quote(run_dir)} && python3 match_harness.py "
           f"--candidate {shlex.quote(cand)} --reference {shlex.quote(ref)} "
           f"--games 1000 --budget 3.0 --out {shlex.quote(run_dir)}/match.json")
    stdout = await _run_bg(session, cmd, tag, run_dir)
    try:
        payload = _parse_json(stdout)
    except Exception as exc:
        logger.error("[%s] harness JSON parse failed: %s", tag, exc); return [0.0]
    score = float(payload.get("score", 0.0))
    logger.info("[%s] cand_winrate=%.4f score=%.4f", tag, payload.get("cand_winrate", 0.0), score)
    return [max(0.0, min(1.0, score))]


def _parse_json(raw: str) -> dict:
    for line in reversed([l.strip() for l in (raw or "").splitlines() if l.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON in harness stdout: {(raw or '')[:400]}")


async def _run_bg(session, cmd, tag, run_dir):
    out, rc = f"{run_dir}/h.out", f"{run_dir}/h.rc"
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
                "bash -lc " + json.dumps(f"tail -n 5 {shlex.quote(out)}"), check=False)
            return tail.get("stdout", "") or ""
    if pid and pid[0].isdigit():
        await session.run_command(f"kill {pid[0]} 2>/dev/null || true", check=False)
    raise TimeoutError(f"match harness timed out after {VERIFIER_TIMEOUT_S}s")

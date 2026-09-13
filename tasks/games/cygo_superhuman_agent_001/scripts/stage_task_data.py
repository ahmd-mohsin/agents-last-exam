"""Assemble the CYGO task's input/ (agent-visible) and reference/ (evaluator-only)
bundles from the shipped scripts + a trained reference checkpoint.

input/                          reference/
  task_brief.md                   reference_agent/
  cygo_rules.json                   run.sh          (launches agent_proto on the reference weights)
  engine/    cygo_engine.py         weights.pt      (the trained reference net; pulled from S3)
             cygo_cpp.cpp           agent_proto.py + cygo_engine.py + cygo_cpp.cpp
  baseline/  agent_proto.py (random-rollout smoke opponent)
  harness/   match_harness.py agent_proto.py cygo_cpp.cpp cygo_engine.py
  starter/   run.sh (=starter_run.sh) agent_proto.py cygo_cpp.cpp cygo_engine.py
  reference_metrics.json  (score->Elo map + reachability numbers)

The reference weights come from a matured training run (s3ckpt cygo-run1 latest). Usage:
  python3 stage_task_data.py --out <task_data_dir> --reference-weights <ckpt.pt>
Produces <out>/input and <out>/reference; upload as the task_data_source.
"""
import argparse, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = ["cygo_engine.py", "cygo_cpp.cpp"]
AGENT = ["agent_proto.py", "cygo_engine.py", "cygo_cpp.cpp"]


def cp(names, dst):
    os.makedirs(dst, exist_ok=True)
    for n in names:
        shutil.copy(os.path.join(HERE, n), os.path.join(dst, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-weights", required=True, help="trained reference net ckpt.pt")
    ap.add_argument("--ch", type=int, default=256); ap.add_argument("--blocks", type=int, default=15)
    a = ap.parse_args()
    inp = os.path.join(a.out, "input"); ref = os.path.join(a.out, "reference")

    # input/
    cp([], inp)
    for f in ("task_brief.md", "cygo_rules.json"):
        shutil.copy(os.path.join(HERE, f), os.path.join(inp, f))
    cp(ENGINE, os.path.join(inp, "engine"))
    cp(AGENT, os.path.join(inp, "baseline"))
    cp(["match_harness.py"] + AGENT, os.path.join(inp, "harness"))
    cp(AGENT, os.path.join(inp, "starter"))
    shutil.copy(os.path.join(HERE, "starter_run.sh"), os.path.join(inp, "starter", "run.sh"))

    # reference/ (hidden)
    ra = os.path.join(ref, "reference_agent")
    cp(AGENT, ra)
    shutil.copy(a.reference_weights, os.path.join(ra, "weights.pt"))
    with open(os.path.join(ra, "run.sh"), "w") as f:
        f.write('#!/usr/bin/env bash\ncd "$(dirname "$0")"\n'
                f'exec python3 agent_proto.py --mode mcts --weights weights.pt '
                f'--ch {a.ch} --blocks {a.blocks} --sims 200\n')
    os.chmod(os.path.join(ra, "run.sh"), 0o755)
    with open(os.path.join(ref, "reference_metrics.json"), "w") as f:
        json.dump({
            "score_formula": "clip(2*winrate,0,1)",
            "elo_to_score": {"-100": 0.72, "-300": 0.30, "-520": 0.10, "-700": 0.036, "-1000": 0.007},
            "reachability": {"second_independent_seed": "TBD (cygo-run2)",
                             "from_scratch_replication_agent_budget": "TBD",
                             "note": "filled from measured runs before submission"},
            "reference_net": {"ch": a.ch, "blocks": a.blocks, "arch": "column-circular ResNet, policy+value"}
        }, f, indent=1)
    print(f"staged input/ + reference/ under {a.out}")


if __name__ == "__main__":
    main()

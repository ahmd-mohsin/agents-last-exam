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
            "elo_to_score_note": "logistic Elo->win-rate then clip(2*wr); a fixed mathematical mapping, not measured.",
            "reference_net": {"ch": a.ch, "blocks": a.blocks,
                              "arch": "column-circular (cylinder-equivariant) ResNet, policy+value",
                              "shipped_checkpoint": "cygo-run2 global_step_977400 (the more-trained seed)",
                              "training": "AlphaZero self-play (distributed ~48xA100), sims-256 self-play, "
                                          "4M-game replay, AdamW + warmup->cosine. Net search scales strongly "
                                          "(+363 Elo at 512 vs 64 sims) and beats uniform-random 100%."},
            "measured_strength": {
                "sims_scaling_512_vs_64": "0.890 winrate / +363 Elo (same net; confirms MCTS is correct)",
                "c2_step977k_vs_early_272900": "0.983 winrate / +708 Elo (genuine large improvement)",
                "c1_sims256_window_218k_vs_165k": "0.592 / +64 Elo",
                "vs_uniform_random": "~1.0 (policy and MCTS)",
                "method": "batched_match.py, 120 games, 128 sims, noise 0.0, on a held GPU"},
            "reachability": {
                "independent_seeds_trained": "2 (cygo-run1 -> step 712200, cygo-run2 -> step 977400; "
                                             "distinct region/account/seed)",
                "from_scratch_replication_under_agent_budget": "NOT independently verified",
                "difficulty_basis": "adversarial curation by frontier models (Fable 5.1 blue, "
                                    "GPT-6-Astra + Fable red, 4 rounds); their in-session self-estimate "
                                    "was ~0.06-0.12 with no shortcut found. This is a design-time "
                                    "estimate, NOT an empirical frontier-attempt result.",
                "honesty_note": "Empirical frontier self-test and from-scratch reachability run are "
                                "pending; do not cite the 0.06-0.12 as a measured score."}
        }, f, indent=1)
    print(f"staged input/ + reference/ under {a.out}")


if __name__ == "__main__":
    main()

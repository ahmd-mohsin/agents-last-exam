"""Self-contained in-VM grader for coding_posttraining_recovery_001.

Loads ONLY the submitted LoRA adapter tensors into a fresh copy of the pinned base
model (agent training code is never imported), regenerates K completions per
problem under the FIXED protocol from release_contract.json, executes them in the
isolated sandbox, and scores per predeclared cohort. Release coverage:

    S = mean_cohort 1[ pass1_cohort >= T1_cohort AND pass8_cohort >= T8_cohort ]

Global validity failures (unloadable adapter, wrong base compatibility, resource
budget violation) score 0.0 regardless of cohort metrics. Prints a single JSON
line to stdout: {normalized_score, passed, by_cohort, resource, protocol}.

Cohorts live under <reference-dir>/sealed/<cohort>.jsonl, each row
{id, family, cohort, prompt, tests:[[args, expected], ...]}. Families in the
sealed pool are DISJOINT from the training families; protected_synthesis is a
from-docstring synthesis capability the release must not regress.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sandbox import extract_code, run_tests


def load_model(base, adapter, dev):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(dev)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def gen_batch(tok, model, prompts, k, dev, temp, top_p, max_new):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=1024, add_special_tokens=False).to(dev)
    gkw = dict(max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p,
               num_return_sequences=k, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    with torch.no_grad():
        out = model.generate(**enc, **gkw)
    gen = out[:, enc["input_ids"].shape[1]:]
    dec = tok.batch_decode(gen, skip_special_tokens=True)
    return [dec[i * k:(i + 1) * k] for i in range(len(prompts))]


def load_cohorts(reference_dir):
    cohorts = {}
    for path in sorted(glob.glob(os.path.join(reference_dir, "sealed", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        cohorts[name] = [json.loads(l) for l in open(path) if l.strip()]
    return cohorts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--contract", required=True)
    args = ap.parse_args()

    result = {"normalized_score": 0.0, "passed": False, "by_cohort": {},
              "resource": {}, "protocol": {}, "error": None}
    try:
        contract = json.load(open(args.contract))
        proto = contract["protocol"]
        k = int(proto["k"]); temp = float(proto["temperature"])
        top_p = float(proto["top_p"]); max_new = int(proto["max_new_tokens"])
        bs = int(proto.get("batch_size", 12))
        seed = int(proto.get("seed", 1234))
        budget_gb = float(contract["resource"]["max_eval_vram_gb"])
        thresholds = contract["cohort_thresholds"]  # {cohort: {pass1, pass8}}
        result["protocol"] = {"k": k, "temperature": temp, "top_p": top_p,
                              "max_new_tokens": max_new, "seed": seed}

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()

        cohorts = load_cohorts(args.reference_dir)
        if not cohorts:
            result["error"] = "no sealed cohorts found"; print(json.dumps(result)); return

        t0 = time.time()
        tok, model = load_model(args.base, args.adapter, dev)

        by_cohort = {}
        for name, probs in cohorts.items():
            p1s, p8s = [], []
            for i in range(0, len(probs), bs):
                chunk = probs[i:i + bs]
                comps = gen_batch(tok, model, [p["prompt"] for p in chunk], k, dev,
                                  temp, top_p, max_new)
                for p, cs in zip(chunk, comps):
                    c = 0
                    for comp in cs:
                        npass, nran, ntot, _ = run_tests(extract_code(comp), p["tests"])
                        if ntot and npass == ntot:
                            c += 1
                    p1s.append(c / k)
                    p8s.append(1.0 if c > 0 else 0.0)
            by_cohort[name] = {"pass1": sum(p1s) / len(p1s), "pass8": sum(p8s) / len(p8s),
                               "n": len(p1s)}

        peak_gb = (torch.cuda.max_memory_reserved() / 2**30) if dev == "cuda" else 0.0
        wall = time.time() - t0
        result["resource"] = {"eval_peak_vram_gb": round(peak_gb, 3),
                              "eval_wall_seconds": round(wall, 1),
                              "budget_vram_gb": budget_gb}

        # Global validity: eval must fit the declared budget.
        if peak_gb > budget_gb + 1e-6:
            result["by_cohort"] = by_cohort
            result["error"] = f"eval VRAM {peak_gb:.2f} GB exceeds budget {budget_gb} GB"
            print(json.dumps(result)); return

        inds = {}
        for name, m in by_cohort.items():
            th = thresholds.get(name)
            if th is None:
                inds[name] = 0
                continue
            inds[name] = int(m["pass1"] >= th["pass1"] and m["pass8"] >= th["pass8"])
            m["threshold"] = th
            m["meets"] = bool(inds[name])
        # Cohorts declared in the contract but absent from the pool cannot pass.
        for name in thresholds:
            inds.setdefault(name, 0)

        S = sum(inds.values()) / len(inds)
        result["by_cohort"] = by_cohort
        result["normalized_score"] = round(S, 6)
        result["passed"] = bool(S >= 1.0 - 1e-9)
    except Exception as exc:  # any failure -> 0.0, reported
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-1500:]
    print(json.dumps(result))


if __name__ == "__main__":
    main()

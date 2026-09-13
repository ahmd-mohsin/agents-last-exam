#!/usr/bin/env python3
"""Self-contained in-VM verifier for engineering/wireless_controller_commissioning_001.

Runs on the evaluation VM. Loads the submitted controller (output/controller.py),
scores it against the withheld hidden acceptance packs under the published
contract, and prints one JSON object with the normalized coverage score.

Rigor guarantees:
  * The scoring primitives (pilot observation, MMSE combiner, true SINR, rate)
    are embedded here verbatim so evaluation cannot drift from the reference.
  * Pilot noise is SEEDED per pack, so scoring is deterministic and reproducible.
  * The candidate's allocate() only ever receives the delayed pilot-estimate
    HISTORY and the demand; it never sees the true serve-frame channel. Ground
    truth and the scorer stay in this process; only observations cross into the
    controller call.
  * Missing / unloadable / crashing candidate output scores 0.0 (never raises).

Contract (from reference/reference_metrics.json): per-pack pass iff worst
per-user service-failure <= reliability_fail_max AND mean energy <= energy_budget.
Normalized score = fraction of hidden packs that pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
import zarr

Tensor = torch.Tensor


# ---- embedded scoring primitives (must match the construction/reference code) ----

def observe(H: Tensor, noise_var: float, delay: int, seed: int):
    """H (N,T,U,F,M) true channel -> (hist_est up to the delay, serve_true).
    Pilot power is unity, so the LS estimate is H + CN(0, noise_var). Seeded."""
    g = torch.Generator(device=H.device).manual_seed(seed)
    r = torch.randn(H.shape, generator=g, device=H.device)
    i = torch.randn(H.shape, generator=g, device=H.device)
    noise = ((r + 1j * i) / math.sqrt(2.0)).to(torch.complex64) * math.sqrt(noise_var)
    Hhat = H + noise
    last_obs = H.shape[1] - 1 - delay
    return Hhat[:, : last_obs + 1], H[:, -1]


def mmse_combiner(Hhat_frame: Tensor, noise_var: float, nominal_power: float) -> Tensor:
    """(N,U,F,M) channel prediction -> per-tone MMSE combiner g (N,U,F,M) at the
    nominal power. Evaluator-fixed: built from the candidate's PREDICTION."""
    b, U, F, M = Hhat_frame.shape
    Rn = noise_var * torch.eye(M, dtype=torch.complex64, device=Hhat_frame.device)
    Hf = Hhat_frame.permute(0, 2, 3, 1)                       # (b,F,M,U)
    Ry = nominal_power * (Hf @ Hf.conj().transpose(-1, -2)) + Rn
    return torch.linalg.solve(Ry, Hf).permute(0, 3, 1, 2)     # (b,U,F,M)


def true_sinr(g: Tensor, H_true: Tensor, powers: Tensor, noise_var: float) -> Tensor:
    """SINR (N,U,F) from the TRUE serve-frame channel, fixed combiner, powers."""
    a = (torch.einsum("bufm,bvfm->buvf", g.conj(), H_true).abs() ** 2)  # (b,U,U,F)
    gnorm2 = (g.abs() ** 2).sum(-1)
    U = g.shape[1]
    diag = torch.diagonal(a, dim1=1, dim2=2).permute(0, 2, 1)  # (b,U,F)
    sig = diag * powers
    interf = torch.einsum("buvf,bvf->buf", a, powers) - sig
    return sig / (interf + noise_var * gnorm2 + 1e-12)


def load_controller(path, ctx):
    spec = importlib.util.spec_from_file_location("submission_controller", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_controller"] = mod
    spec.loader.exec_module(mod)
    return mod.Controller(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--input-dir", required=True, help="staged input/ (archive, radio_system)")
    ap.add_argument("--reference-dir", required=True, help="staged reference/ (hidden, metrics)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    result = {"normalized_score": 0.0, "passed": False, "packs": [], "error": None}
    try:
        radio = json.load(open(f"{args.input_dir}/radio_system.json"))
        ref = json.load(open(f"{args.reference_dir}/reference_metrics.json"))
    except Exception as e:
        result["error"] = f"missing spec/metrics: {e}"; print(json.dumps(result)); return

    task = radio.get("task", {})
    delay = int(task.get("delay_frames", 2))
    demand_v = float(task.get("demand_bits_per_frame", 14.0))
    noise_var = float(radio.get("measurement", {}).get("noise", {}).get("noise_var", 2e-3)) \
        if isinstance(radio.get("measurement"), dict) else 2e-3
    nominal = 1.0
    rel = float(ref["config"]["reliability_fail_max"])
    budget = float(ref["config"]["energy_budget"])

    hidden = zarr.open(f"{args.reference_dir}/hidden", mode="r")
    packs = sorted(hidden.keys())
    if not packs:
        result["error"] = "no hidden packs"; print(json.dumps(result)); return
    U = int(np.asarray(hidden[packs[0]]).shape[2])
    demand = torch.full((U,), demand_v, device=dev)

    ctx = {"archive_root": f"{args.input_dir}/archive", "radio_system": radio,
           "device": dev, "delay": delay}
    try:
        ctrl = load_controller(args.submission, ctx)
    except Exception as e:
        result["error"] = f"controller init failed: {e}"; print(json.dumps(result)); return

    npass = 0
    for si, pk in enumerate(packs):
        H = torch.from_numpy(np.asarray(hidden[pk])).to(dev)
        hist_est, serve_true = observe(H, noise_var, delay, seed=10_000 + si)
        # EVALUATOR-CONTROLLED combiner: fixed rule on the permitted observations
        # (MMSE on the LS estimate of the most recent OBSERVED frame, nominal
        # power). The candidate cannot influence it -> no combiner co-design.
        g = mmse_combiner(hist_est[:, -1], noise_var, nominal)
        try:
            ret = ctrl.allocate(hist_est, demand)            # candidate: obs in, POWERS out
            powers = ret[1] if isinstance(ret, (tuple, list)) else ret
            powers = torch.clamp(powers.to(dev).float(), min=0.0)
            assert powers.shape == (H.shape[0], U, serve_true.shape[2])
        except Exception as e:
            result["error"] = f"allocate failed on {pk}: {e}"; print(json.dumps(result)); return
        sinr = true_sinr(g, serve_true, powers, noise_var)
        rate = torch.log2(1.0 + sinr).sum(-1)
        fail = (rate < demand).float().mean(0).max().item()
        energy = powers.sum(dim=(1, 2)).mean().item()
        ok = fail <= rel and energy <= budget
        npass += int(ok)
        result["packs"].append({"pack": pk, "worst_user_fail": fail,
                                "energy": energy, "pass": ok})
    result["normalized_score"] = npass / len(packs)
    result["passed"] = result["normalized_score"] >= 1.0
    result["contract"] = {"reliability_fail_max": rel, "energy_budget": budget}
    print(json.dumps(result))


if __name__ == "__main__":
    main()

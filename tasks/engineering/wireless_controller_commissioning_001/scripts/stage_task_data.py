#!/usr/bin/env python3
"""Assemble the ALE task-data tree from the constructed archive + artifacts.

Produces  <out>/engineering/wireless_controller_commissioning_001/base/
    input/       (agent-visible)  radio_system.json, task_brief.md, archive/,
                                  public_eval/, starter/
    reference/   (evaluator-only) hidden/, reference_metrics.json

Run on the box where the archive lives:
    python3 stage_task_data.py --archive ~/wcc_archive --build-out <gen_reference out/> \
                               --out ~/wcc_task_data
The framework then bakes/stages input/ before the agent runs and reference/ at
eval time (see main.py). Large arrays are copied as Zarr.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import zarr

BRIEF = """# Wireless Controller Commissioning

You are commissioning a replacement uplink power controller for a 4-user,
8-antenna MIMO base station over 24 OFDM tones (3GPP TR 38.901 UMi street canyon).

## The problem
At each decision the latest channel estimate is `delay` frames stale and the
channel evolves with dense-urban Doppler. You must act on the (unobserved) serve
frame using only observed history. Deliver a controller that meets a per-user
reliability contract at minimum transmit energy.

## Read first
- `radio_system.json`  full physical spec: array geometry, 38.901 parameters,
  the evaluator-fixed MMSE combiner rule, SINR/rate definitions, units, the
  observation-to-action delay, per-user demand, the declared operating envelope,
  and the acceptance contract.

## Data (read-only, under input/)
- `archive/`  Zarr store with `train`, `dev` (nominal speeds 1-5 m/s) and
  `envelope` (the full declared envelope 1-6 m/s) true-channel trajectories,
  shape (N, T, U, F, M), T = 8 history frames + 1 serve frame.
- `public_eval/`  dev replay, numerical checks, and the Sionna coded-link BLER
  curve (`coded_link_curve.npz`) relating post-combining SINR to decoding.
- `starter/`  a conservative baseline controller and the Controller interface.

## Deliverable  (output/controller.py)
    class Controller:
        def __init__(self, ctx):
            # ctx = {"archive_root", "radio_system", "device", "delay"}
            # Fit predictor / calibrate uncertainty offline. No hidden access.
        def allocate(self, hist_est, demand):
            # hist_est: torch complex (B, L, U, F, M) delayed pilot estimates
            # demand:   torch (U,) bits/frame
            # returns powers (B, U, F) real >= 0        # POWERS ONLY

## The receiver combiner is EVALUATOR-CONTROLLED
You do NOT choose the combiner. The evaluator builds a per-tone MMSE combiner from
the LS estimate of the most recent OBSERVED frame (hist_est[:, -1]) at nominal
power 1, and scores true-channel SINR with THAT combiner and YOUR powers. The
combiner is stale (delay frames old) and imperfect; your power allocation must be
robust to that and to the channel evolution. You may recompute the exact combiner
locally (see starter) to plan, but you cannot change what the evaluator uses.

## How you are scored
An independent evaluator replays your controller on withheld packs from a FASTER
part of the envelope than dev, applies the fixed combiner above, and a pack passes
iff worst per-user service-failure <= 1% AND mean energy <= the budget. Score =
fraction of packs passing. Naive approaches (persistence prediction, ignoring
prediction uncertainty, calibrating uncertainty on nominal dev only, flat/max
power) each fail a gate. Raising power blindly increases mutual interference and
reduces reliability.
"""

STARTER = '''"""Conservative baseline controller (starter). Returns a flat power on every
(user, tone). It DOES NOT meet the contract; improve it: predict the delayed
channel, calibrate joint uncertainty over the envelope, and choose energy-optimal
robust powers. The COMBINER IS EVALUATOR-CONTROLLED: the evaluator builds a
per-tone MMSE combiner from the LS estimate of the most recent OBSERVED frame
(hist_est[:, -1]) at nominal power 1 -- you only return powers. You may rebuild
that same combiner locally (below) to compute SINR coefficients for allocation."""
import torch


class Controller:
    def __init__(self, ctx):
        self.dev = ctx["device"]
        self.noise_var = 2e-3
        self.nominal = 1.0

    def evaluator_combiner(self, hist_est):
        # The exact rule the evaluator uses; provided so you can plan allocation.
        Hhat = hist_est[:, -1]
        b, U, F, M = Hhat.shape
        Rn = self.noise_var * torch.eye(M, dtype=torch.complex64, device=Hhat.device)
        Hf = Hhat.permute(0, 2, 3, 1)
        Ry = self.nominal * (Hf @ Hf.conj().transpose(-1, -2)) + Rn
        return torch.linalg.solve(Ry, Hf).permute(0, 3, 1, 2)

    def allocate(self, hist_est, demand):
        B, L, U, F, M = hist_est.shape
        powers = torch.full((B, U, F), 4.0, device=hist_est.device)   # flat, wasteful
        return powers                          # POWERS ONLY (B, U, F)
'''

PUBLIC_EVAL = '''"""Dev feedback helper: score a controller on a chunk of the dev archive so you
can iterate before submitting. Mirrors the evaluator's scoring. Usage:
    from public_eval.dev_eval import evaluate_on_dev
    evaluate_on_dev(MyController, ctx, n=4096)
"""
import math, numpy as np, torch, zarr


def _observe(H, noise_var, delay, seed=0):
    g = torch.Generator(device=H.device).manual_seed(seed)
    n = ((torch.randn(H.shape, generator=g, device=H.device)
          + 1j*torch.randn(H.shape, generator=g, device=H.device))/math.sqrt(2)).to(torch.complex64)
    Hhat = H + n*math.sqrt(noise_var)
    lo = H.shape[1]-1-delay
    return Hhat[:, :lo+1], H[:, -1]


def _combiner(Hh, nv, p0=1.0):
    b,U,F,M = Hh.shape
    Rn = nv*torch.eye(M, dtype=torch.complex64, device=Hh.device)
    Hf = Hh.permute(0,2,3,1); Ry = p0*(Hf@Hf.conj().transpose(-1,-2))+Rn
    return torch.linalg.solve(Ry, Hf).permute(0,3,1,2)


def evaluate_on_dev(Controller, ctx, n=4096, demand_bits=14.0, noise_var=2e-3):
    dev = ctx["device"]; store = zarr.open(ctx["archive_root"], mode="r")
    H = torch.from_numpy(np.asarray(store["dev"])[:n]).to(dev)
    U = H.shape[2]; demand = torch.full((U,), demand_bits, device=dev)
    ctrl = Controller(ctx)
    hist, serve = _observe(H, noise_var, ctx["delay"], seed=0)
    pred, powers = ctrl.allocate(hist, demand)
    g = _combiner(pred, noise_var)
    a = (torch.einsum("bufm,bvfm->buvf", g.conj(), serve).abs()**2)
    diag = torch.diagonal(a, dim1=1, dim2=2).permute(0,2,1)
    sinr = diag*powers/((torch.einsum("buvf,bvf->buf", a, powers)-diag*powers)
                        + noise_var*(g.abs()**2).sum(-1) + 1e-12)
    rate = torch.log2(1+sinr).sum(-1)
    fail = (rate < demand).float().mean(0)
    print("per-user failure:", [round(x,4) for x in fail.tolist()],
          "mean energy:", round(powers.sum((1,2)).mean().item(),2))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.expanduser("~/wcc_archive"))
    ap.add_argument("--build-out", default="out", help="gen_reference out/ dir")
    ap.add_argument("--radio", default="radio_system.json")
    ap.add_argument("--out", default=os.path.expanduser("~/wcc_task_data"))
    ap.add_argument("--password", default=os.environ.get("ALE_REFERENCE_ARCHIVE_PASSWORD", "ale-wcc"),
                    help="password for reference.7z (matches ALE_REFERENCE_ARCHIVE_PASSWORD at eval)")
    ap.add_argument("--plain-reference", action="store_true",
                    help="also keep a plain reference/ dir (for local debugging)")
    args = ap.parse_args()

    base = f"{args.out}/engineering/wireless_controller_commissioning_001/base"
    inp, ref = f"{base}/input", f"{base}/reference"
    for d in (f"{inp}/archive", f"{inp}/public_eval", f"{inp}/starter", f"{ref}/hidden"):
        os.makedirs(d, exist_ok=True)

    src = zarr.open(args.archive, mode="r")
    dst = zarr.open(f"{inp}/archive", mode="w")
    for split in ("train", "dev", "envelope"):
        a = np.asarray(src[split])
        dst.create_dataset(split, data=a, chunks=(512,)+a.shape[1:], overwrite=True)
    rgrp = zarr.open(f"{ref}/hidden", mode="w")
    for pk in sorted(src["hidden"].keys()):
        a = np.asarray(src["hidden"][pk])
        rgrp.create_dataset(pk, data=a, chunks=(512,)+a.shape[1:], overwrite=True)

    open(f"{inp}/task_brief.md", "w").write(BRIEF)
    open(f"{inp}/starter/controller.py", "w").write(STARTER)
    open(f"{inp}/public_eval/dev_eval.py", "w").write(PUBLIC_EVAL)
    if os.path.exists(args.radio):
        shutil.copy(args.radio, f"{inp}/radio_system.json")
    if os.path.exists(f"{args.build_out}/coded_link_curve.npz"):
        shutil.copy(f"{args.build_out}/coded_link_curve.npz", f"{inp}/public_eval/")
    if os.path.exists(f"{args.build_out}/reference_metrics.json"):
        shutil.copy(f"{args.build_out}/reference_metrics.json", f"{ref}/reference_metrics.json")
    # bundle the self-contained reference solution with the reference (evaluator-only).
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("reference_controller.py", "second_expert.py"):
        if os.path.exists(f"{here}/{f}"):
            shutil.copy(f"{here}/{f}", f"{ref}/{f}")

    # Canonical baked_in_sandbox layout: input/ + password-encrypted reference.7z
    # (flat: reference files at archive root, decrypted into reference/ at eval).
    archive = f"{base}/reference.7z"
    if os.path.exists(archive):
        os.remove(archive)
    subprocess.run(["7z", "a", f"-p{args.password}", "-mhe=on", archive] +
                   [f"{ref}/{n}" for n in os.listdir(ref)],
                   check=True, stdout=subprocess.DEVNULL)
    # Keep BOTH layouts: plain reference/ for the local: (docker) provider
    # (hidden by timing), encrypted reference.7z for baked_in_sandbox / gs:// / s3://.
    print(f"staged task data at {base}")
    print("  input/:", sorted(os.listdir(inp)))
    print("  reference/ (plain, for local:):", sorted(os.listdir(ref)))
    print(f"  reference.7z (encrypted, for baked/bucket): {os.path.getsize(archive)/1e6:.1f} MB, "
          f"password env ALE_REFERENCE_ARCHIVE_PASSWORD")
    print("\ntask_data_source options (task_card / environment yaml):")
    print(f"  local:   local:{args.out}   (docker/local runs)")
    print(f"  baked:   baked_in_sandbox   (bake {args.out} into the image)")
    print(f"  bucket:  gs://<bucket>/... | s3://<bucket>/...  (upload {args.out})")


if __name__ == "__main__":
    main()

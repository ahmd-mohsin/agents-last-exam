"""Self-contained reference (expert) controller for
engineering/wireless_controller_commissioning_001. Evaluator-only.

No external task modules: the predictor, the envelope-calibrated scenario-robust
energy-optimal allocator, and the MMSE combiner are all embedded here. Reproduces
the reference result (coverage 1.0). This is the artifact shipped for the ALE
quality check; it must load and run against the staged input/ archive alone.

Method: complex LMMSE predictor fit on train history->serve pairs; uncertainty
scenarios bootstrap-sampled from ENVELOPE residuals (robust to the deployment
shift); powers by projected-Adam minimization of total power s.t. worst-case
per-scenario sum-rate >= demand.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import zarr

NOISE_VAR = 2e-3
NOMINAL = 1.0
SCEN = 48


def _observe(H, delay, seed):
    g = torch.Generator(device=H.device).manual_seed(seed)
    n = ((torch.randn(H.shape, generator=g, device=H.device)
          + 1j * torch.randn(H.shape, generator=g, device=H.device)) / math.sqrt(2)).to(torch.complex64)
    Hhat = H + n * math.sqrt(NOISE_VAR)
    lo = H.shape[1] - 1 - delay
    return Hhat[:, : lo + 1], H[:, -1]


def _fit_lmmse(hist, serve):
    B, L, U, F, M = hist.shape
    x = hist.permute(0, 2, 3, 4, 1).reshape(-1, L)
    y = serve.reshape(-1)
    R = (x.conj().unsqueeze(-1) * x.unsqueeze(-2)).mean(0)
    b = (x.conj() * y.unsqueeze(-1)).mean(0)
    R = R + 1e-6 * torch.eye(L, dtype=R.dtype, device=R.device)
    return torch.linalg.solve(R, b)


def _predict(hist, c):
    return torch.einsum("blufm,l->bufm", hist, c)


def _combiner(Hh):
    b, U, F, M = Hh.shape
    Rn = NOISE_VAR * torch.eye(M, dtype=torch.complex64, device=Hh.device)
    Hf = Hh.permute(0, 2, 3, 1)
    Ry = NOMINAL * (Hf @ Hf.conj().transpose(-1, -2)) + Rn
    return torch.linalg.solve(Ry, Hf).permute(0, 3, 1, 2)


def _energy_optimal_robust(g, scen, demand, p_max=200.0, iters=400, lr=0.15, lam=50.0):
    B, S, U, F, M = scen.shape
    a = (torch.einsum("bufm,bsvfm->bsuvf", g.conj(), scen).abs() ** 2)
    diag = torch.diagonal(a, dim1=2, dim2=3).permute(0, 1, 3, 2)
    noise = (NOISE_VAR * (g.abs() ** 2).sum(-1))
    z = torch.zeros(B, U, F, device=scen.device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        p = torch.sigmoid(z) * p_max
        sig = diag * p[:, None]
        interf = torch.einsum("bsuvf,bvf->bsuf", a, p) - sig
        sinr = sig / (interf + noise[:, None] + 1e-12)
        rate = torch.log2(1 + sinr.min(dim=1).values).sum(-1)
        loss = p.sum((1, 2)) + lam * (torch.clamp(demand - rate, min=0.0) ** 2).sum(-1)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        return (torch.sigmoid(z) * p_max).detach()


class Controller:
    def __init__(self, ctx):
        self.dev = ctx["device"]
        self.delay = ctx["delay"]
        store = zarr.open(ctx["archive_root"], mode="r")
        Htr = torch.from_numpy(np.asarray(store["train"])).to(self.dev)
        Henv = torch.from_numpy(np.asarray(store["envelope"])).to(self.dev)
        htr, str_ = _observe(Htr, self.delay, seed=1)
        henv, senv = _observe(Henv, self.delay, seed=2)
        self.c = _fit_lmmse(htr, str_)
        self.pool = senv - _predict(henv, self.c)          # envelope residuals

    def allocate(self, hist_est, demand):
        pred = _predict(hist_est, self.c)
        g = _combiner(hist_est[:, -1])          # evaluator-fixed combiner (last observed frame)
        B = pred.shape[0]
        idx = torch.randint(0, self.pool.shape[0], (B, SCEN), device=self.dev)
        scen = pred[:, None] + self.pool[idx]
        dm = demand[None].expand(B, demand.shape[0]).contiguous()
        powers = _energy_optimal_robust(g, scen, dm)
        return pred, powers

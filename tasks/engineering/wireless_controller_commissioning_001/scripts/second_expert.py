"""Independent SECOND-EXPERT controller (ALE quality-check reconstruction).

Built from the task brief alone, deliberately using a DIFFERENT method from the
reference: instead of bootstrap uncertainty scenarios, it uses an ANALYTIC
per-(user,tone) prediction-error-variance model (a second-moment / Gaussian
back-off) inside the energy-optimal allocation. If this also meets the contract,
the task is feasible by more than one route -- i.e. not an author-head artifact.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import zarr

NOISE_VAR = 2e-3
NOMINAL = 1.0


def _observe(H, delay, seed):
    g = torch.Generator(device=H.device).manual_seed(seed)
    n = ((torch.randn(H.shape, generator=g, device=H.device)
          + 1j * torch.randn(H.shape, generator=g, device=H.device)) / math.sqrt(2)).to(torch.complex64)
    return (H + n * math.sqrt(NOISE_VAR))[:, : H.shape[1] - delay], H[:, -1]


def _fit_lmmse(hist, serve):
    B, L, U, F, M = hist.shape
    x = hist.permute(0, 2, 3, 4, 1).reshape(-1, L); y = serve.reshape(-1)
    R = (x.conj().unsqueeze(-1) * x.unsqueeze(-2)).mean(0) + 1e-6 * torch.eye(L, dtype=torch.complex64, device=x.device)
    return torch.linalg.solve(R, (x.conj() * y.unsqueeze(-1)).mean(0))


def _predict(hist, c):
    return torch.einsum("blufm,l->bufm", hist, c)


def _combiner(Hh):
    b, U, F, M = Hh.shape
    Rn = NOISE_VAR * torch.eye(M, dtype=torch.complex64, device=Hh.device)
    Hf = Hh.permute(0, 2, 3, 1)
    return torch.linalg.solve(NOMINAL * (Hf @ Hf.conj().transpose(-1, -2)) + Rn, Hf).permute(0, 3, 1, 2)


class Controller:
    def __init__(self, ctx):
        self.dev = ctx["device"]; self.delay = ctx["delay"]
        store = zarr.open(ctx["archive_root"], mode="r")
        Htr = torch.from_numpy(np.asarray(store["train"])).to(self.dev)
        Henv = torch.from_numpy(np.asarray(store["envelope"])).to(self.dev)
        htr, str_ = _observe(Htr, self.delay, 1)
        henv, senv = _observe(Henv, self.delay, 2)
        self.c = _fit_lmmse(htr, str_)
        e = senv - _predict(henv, self.c)                 # envelope residuals (N,U,F,M)
        # analytic per-(user,tone) error power, with a safety scale for the tail.
        self.eps = 3.0 * (e.abs() ** 2).mean(dim=(0, 3))  # (U,F)

    def allocate(self, hist_est, demand, iters=400, lr=0.15, lam=50.0, p_max=200.0):
        pred = _predict(hist_est, self.c)
        g = _combiner(hist_est[:, -1])          # evaluator-fixed combiner (last observed frame)
        B, U, F, M = pred.shape
        a = (torch.einsum("bufm,bvfm->buvf", g.conj(), pred).abs() ** 2)   # pred coeffs
        gn = (g.abs() ** 2).sum(-1)                                        # (B,U,F)
        diag = torch.diagonal(a, dim1=1, dim2=2).permute(0, 2, 1)          # (B,U,F)
        eps = self.eps[None]                                              # (1,U,F)
        z = torch.zeros(B, U, F, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            p = torch.sigmoid(z) * p_max
            sig = diag * p                                                # signal (pred)
            # interference: cross pred terms + interferers' error energy (isotropic
            # gn*eps_v) + own error energy; all analytic (no scenarios).
            cross = torch.einsum("buvf,bvf->buf", a, p) - sig
            s = torch.einsum("bvf,vf->bf", p, self.eps)                  # sum_v p_v eps_vf (B,F)
            unc = gn * s[:, None, :]                                     # * ||g_u||^2
            interf = cross + unc + NOISE_VAR * gn
            rate = torch.log2(1 + sig / (interf + 1e-12)).sum(-1)
            loss = p.sum((1, 2)) + lam * (torch.clamp(demand[None] - rate, min=0.0) ** 2).sum(-1)
            loss.sum().backward(); opt.step()
        with torch.no_grad():
            return pred, (torch.sigmoid(z) * p_max).detach()

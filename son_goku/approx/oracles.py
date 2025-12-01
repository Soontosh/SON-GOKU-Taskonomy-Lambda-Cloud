# son_goku/approx/oracles.py
from __future__ import annotations
from typing import Optional, Tuple
import time
import math
import numpy as np
import torch
import torch.nn.functional as F


def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [K, d]
    return x / (x.norm(dim=1, keepdim=True) + eps)


class CosineBuildResult:
    def __init__(self, cos: torch.Tensor, timings: dict):
        # cos: [K, K] (upper triangle used), symmetric with diag=1
        self.cos = cos
        self.timings = timings  # {"embed": ms, "pairs": ms, "total": ms}


class BaseCosineOracle:
    def build(self, E: torch.Tensor) -> CosineBuildResult:
        raise NotImplementedError()


class ExactOracle(BaseCosineOracle):
    """Full KxK cosine from EMA gradient matrix E∈R^{Kxd}."""
    def build(self, E: torch.Tensor) -> CosineBuildResult:
        t0 = time.time()
        # Normalize and batch matmul
        t_embed0 = time.time()
        U = _normalize_rows(E)
        t_embed = (time.time() - t_embed0) * 1000.0
        t_pairs0 = time.time()
        C = torch.clamp(U @ U.t(), -1.0, 1.0)
        # Ensure symmetry & diag
        K = C.size(0)
        C.fill_diagonal_(1.0)
        t_pairs = (time.time() - t_pairs0) * 1000.0
        t_total = (time.time() - t0) * 1000.0
        return CosineBuildResult(C, {"embed": t_embed, "pairs": t_pairs, "total": t_total})


class RandomProjectionOracle(BaseCosineOracle):
    """JL projection: E -> E @ P, with P∈R^{d x r} i.i.d. N(0,1/√r)."""
    def __init__(self, d: int, r: int, device: torch.device):
        self.P = torch.randn(d, r, device=device) / math.sqrt(r)

    def build(self, E: torch.Tensor) -> CosineBuildResult:
        t0 = time.time()
        t_embed0 = time.time()
        Z = E @ self.P   # [K, r]
        U = _normalize_rows(Z)
        t_embed = (time.time() - t_embed0) * 1000.0
        t_pairs0 = time.time()
        C = torch.clamp(U @ U.t(), -1.0, 1.0)
        C.fill_diagonal_(1.0)
        t_pairs = (time.time() - t_pairs0) * 1000.0
        t_total = (time.time() - t0) * 1000.0
        return CosineBuildResult(C, {"embed": t_embed, "pairs": t_pairs, "total": t_total})


class _FrequentDirections:
    """
    Minimal Frequent Directions sketcher for covariance approximation.

    Maintains B∈R^{ℓ x d}. After streaming rows of E, we SVD B to obtain
    a right subspace W∈R^{d x ℓ}, and use Z = E W to compute cosines in ℓ-dim.
    """
    def __init__(self, d: int, ell: int, device: torch.device):
        self.ell = ell
        cpu_device = torch.device("cpu")
        self.B = torch.zeros(ell, d, device=cpu_device)

    @torch.no_grad()
    def push_rows(self, A: torch.Tensor):
        # A: [n, d], stream rows into FD sketch
        for i in range(A.size(0)):
            a = A[i]  # [d]
            # Try to insert into a zero row
            zero_rows = torch.where(self.B.norm(dim=1) == 0)[0]
            if len(zero_rows) > 0:
                self.B[zero_rows[0]] = a
                continue
            # Full: SVD and shrink
            U, S, Vh = torch.linalg.svd(self.B, full_matrices=False)
            delta = S[-1].pow(2)
            S_shrunk = torch.sqrt(torch.clamp(S.pow(2) - delta, min=0.0))
            self.B = (U @ torch.diag(S_shrunk)) @ Vh
            # insert a
            zero_rows = torch.where(self.B.norm(dim=1) == 0)[0]
            if len(zero_rows) > 0:
                self.B[zero_rows[0]] = a
            else:
                # In rare degeneracy, overwrite the smallest row
                idx = torch.argmin(self.B.norm(dim=1))
                self.B[idx] = a

    @torch.no_grad()
    def projection(self) -> torch.Tensor:
        # Return W∈R^{d x ℓ} (top right singular vectors of B)
        if (self.B == 0).all():
            raise RuntimeError("FD sketch is empty")
        U, S, Vh = torch.linalg.svd(self.B, full_matrices=False)
        # Vh: [ℓ, d]; return W = V (d x ℓ)
        return Vh.t()


class FrequentDirectionsOracle(BaseCosineOracle):
    def __init__(self, d: int, ell: int, device: torch.device):
        self.d = d
        self.ell = ell
        self.device = device

    def build(self, E: torch.Tensor) -> CosineBuildResult:
        t0 = time.time()
        t_embed0 = time.time()
        E_cpu = E.cpu()
        fd = _FrequentDirections(E_cpu.size(1), self.ell, torch.device("cpu"))
        fd.push_rows(E_cpu)                 # stream
        W = fd.projection().to(E.device)    # [d, ell]
        Z = E @ W                           # [K, ell]
        U = _normalize_rows(Z)
        t_embed = (time.time() - t_embed0) * 1000.0
        t_pairs0 = time.time()
        C = torch.clamp(U @ U.t(), -1.0, 1.0)
        C.fill_diagonal_(1.0)
        t_pairs = (time.time() - t_pairs0) * 1000.0
        t_total = (time.time() - t0) * 1000.0
        return CosineBuildResult(C, {"embed": t_embed, "pairs": t_pairs, "total": t_total})


class EdgeSamplingOracle(BaseCosineOracle):
    """
    Wrap another oracle and compute only a fraction p of pairwise cosines.
    Uncomputed pairs default to 0 (no edge if τ>0). This accelerates graph build.
    """
    def __init__(self, base: BaseCosineOracle, p: float, gen: Optional[torch.Generator] = None):
        assert 0 < p <= 1
        self.base = base
        self.p = p
        self.gen = gen

    def build(self, E: torch.Tensor) -> CosineBuildResult:
        t0 = time.time()
        # Get normalized embeddings from base but *not* the full C if possible.
        # We call base.build to reuse its timing; then sparsify C by sampling.
        res = self.base.build(E)
        C = res.cos.clone()
        K = C.size(0)
        # Sample upper triangle
        t_pairs0 = time.time()
        device = C.device
        mask = torch.zeros_like(C, dtype=torch.bool, device=device)
        iu, ju = torch.triu_indices(K, K, offset=1)
        iu = iu.to(device)
        ju = ju.to(device)
        num_pairs = iu.numel()
        keep = torch.rand(num_pairs, generator=self.gen, device=device) < self.p
        mask[iu[keep], ju[keep]] = True
        mask = mask | mask.t() | torch.eye(K, dtype=torch.bool, device=device)
        C = torch.where(mask, C, torch.zeros_like(C))
        t_pairs = (time.time() - t_pairs0) * 1000.0
        t_total = (time.time() - t0) * 1000.0
        timings = dict(res.timings)
        timings["pairs"] = t_pairs
        timings["total"] = t_total
        return CosineBuildResult(C, timings)


class IncrementalGramOracle(BaseCosineOracle):
    """
    Reuse previous cosine rows/cols if EMA vectors changed less than epsilon.
    Wraps a base oracle that computes rows we decide to refresh.
    """
    def __init__(self, base: BaseCosineOracle, epsilon: float):
        self.base = base
        self.epsilon = epsilon
        self.prev_E: Optional[torch.Tensor] = None
        self.prev_C: Optional[torch.Tensor] = None

    @torch.no_grad()
    def build(self, E: torch.Tensor) -> CosineBuildResult:
        t0 = time.time()
        K, d = E.shape
        # First time => compute full
        if self.prev_E is None or self.prev_C is None:
            res = self.base.build(E)
            self.prev_E = E.detach().clone()
            self.prev_C = res.cos.detach().clone()
            return res

        # Determine which rows changed enough
        t_embed0 = time.time()
        diff = (E - self.prev_E).norm(dim=1)  # [K]
        active = (diff >= self.epsilon).nonzero(as_tuple=False).view(-1)
        t_embed = (time.time() - t_embed0) * 1000.0

        C = self.prev_C.clone()
        t_pairs0 = time.time()
        if active.numel() > 0:
            # Build full from base, but we'll only copy active rows/cols
            res = self.base.build(E)
            C[active, :] = res.cos[active, :]
            C[:, active] = res.cos[:, active]
            timings = dict(res.timings)
            timings["embed"] += t_embed  # include our detection
            timings["pairs"] = (time.time() - t_pairs0) * 1000.0
            timings["total"] = (time.time() - t0) * 1000.0
        else:
            timings = {"embed": t_embed, "pairs": 0.0, "total": (time.time() - t0) * 1000.0}

        self.prev_E = E.detach().clone()
        self.prev_C = C.detach().clone()
        return CosineBuildResult(C, timings)

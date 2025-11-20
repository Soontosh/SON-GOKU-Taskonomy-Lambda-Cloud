# son_goku/affinity/affinity.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import time
import numpy as np
import torch
from torch import nn
from son_goku import TaskSpec


def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    # simple, tie-naive Spearman (rank Pearson)
    def _rank(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(v), dtype=torch.float32, device=v.device)
        return ranks
    xr, yr = _rank(x), _rank(y)
    xm = xr - xr.mean()
    ym = yr - yr.mean()
    num = (xm * ym).sum()
    den = torch.sqrt((xm * xm).sum() * (ym * ym).sum()) + 1e-8
    return float((num / den).item())


def _upper_tri_vec(M: torch.Tensor) -> torch.Tensor:
    iu, ju = torch.triu_indices(M.size(0), M.size(1), offset=1)
    return M[iu, ju]


def _flatten_params(params: List[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params])


def _set_flat_params(params: List[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[offset:offset+n].view_as(p))
        offset += n


@dataclass
class AffinityResult:
    # Symmetric matrix S[K, K], larger = "more conflict" by the chosen backend
    S: torch.Tensor
    timings_ms: Dict[str, float]
    # Optional diagnostics (e.g., correlation vs another S')
    aux: Dict[str, float]


class AffinityComputer:
    """Interface."""
    def compute(
        self,
        model: nn.Module,
        task_specs: Sequence[TaskSpec],
        batch: Dict[str, Any],
        device: torch.device,
        shared_only: bool = True,
        pair_fraction: float = 1.0,
        eta_v: float = 0.0,
    ) -> AffinityResult:
        raise NotImplementedError


class GradCosineAffinity(AffinityComputer):
    """Builds cosine on per-task gradients (shared params by default)."""
    def __init__(self, shared_param_filter):
        self.shared_param_filter = shared_param_filter

    def compute(
        self, model, task_specs, batch, device,
        shared_only: bool = True, pair_fraction: float = 1.0, eta_v: float = 0.0
    ) -> AffinityResult:
        t0 = time.time()
        # pick params
        params = [p for p in model.parameters() if self.shared_param_filter(p)] \
                 or [p for p in model.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)
        K = len(task_specs)

        G = torch.zeros(K, P, device=device)
        for i, spec in enumerate(task_specs):
            model.zero_grad(set_to_none=True)
            loss = spec.loss_fn(model, batch)
            loss.backward()
            # flatten grads on chosen params
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
            G[i] = g.detach()

        # cosine
        U = G / (G.norm(dim=1, keepdim=True) + 1e-8)
        S = torch.clamp(U @ U.t(), -1.0, 1.0)
        S.fill_diagonal_(1.0)
        return AffinityResult(S=S, timings_ms={"total": (time.time()-t0)*1e3}, aux={})


class TAGExactAffinity(AffinityComputer):
    """
    TAG lookahead: D_{j|i} = L_j(theta - eta_v * g_i) - L_j(theta), g_i = grad_i(shared).
    We return the symmetrized matrix S_ij = 0.5*(D_{j|i}+D_{i|j}).
    """
    def __init__(self, shared_param_filter, symmetrize: bool = True):
        self.shared_param_filter = shared_param_filter
        self.symmetrize = symmetrize

    @torch.no_grad()
    def compute(
        self, model, task_specs, batch, device,
        shared_only: bool = True, pair_fraction: float = 1.0, eta_v: float = 1e-3
    ) -> AffinityResult:
        t0 = time.time()
        # params we update virtually
        params = [p for p in model.parameters() if self.shared_param_filter(p)] \
                 or [p for p in model.parameters() if p.requires_grad]
        # store flat weights to restore quickly
        w0 = _flatten_params(params).clone()
        K = len(task_specs)

        # base L_j(theta)
        base = []
        for spec in task_specs:
            base.append(float(spec.loss_fn(model, batch).item()))
        base = torch.tensor(base, device=device, dtype=torch.float32)

        # per-task gradient g_i (ONLY for shared params)
        G = []
        for i, spec in enumerate(task_specs):
            for p in model.parameters():
                p.grad = None
            loss = spec.loss_fn(model, batch)
            loss.backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
            G.append(g.detach().clone())
        G = torch.stack(G, dim=0)  # [K, P]

        # sample pairs (upper triangle)
        iu, ju = torch.triu_indices(K, K, offset=1)
        if pair_fraction < 1.0:
            keep = torch.rand(iu.numel(), device=device) < pair_fraction
            iu, ju = iu[keep], ju[keep]

        # directional deltas we compute (sparse)
        D = torch.zeros(K, K, device=device)

        # compute D_{j|i} for sampled pairs
        for idx in range(iu.numel()):
            i, j = int(iu[idx]), int(ju[idx])

            # step along -eta*g_i
            step = -eta_v * G[i]
            _set_flat_params(params, w0 + step)
            Lj_theta_step = float(task_specs[j].loss_fn(model, batch).item())
            D[j, i] = Lj_theta_step - base[j]

            # and the reverse direction if symmetrize
            if self.symmetrize:
                step_rev = -eta_v * G[j]
                _set_flat_params(params, w0 + step_rev)
                Li_theta_step = float(task_specs[i].loss_fn(model, batch).item())
                D[i, j] = Li_theta_step - base[i]

            # restore
            _set_flat_params(params, w0)

        # symmetrize
        S = 0.5 * (D + D.t())
        S.fill_diagonal_(0.0)  # by definition
        return AffinityResult(S=S, timings_ms={"total": (time.time()-t0)*1e3}, aux={})


class TAGLinearizedAffinity(AffinityComputer):
    """
    First-order TAG: \tilde D_{j|i} = -eta_v * <g_j, g_i> (shared grads).
    Symmetric result S_ij = \tilde D_{j|i} == \tilde D_{i|j}.
    """
    def __init__(self, shared_param_filter, cosine_normalize: bool = False):
        self.shared_param_filter = shared_param_filter
        self.cosine_normalize = cosine_normalize

    def compute(
        self, model, task_specs, batch, device,
        shared_only: bool = True, pair_fraction: float = 1.0, eta_v: float = 1e-3
    ) -> AffinityResult:
        t0 = time.time()
        params = [p for p in model.parameters() if self.shared_param_filter(p)] \
                 or [p for p in model.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)
        K = len(task_specs)

        G = torch.zeros(K, P, device=device)
        for i, spec in enumerate(task_specs):
            model.zero_grad(set_to_none=True)
            loss = spec.loss_fn(model, batch)
            loss.backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
            G[i] = g.detach()

        if self.cosine_normalize:
            U = G / (G.norm(dim=1, keepdim=True) + 1e-8)
            Dlin = -eta_v * (U @ U.t())
        else:
            Dlin = -eta_v * (G @ G.t())

        S = Dlin
        S.fill_diagonal_(0.0)
        return AffinityResult(S=S, timings_ms={"total": (time.time()-t0)*1e3}, aux={})


def cosine_vs_tag_stats(C: torch.Tensor, Dsym_neg: torch.Tensor) -> Dict[str, float]:
    """
    Compare cosine C (KxK) with negative lookahead -D (KxK): Spearman + sign agreement.
    """
    x = _upper_tri_vec(C).detach()
    y = _upper_tri_vec(Dsym_neg).detach()
    # normalize to zero-mean for stability
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    spearman = _spearmanr(x, y)
    sign_agree = float(((x * y) >= 0).float().mean().item())
    return {"spearman": spearman, "sign_agree": sign_agree}
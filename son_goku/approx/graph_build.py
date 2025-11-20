# son_goku/approx/graph_build.py
from __future__ import annotations
import math
import torch

def upper_tri_values(C: torch.Tensor) -> torch.Tensor:
    iu, ju = torch.triu_indices(C.size(0), C.size(1), offset=1)
    return C[iu, ju]

def edge_density(A: torch.Tensor) -> float:
    """Undirected density in [0,1], counting edges once."""
    K = A.size(0)
    if K <= 1:
        return 0.0
    iu, ju = torch.triu_indices(K, K, offset=1)
    m = A[iu, ju].sum().item()
    return float(m / (K * (K - 1) / 2.0))

def calibrate_tau_for_density(C: torch.Tensor, target_density: float) -> float:
    """
    Find τ as the percentile p of upper-triangular cos values such that
    density(A) ~= target_density when using A = (C < τ).
    """
    p = max(0.0, min(1.0, float(target_density)))
    vals = upper_tri_values(C)
    if vals.numel() == 0:
        return 1.0
    tau = torch.quantile(vals, p).item()
    return float(max(-1.0, min(1.0, tau)))

def knn_k_for_density(K: int, target_density: float) -> int:
    """Choose k to roughly hit density δ ≈ 2*k / (K-1). (Union of directed kNN graph.)"""
    δ = max(0.0, min(1.0, float(target_density)))
    if K <= 1:
        return 0
    k = int(round(δ * (K - 1) / 2.0))
    return max(1, min(K - 1, k))

def adjacency_from_cos(
    C: torch.Tensor,
    mode: str,
    *,
    tau: float = 0.25,
    knn_k: int = 3,
    quantile_p: float = 0.3,
) -> torch.Tensor:
    """
    Build symmetric, hollow adjacency (bool) from cosine matrix C (KxK).
    Modes:
      - 'threshold' : edge if C_ij < tau
      - 'signed'    : edge if C_ij < 0.0
      - 'quantile'  : edge if C_ij < Quantile_p(upper-tri(C))
      - 'knn'       : union of each node's k smallest-cos neighbors (excluding self)
    """
    assert C.dim() == 2 and C.size(0) == C.size(1), "C must be KxK"
    K = C.size(0)
    A = torch.zeros_like(C, dtype=torch.bool)

    if mode == "threshold":
        A = (C < tau)
    elif mode == "signed":
        A = (C < 0.0)
    elif mode == "quantile":
        p = quantile_p
        if p > 1.0:  # allow 0..100
            p = p / 100.0
        p = max(0.0, min(1.0, float(p)))
        vals = upper_tri_values(C)
        q = torch.quantile(vals, p) if vals.numel() else torch.tensor(1.0, device=C.device, dtype=C.dtype)
        A = (C < q)
    elif mode == "knn":
        # smallest values = most conflicting (lowest cosine)
        order = torch.argsort(C, dim=1, stable=True)  # ascending
        A_local = torch.zeros_like(C, dtype=torch.bool)
        for i in range(K):
            row = order[i]
            row = row[row != i][:knn_k]  # drop self, take k
            if row.numel():
                A_local[i, row] = True
        A = torch.logical_or(A_local, A_local.t())
    else:
        raise ValueError(f"Unknown graph mode: {mode}")

    A.fill_diagonal_(False)
    return torch.logical_or(A, A.t())  # ensure symmetry
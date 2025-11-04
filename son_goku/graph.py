from typing import Dict, List, Optional, Set, Tuple
import torch
from torch import Tensor

def cosine_interference_matrix(emas: List[Optional[Tensor]], eps: float = 1e-12) -> Tensor:
    """
    Compute pairwise interference matrix rho_ij = -cos(g_i, g_j).
    emas: list of length K; each item is a 1-D flattened EMA vector or None (if unavailable).
    Returns:
        (K, K) symmetric matrix with zeros on diagonal; if an EMA is missing, its row/col is 0.
    """
    K = len(emas)
    device = None
    for g in emas:
        if g is not None:
            device = g.device
            dtype = g.dtype
            break
    if device is None:
        # No data at all
        return torch.zeros((K, K))
    # Stack with zeros where missing
    vecs = []
    mask = []
    for g in emas:
        if g is None:
            vecs.append(torch.zeros_like(emas[0]))
            mask.append(0.0)
        else:
            vecs.append(g)
            mask.append(1.0)
    V = torch.stack(vecs, dim=0)  # (K, D)
    norms = (V.norm(dim=1) + eps)  # (K,)
    # Cosine matrix: C_ij = <vi, vj> / (||vi|| ||vj||)
    dots = V @ V.t()
    denom = norms[:, None] * norms[None, :]
    C = dots / denom
    # Interference is negative cosine
    R = -C
    # Zero diagonal
    R.fill_diagonal_(0.0)
    # If either side missing, zero out
    m = torch.tensor(mask, device=device, dtype=V.dtype)
    R = R * (m[:, None] * m[None, :])
    return R

def build_conflict_graph(rho: Tensor, tau: float) -> Dict[int, Set[int]]:
    """
    Build adjacency list where an undirected edge (i,j) exists if rho_ij > tau.
    Args:
        rho: (K, K) interference matrix
        tau: threshold in (0,1)
    Returns:
        dict: i -> set of neighbors j
    """
    assert rho.dim() == 2 and rho.size(0) == rho.size(1), "rho must be square"
    K = rho.size(0)
    adj = {i: set() for i in range(K)}
    # Strictly > tau per paper
    where = (rho > tau).nonzero(as_tuple=False)
    for i, j in where.tolist():
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj

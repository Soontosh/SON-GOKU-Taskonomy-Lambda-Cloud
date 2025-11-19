# son_goku/approx/coloring.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import time
import torch


def build_adjacency(C: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Build symmetric adjacency (bool) where edge(i,j)=True if cosine > tau.
    C: [K, K] cosine matrix (diag=1).
    """
    K = C.size(0)
    A = (C > tau)
    A.fill_diagonal_(False)
    # ensure symmetry
    A = A | A.t()
    return A


def welsh_powell_coloring(A: torch.Tensor, warm_start_colors: Optional[List[int]] = None) -> Tuple[List[int], Dict[str, float]]:
    """
    Simple greedy coloring; optionally warm-start by ordering vertices by their previous colors.

    Returns:
      colors: list of color ids (0..m-1) per vertex
      timings: {"order_ms":..., "assign_ms":..., "total_ms":...}
    """
    t0 = time.time()
    K = A.size(0)
    deg = A.sum(dim=1).cpu().tolist()  # degrees
    nodes = list(range(K))

    if warm_start_colors is not None and len(warm_start_colors) == K:
        # Put nodes grouped by previous colors, and within each group, by degree desc
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(warm_start_colors):
            groups.setdefault(c, []).append(i)
        order = []
        for c in sorted(groups.keys()):
            grp = groups[c]
            grp.sort(key=lambda i: -deg[i])
            order.extend(grp)
    else:
        # standard Welsh–Powell: sort by degree desc
        order = sorted(nodes, key=lambda i: -deg[i])
    t_order = (time.time() - t0) * 1000.0

    # Assign colors greedily
    t1 = time.time()
    colors = [-1] * K
    current_color = 0
    for u in order:
        if colors[u] != -1:
            continue
        colors[u] = current_color
        # Color all non-adjacent remaining nodes with the same color
        for v in order:
            if colors[v] == -1 and not bool(A[u, v].item()):
                # v must not be adjacent to any already colored with current_color
                conflict = False
                for w in range(K):
                    if colors[w] == current_color and bool(A[v, w].item()):
                        conflict = True
                        break
                if not conflict:
                    colors[v] = current_color
        current_color += 1
    t_assign = (time.time() - t1) * 1000.0

    return colors, {"order_ms": t_order, "assign_ms": t_assign, "total_ms": (time.time() - t0) * 1000.0}

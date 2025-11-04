from typing import Dict, List, Set, Tuple

def welsh_powell_coloring(adj: Dict[int, Set[int]]) -> List[List[int]]:
    """
    Welsh–Powell largest-first greedy coloring.
    Args:
        adj: adjacency list mapping vertex -> set(neighbors), vertices are 0..K-1
    Returns:
        List of color classes; each is a list of vertices (ints).
    Guarantees <= Delta + 1 colors in practice for simple graphs.
    """
    # Sort vertices by descending degree
    vertices = sorted(adj.keys(), key=lambda v: len(adj[v]), reverse=True)
    color_of = {}
    colors: List[Set[int]] = []
    for v in vertices:
        # find first color index that is compatible
        assigned = False
        for ci, color_set in enumerate(colors):
            if all((nbr not in color_set) for nbr in adj[v]):
                color_set.add(v)
                color_of[v] = ci
                assigned = True
                break
        if not assigned:
            colors.append({v})
            color_of[v] = len(colors) - 1
    # Return classes as sorted lists for determinism
    return [sorted(list(s)) for s in colors]

def duplicate_min_coverage(
    classes: List[List[int]], 
    adj: Dict[int, Set[int]], 
    min_updates_per_cycle: int
) -> List[List[int]]:
    """
    Ensure each vertex appears at least `min_updates_per_cycle` times across classes by
    duplicating it into any compatible classes.
    This does NOT add new time steps; it augments existing classes.
    """
    from collections import Counter
    counts = Counter()
    for c in classes:
        for v in c:
            counts[v] += 1
    # For each vertex lacking coverage, try to add it to compatible classes
    for v, cnt in counts.items():
        needed = max(1, min_updates_per_cycle) - cnt
        if needed <= 0:
            continue
        # Try each class in order; add if compatible
        for cls in classes:
            if v in cls:
                continue
            # Check compatibility: v must have no conflicts with any member of cls
            if all((u not in adj[v]) for u in cls):
                cls.append(v)
                needed -= 1
                if needed == 0:
                    break
    # Normalize: sort each class
    for cls in classes:
        cls.sort()
    return classes

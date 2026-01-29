"""
Topology Persistence Primitives (70-72)

Persistence diagrams, Betti numbers, persistence entropy.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


def persistence_diagram(
    points: np.ndarray,
    max_dimension: int = 1,
    max_edge_length: float = None
) -> Dict[int, np.ndarray]:
    """
    Compute persistence diagram using Vietoris-Rips filtration.

    Parameters
    ----------
    points : np.ndarray
        Point cloud (n_points x n_dims)
    max_dimension : int
        Maximum homology dimension to compute
    max_edge_length : float, optional
        Maximum edge length in filtration

    Returns
    -------
    dict
        {dimension: array of (birth, death) pairs}

    Notes
    -----
    Persistence diagram encodes topological features:
    - H0: connected components (clusters)
    - H1: loops (1-dimensional holes)
    - H2: voids (2-dimensional holes)

    Long bars (death - birth) = persistent features
    Short bars = noise
    """
    try:
        from ripser import ripser

        points = np.asarray(points)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if max_edge_length is None:
            # Estimate from data
            dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            max_edge_length = np.percentile(dists[dists > 0], 90) if dists.size > 1 else 1.0

        result = ripser(
            points,
            maxdim=max_dimension,
            thresh=max_edge_length
        )

        diagrams = {}
        for dim, dgm in enumerate(result['dgms']):
            # Replace inf with max_edge_length for plotting
            dgm = dgm.copy()
            dgm[np.isinf(dgm)] = max_edge_length
            diagrams[dim] = dgm

        return diagrams

    except ImportError:
        # Fallback: simple H0 computation using union-find
        return _simple_persistence(points, max_edge_length)


def betti_numbers(
    diagrams: Dict[int, np.ndarray],
    threshold: float = None,
    filtration_value: float = None
) -> Dict[int, int]:
    """
    Compute Betti numbers from persistence diagram.

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams from persistence_diagram()
    threshold : float, optional
        Minimum persistence to count as a feature
    filtration_value : float, optional
        Count features alive at this filtration value

    Returns
    -------
    dict
        {dimension: Betti number}

    Notes
    -----
    Betti numbers count topological features:
    - β_0: number of connected components
    - β_1: number of loops
    - β_2: number of voids

    If filtration_value specified: count features where birth ≤ value < death
    If threshold specified: count features where death - birth > threshold
    """
    betti = {}

    for dim, dgm in diagrams.items():
        if len(dgm) == 0:
            betti[dim] = 0
            continue

        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistence = deaths - births

        if filtration_value is not None:
            # Count features alive at filtration_value
            alive = (births <= filtration_value) & (deaths > filtration_value)
            count = np.sum(alive)
        elif threshold is not None:
            # Count features with sufficient persistence
            count = np.sum(persistence > threshold)
        else:
            # Count all features (excluding infinite persistence for H0)
            if dim == 0:
                # For H0, one component survives to infinity (the whole space)
                count = np.sum(np.isfinite(deaths))
            else:
                count = len(dgm)

        betti[dim] = int(count)

    return betti


def persistence_entropy(
    diagrams: Dict[int, np.ndarray],
    dimension: int = 1,
    normalized: bool = True
) -> float:
    """
    Compute persistence entropy from persistence diagram.

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams from persistence_diagram()
    dimension : int
        Homology dimension to analyze
    normalized : bool
        If True, normalize to [0, 1]

    Returns
    -------
    float
        Persistence entropy

    Notes
    -----
    H = -Σ p_i log(p_i)
    where p_i = |death_i - birth_i| / Σ|death - birth|

    High entropy: many features of similar importance
    Low entropy: few dominant features
    """
    if dimension not in diagrams:
        return np.nan

    dgm = diagrams[dimension]

    if len(dgm) == 0:
        return 0.0

    persistence = dgm[:, 1] - dgm[:, 0]

    # Remove infinite persistence
    persistence = persistence[np.isfinite(persistence)]

    if len(persistence) == 0:
        return 0.0

    # Normalize to probabilities
    total = np.sum(persistence)

    if total <= 0:
        return 0.0

    p = persistence / total

    # Entropy
    entropy = -np.sum(p * np.log(p + 1e-10))

    if normalized:
        max_entropy = np.log(len(p)) if len(p) > 1 else 1.0
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(entropy)


def persistence_landscape(
    diagrams: Dict[int, np.ndarray],
    dimension: int = 1,
    num_landscapes: int = 5,
    resolution: int = 100
) -> np.ndarray:
    """
    Compute persistence landscape from persistence diagram.

    Parameters
    ----------
    diagrams : dict
        Persistence diagrams from persistence_diagram()
    dimension : int
        Homology dimension to analyze
    num_landscapes : int
        Number of landscape functions to compute
    resolution : int
        Number of points in each landscape

    Returns
    -------
    np.ndarray
        Persistence landscapes (num_landscapes x resolution)

    Notes
    -----
    Persistence landscape is a functional summary of persistence diagram.
    Stable under small perturbations (Lipschitz continuous).
    Can be used for statistical analysis (mean, variance, etc.).
    """
    if dimension not in diagrams:
        return np.zeros((num_landscapes, resolution))

    dgm = diagrams[dimension]

    if len(dgm) == 0:
        return np.zeros((num_landscapes, resolution))

    births = dgm[:, 0]
    deaths = dgm[:, 1]

    # Remove infinite deaths
    finite_mask = np.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]

    if len(births) == 0:
        return np.zeros((num_landscapes, resolution))

    # Grid
    t_min = np.min(births)
    t_max = np.max(deaths)
    t_grid = np.linspace(t_min, t_max, resolution)

    # Tent functions
    def tent(t, b, d):
        """Tent function for (birth, death) pair at point t."""
        mid = (b + d) / 2
        height = (d - b) / 2
        if t <= b or t >= d:
            return 0.0
        elif t <= mid:
            return t - b
        else:
            return d - t

    # Compute landscape at each grid point
    landscapes = np.zeros((num_landscapes, resolution))

    for i, t in enumerate(t_grid):
        # Evaluate all tent functions at t
        values = [tent(t, b, d) for b, d in zip(births, deaths)]
        values = sorted(values, reverse=True)

        # Fill landscapes
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]

    return landscapes


# Fallback implementation

def _simple_persistence(
    points: np.ndarray,
    max_edge_length: float = None
) -> Dict[int, np.ndarray]:
    """Simple H0 persistence using union-find."""
    points = np.asarray(points)
    n = len(points)

    if n == 0:
        return {0: np.array([])}

    # Compute all pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            dists[i, j] = d
            dists[j, i] = d

    if max_edge_length is None:
        max_edge_length = np.max(dists)

    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    # Union-find
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Track births and deaths
    # Each point is born at 0
    births = np.zeros(n)
    deaths = np.full(n, np.inf)

    # Process edges
    for d, i, j in edges:
        if d > max_edge_length:
            break

        if find(i) != find(j):
            # Merge components - younger one dies
            pi, pj = find(i), find(j)
            # Younger = higher birth time (but all born at 0, so arbitrary)
            # Kill the higher-indexed root
            dying = max(pi, pj)
            deaths[dying] = d
            union(i, j)

    # Build diagram (only finite deaths, plus one infinite for remaining component)
    h0_diagram = []
    for i in range(n):
        if deaths[i] < np.inf:
            h0_diagram.append([births[i], deaths[i]])

    # Add one point for the component that survives
    h0_diagram.append([0.0, max_edge_length])

    return {0: np.array(h0_diagram) if h0_diagram else np.zeros((0, 2))}

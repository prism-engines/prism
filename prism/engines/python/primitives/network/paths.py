"""
Network Path Primitives (83)

Shortest paths, average path length, diameter.
"""

import numpy as np
from typing import Tuple, Optional


def shortest_paths(
    adjacency: np.ndarray,
    weighted: bool = False,
    source: int = None
) -> np.ndarray:
    """
    Compute shortest path distances.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, edge weights are distances (lower = closer)
        If False, all edges have unit weight
    source : int, optional
        If specified, only compute paths from this source

    Returns
    -------
    np.ndarray
        If source specified: 1D array of distances from source
        Otherwise: n x n distance matrix

    Notes
    -----
    Uses Floyd-Warshall for all pairs, Dijkstra for single source.
    Disconnected pairs have distance inf.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Build distance matrix
    if weighted:
        # Assume weights are already distances (lower = better)
        # If weights are similarities, caller should invert
        dist = np.where(adjacency != 0, adjacency, np.inf)
    else:
        dist = np.where(adjacency != 0, 1.0, np.inf)

    np.fill_diagonal(dist, 0)

    if source is not None:
        # Single-source Dijkstra
        return _dijkstra(dist, source)

    # All-pairs Floyd-Warshall
    dist = dist.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def average_path_length(
    adjacency: np.ndarray,
    weighted: bool = False
) -> float:
    """
    Compute average shortest path length.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted paths

    Returns
    -------
    float
        Average path length (only counting finite paths)

    Notes
    -----
    L = (1 / n(n-1)) Σ_{i≠j} d(i,j)

    Only finite paths are included.
    For disconnected graphs, this is the average within connected pairs.
    """
    dist = shortest_paths(adjacency, weighted)
    n = len(dist)

    if n < 2:
        return 0.0

    # Exclude diagonal and infinite
    mask = ~np.eye(n, dtype=bool)
    finite_dists = dist[mask]
    finite_dists = finite_dists[np.isfinite(finite_dists)]

    if len(finite_dists) == 0:
        return np.inf

    return float(np.mean(finite_dists))


def diameter(
    adjacency: np.ndarray,
    weighted: bool = False
) -> float:
    """
    Compute graph diameter (maximum shortest path length).

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted paths

    Returns
    -------
    float
        Diameter (inf if disconnected)

    Notes
    -----
    D = max_{i,j} d(i,j)

    For disconnected graphs, returns inf.
    """
    dist = shortest_paths(adjacency, weighted)
    n = len(dist)

    if n < 2:
        return 0.0

    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    finite_dists = dist[mask]

    return float(np.max(finite_dists))


def eccentricity(
    adjacency: np.ndarray,
    weighted: bool = False
) -> np.ndarray:
    """
    Compute eccentricity of each node.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted paths

    Returns
    -------
    np.ndarray
        Eccentricity for each node

    Notes
    -----
    e(v) = max_u d(v, u)

    Maximum distance from v to any other node.
    Diameter = max eccentricity
    Radius = min eccentricity
    """
    dist = shortest_paths(adjacency, weighted)
    n = len(dist)

    ecc = np.zeros(n)
    for i in range(n):
        other_dists = np.concatenate([dist[i, :i], dist[i, i+1:]])
        ecc[i] = np.max(other_dists) if len(other_dists) > 0 else 0

    return ecc


def radius(
    adjacency: np.ndarray,
    weighted: bool = False
) -> float:
    """
    Compute graph radius (minimum eccentricity).

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted paths

    Returns
    -------
    float
        Radius

    Notes
    -----
    r = min_v e(v) = min_v max_u d(v, u)

    The center nodes are those with eccentricity equal to radius.
    """
    ecc = eccentricity(adjacency, weighted)

    if len(ecc) == 0:
        return 0.0

    finite_ecc = ecc[np.isfinite(ecc)]
    return float(np.min(finite_ecc)) if len(finite_ecc) > 0 else np.inf


# Helper functions

def _dijkstra(dist_matrix: np.ndarray, source: int) -> np.ndarray:
    """Single-source Dijkstra's algorithm."""
    import heapq

    n = dist_matrix.shape[0]
    dist = np.full(n, np.inf)
    dist[source] = 0

    heap = [(0, source)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        for v in range(n):
            if v not in visited and dist_matrix[u, v] < np.inf:
                new_dist = dist[u] + dist_matrix[u, v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))

    return dist

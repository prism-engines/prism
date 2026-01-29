"""
Topology Distance Primitives (73-74)

Wasserstein and bottleneck distances between persistence diagrams.
"""

import numpy as np
from typing import Optional
from scipy.optimize import linear_sum_assignment


def wasserstein_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    p: float = 2.0,
    internal_p: float = np.inf
) -> float:
    """
    Compute p-Wasserstein distance between persistence diagrams.

    Parameters
    ----------
    dgm1, dgm2 : np.ndarray
        Persistence diagrams (n x 2) arrays of (birth, death) pairs
    p : float
        Wasserstein parameter (p >= 1)
    internal_p : float
        Internal metric (inf for L-infinity, 2 for L2)

    Returns
    -------
    float
        Wasserstein distance

    Notes
    -----
    W_p(D1, D2) = (inf_γ Σ ||x - γ(x)||^p)^(1/p)

    where γ ranges over all matchings, and points can be matched
    to their projection on the diagonal.

    Wasserstein distance is more sensitive to the overall distribution
    of points than bottleneck distance.
    """
    dgm1 = np.asarray(dgm1)
    dgm2 = np.asarray(dgm2)

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    if len(dgm1) == 0:
        dgm1 = np.zeros((0, 2))
    if len(dgm2) == 0:
        dgm2 = np.zeros((0, 2))

    # Filter out infinite points
    dgm1 = dgm1[np.isfinite(dgm1).all(axis=1)]
    dgm2 = dgm2[np.isfinite(dgm2).all(axis=1)]

    n1, n2 = len(dgm1), len(dgm2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Distance to diagonal
    def dist_to_diagonal(pt):
        return np.abs(pt[1] - pt[0]) / 2 if internal_p == np.inf else \
               np.abs(pt[1] - pt[0]) / np.sqrt(2)

    # Distance between two points
    def dist(p1, p2):
        if internal_p == np.inf:
            return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        else:
            return np.linalg.norm(p1 - p2, ord=internal_p)

    # Build cost matrix
    # Rows: dgm1 points + dgm2 diagonal projections
    # Cols: dgm2 points + dgm1 diagonal projections
    total_size = n1 + n2
    cost = np.zeros((total_size, total_size))

    # dgm1 to dgm2
    for i in range(n1):
        for j in range(n2):
            cost[i, j] = dist(dgm1[i], dgm2[j]) ** p

    # dgm1 to diagonal (columns n2 to n2+n1)
    for i in range(n1):
        cost[i, n2 + i] = dist_to_diagonal(dgm1[i]) ** p

    # dgm2 to diagonal (rows n1 to n1+n2)
    for j in range(n2):
        cost[n1 + j, j] = dist_to_diagonal(dgm2[j]) ** p

    # Diagonal to diagonal is 0 (implicit in zeros initialization)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)
    total_cost = cost[row_ind, col_ind].sum()

    return float(total_cost ** (1.0 / p))


def bottleneck_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    internal_p: float = np.inf
) -> float:
    """
    Compute bottleneck distance between persistence diagrams.

    Parameters
    ----------
    dgm1, dgm2 : np.ndarray
        Persistence diagrams (n x 2) arrays of (birth, death) pairs
    internal_p : float
        Internal metric (inf for L-infinity, 2 for L2)

    Returns
    -------
    float
        Bottleneck distance

    Notes
    -----
    d_B(D1, D2) = inf_γ max_x ||x - γ(x)||

    The bottleneck distance is the maximum cost in the optimal matching.
    It captures the cost of the worst-matched feature.

    Bottleneck ≤ Wasserstein_p for all p ≥ 1.
    """
    dgm1 = np.asarray(dgm1)
    dgm2 = np.asarray(dgm2)

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    if len(dgm1) == 0:
        dgm1 = np.zeros((0, 2))
    if len(dgm2) == 0:
        dgm2 = np.zeros((0, 2))

    # Filter out infinite points
    dgm1 = dgm1[np.isfinite(dgm1).all(axis=1)]
    dgm2 = dgm2[np.isfinite(dgm2).all(axis=1)]

    n1, n2 = len(dgm1), len(dgm2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Collect all candidate distances
    candidates = []

    # Distance to diagonal
    def dist_to_diagonal(pt):
        return np.abs(pt[1] - pt[0]) / 2 if internal_p == np.inf else \
               np.abs(pt[1] - pt[0]) / np.sqrt(2)

    # Distance between two points
    def dist(p1, p2):
        if internal_p == np.inf:
            return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        else:
            return np.linalg.norm(p1 - p2, ord=internal_p)

    # dgm1 to dgm2 distances
    for i in range(n1):
        for j in range(n2):
            candidates.append(dist(dgm1[i], dgm2[j]))

    # dgm1 to diagonal distances
    for i in range(n1):
        candidates.append(dist_to_diagonal(dgm1[i]))

    # dgm2 to diagonal distances
    for j in range(n2):
        candidates.append(dist_to_diagonal(dgm2[j]))

    candidates = sorted(set(candidates))

    # Binary search for bottleneck distance
    # Check if a perfect matching exists with max distance ≤ delta
    def can_match(delta):
        # Build bipartite graph
        total_size = n1 + n2
        edges = np.zeros((total_size, total_size), dtype=bool)

        # dgm1 to dgm2
        for i in range(n1):
            for j in range(n2):
                if dist(dgm1[i], dgm2[j]) <= delta + 1e-10:
                    edges[i, j] = True

        # dgm1 to diagonal
        for i in range(n1):
            if dist_to_diagonal(dgm1[i]) <= delta + 1e-10:
                edges[i, n2 + i] = True

        # dgm2 to diagonal
        for j in range(n2):
            if dist_to_diagonal(dgm2[j]) <= delta + 1e-10:
                edges[n1 + j, j] = True

        # Diagonal to diagonal always ok
        for k in range(max(n1, n2)):
            if n2 + k < total_size and n1 + k < total_size:
                edges[n1 + k, n2 + k] = True

        # Check if perfect matching exists using Hungarian algorithm
        cost = np.where(edges, 0, np.inf)
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
            return np.all(np.isfinite(cost[row_ind, col_ind]))
        except ValueError:
            return False

    # Binary search
    left, right = 0, len(candidates) - 1

    while left < right:
        mid = (left + right) // 2
        if can_match(candidates[mid]):
            right = mid
        else:
            left = mid + 1

    return float(candidates[left]) if candidates else 0.0

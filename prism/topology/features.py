"""
Topological Feature Extraction

Computes summary statistics and vectorizations from persistence diagrams.
"""

import numpy as np
from typing import List, Dict
from .persistence import PersistenceDiagram


def betti_numbers(
    diagrams: List[PersistenceDiagram],
    threshold: float
) -> Dict[int, int]:
    """
    Compute Betti numbers at a given filtration threshold.

    beta_k(threshold) = number of k-dimensional features alive at threshold

    Parameters
    ----------
    diagrams : list of PersistenceDiagram
    threshold : float
        Filtration value

    Returns
    -------
    betti : dict
        {dimension: count}
    """
    betti = {}
    for dgm in diagrams:
        if dgm.n_features == 0:
            betti[dgm.dimension] = 0
        else:
            alive = np.sum(
                (dgm.birth_times <= threshold) & (dgm.death_times > threshold)
            )
            betti[dgm.dimension] = int(alive)
    return betti


def betti_curve(
    diagrams: List[PersistenceDiagram],
    n_points: int = 100
) -> Dict[int, np.ndarray]:
    """
    Compute Betti numbers across all filtration values.

    Parameters
    ----------
    diagrams : list of PersistenceDiagram
    n_points : int
        Number of points in the curve

    Returns
    -------
    curves : dict
        {dimension: betti_curve_array}
    """
    # Find filtration range
    all_births = []
    all_deaths = []
    for d in diagrams:
        if len(d.birth_times) > 0:
            all_births.extend(d.birth_times)
        if len(d.death_times) > 0:
            finite_deaths = d.death_times[np.isfinite(d.death_times)]
            all_deaths.extend(finite_deaths)

    if not all_births:
        return {d.dimension: np.zeros(n_points) for d in diagrams}

    min_val = np.min(all_births)
    max_val = np.max(all_deaths) if all_deaths else np.max(all_births) * 2

    thresholds = np.linspace(min_val, max_val, n_points)

    curves = {}
    for dgm in diagrams:
        curve = np.zeros(n_points)
        if dgm.n_features > 0:
            for i, t in enumerate(thresholds):
                curve[i] = np.sum(
                    (dgm.birth_times <= t) & (dgm.death_times > t)
                )
        curves[dgm.dimension] = curve

    return curves


def persistence_statistics(dgm: PersistenceDiagram) -> Dict[str, float]:
    """
    Compute summary statistics from a persistence diagram.

    Parameters
    ----------
    dgm : PersistenceDiagram

    Returns
    -------
    stats : dict
        Summary statistics
    """
    if dgm.n_features == 0:
        return {
            'n_features': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'mean_birth': 0.0,
            'mean_death': 0.0,
        }

    pers = dgm.persistence
    pers = pers[np.isfinite(pers)]

    if len(pers) == 0:
        return persistence_statistics(PersistenceDiagram(
            dgm.dimension, np.array([]), np.array([])
        ))

    # Persistence entropy
    pers_sum = pers.sum()
    if pers_sum > 0:
        pers_norm = pers / pers_sum
        pers_norm = pers_norm[pers_norm > 0]
        entropy = -np.sum(pers_norm * np.log(pers_norm)) if len(pers_norm) > 0 else 0.0
    else:
        entropy = 0.0

    finite_deaths = dgm.death_times[np.isfinite(dgm.death_times)]

    return {
        'n_features': dgm.n_features,
        'total_persistence': float(np.sum(pers)),
        'max_persistence': float(np.max(pers)),
        'mean_persistence': float(np.mean(pers)),
        'std_persistence': float(np.std(pers)) if len(pers) > 1 else 0.0,
        'persistence_entropy': float(entropy),
        'mean_birth': float(np.mean(dgm.birth_times)),
        'mean_death': float(np.mean(finite_deaths)) if len(finite_deaths) > 0 else 0.0,
    }


def persistence_landscape(
    dgm: PersistenceDiagram,
    n_landscapes: int = 5,
    n_points: int = 100
) -> np.ndarray:
    """
    Compute persistence landscapes - a stable vectorization of persistence diagrams.

    Parameters
    ----------
    dgm : PersistenceDiagram
    n_landscapes : int
        Number of landscape functions
    n_points : int
        Resolution of each landscape

    Returns
    -------
    landscapes : array, shape (n_landscapes, n_points)
    """
    if dgm.n_features == 0:
        return np.zeros((n_landscapes, n_points))

    finite_mask = np.isfinite(dgm.death_times)
    births = dgm.birth_times[finite_mask]
    deaths = dgm.death_times[finite_mask]

    if len(births) == 0:
        return np.zeros((n_landscapes, n_points))

    # Filtration range
    min_val = np.min(births)
    max_val = np.max(deaths)
    t = np.linspace(min_val, max_val, n_points)

    # Tent functions for each feature
    def tent(b, d, x):
        """Tent function: rises from b to midpoint, falls to d."""
        return np.maximum(0, np.minimum(x - b, d - x))

    # Compute all tent functions
    tents = np.zeros((len(births), n_points))
    for i, (b, d) in enumerate(zip(births, deaths)):
        tents[i] = tent(b, d, t)

    # Sort at each t to get landscapes (k-th largest value)
    landscapes = np.zeros((n_landscapes, n_points))
    for j in range(n_points):
        sorted_vals = np.sort(tents[:, j])[::-1]
        for k in range(min(n_landscapes, len(sorted_vals))):
            landscapes[k, j] = sorted_vals[k]

    return landscapes


def topological_complexity(diagrams: List[PersistenceDiagram]) -> float:
    """
    Compute overall topological complexity score.

    Higher = more complex topology (more holes, more persistent features)

    Parameters
    ----------
    diagrams : list of PersistenceDiagram

    Returns
    -------
    complexity : float
    """
    total = 0.0
    for dgm in diagrams:
        stats = persistence_statistics(dgm)
        # Weight higher dimensions more
        weight = dgm.dimension + 1
        total += weight * stats['total_persistence']
    return total


def wasserstein_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
    p: float = 2.0
) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.

    Parameters
    ----------
    dgm1, dgm2 : PersistenceDiagram
    p : float
        Wasserstein order (default: 2)

    Returns
    -------
    distance : float
    """
    if dgm1.n_features == 0 and dgm2.n_features == 0:
        return 0.0

    # Get finite points
    pts1 = np.column_stack([dgm1.birth_times, dgm1.death_times])
    pts2 = np.column_stack([dgm2.birth_times, dgm2.death_times])

    finite1 = np.isfinite(pts1).all(axis=1)
    finite2 = np.isfinite(pts2).all(axis=1)
    pts1 = pts1[finite1]
    pts2 = pts2[finite2]

    if len(pts1) == 0 and len(pts2) == 0:
        return 0.0

    # Simplified approximation using closest point matching
    # (True Wasserstein requires optimal transport)
    if len(pts1) == 0:
        # Distance of pts2 to diagonal
        return np.sum(np.abs(pts2[:, 1] - pts2[:, 0]) ** p) ** (1/p) / 2

    if len(pts2) == 0:
        return np.sum(np.abs(pts1[:, 1] - pts1[:, 0]) ** p) ** (1/p) / 2

    # Greedy matching
    from scipy.spatial.distance import cdist
    cost_matrix = cdist(pts1, pts2)

    total_cost = 0.0
    used2 = set()

    for i in range(len(pts1)):
        min_cost = np.abs(pts1[i, 1] - pts1[i, 0]) / 2  # Cost to diagonal
        best_j = None

        for j in range(len(pts2)):
            if j not in used2:
                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_j = j

        total_cost += min_cost ** p
        if best_j is not None:
            used2.add(best_j)

    # Add unmatched pts2 to diagonal
    for j in range(len(pts2)):
        if j not in used2:
            total_cost += (np.abs(pts2[j, 1] - pts2[j, 0]) / 2) ** p

    return total_cost ** (1/p)

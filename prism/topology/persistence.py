"""
Persistent Homology Computation

Computes persistent homology using Vietoris-Rips complexes.
Tracks topological features (components, loops, voids) across scales.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform


@dataclass
class PersistenceDiagram:
    """Container for persistence diagram data."""
    dimension: int
    birth_times: np.ndarray
    death_times: np.ndarray

    @property
    def persistence(self) -> np.ndarray:
        """Lifetime of each feature."""
        return self.death_times - self.birth_times

    @property
    def n_features(self) -> int:
        return len(self.birth_times)

    def filter_by_persistence(self, min_persistence: float) -> 'PersistenceDiagram':
        """Keep only features with persistence above threshold."""
        mask = self.persistence >= min_persistence
        return PersistenceDiagram(
            dimension=self.dimension,
            birth_times=self.birth_times[mask],
            death_times=self.death_times[mask]
        )


def compute_rips_persistence(
    point_cloud: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: float = None,
    n_landmarks: int = None
) -> List[PersistenceDiagram]:
    """
    Compute persistent homology using Vietoris-Rips complex.

    Parameters
    ----------
    point_cloud : array, shape (n_points, n_dims)
        Point cloud data
    max_dimension : int
        Maximum homology dimension to compute
    max_edge_length : float, optional
        Maximum filtration value
    n_landmarks : int, optional
        If provided, subsample for large point clouds

    Returns
    -------
    diagrams : list of PersistenceDiagram
        One diagram per dimension (0 to max_dimension)
    """
    # Subsample if needed
    if n_landmarks and len(point_cloud) > n_landmarks:
        indices = np.random.choice(len(point_cloud), n_landmarks, replace=False)
        point_cloud = point_cloud[indices]

    try:
        # Try using ripser (fast C++ implementation)
        import ripser

        result = ripser.ripser(
            point_cloud,
            maxdim=max_dimension,
            thresh=max_edge_length
        )

        diagrams = []
        for dim, dgm in enumerate(result['dgms']):
            # Filter out infinite death times
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]

            if len(dgm_finite) == 0:
                diagrams.append(PersistenceDiagram(
                    dimension=dim,
                    birth_times=np.array([]),
                    death_times=np.array([])
                ))
            else:
                diagrams.append(PersistenceDiagram(
                    dimension=dim,
                    birth_times=dgm_finite[:, 0],
                    death_times=dgm_finite[:, 1]
                ))

        return diagrams

    except ImportError:
        # Fallback to pure Python implementation
        return _compute_rips_persistence_pure_python(
            point_cloud, max_dimension, max_edge_length
        )


def _compute_rips_persistence_pure_python(
    point_cloud: np.ndarray,
    max_dimension: int,
    max_edge_length: float
) -> List[PersistenceDiagram]:
    """
    Pure Python fallback for persistent homology (H0 only).

    Uses union-find for connected components.
    """
    # Compute distance matrix
    distances = squareform(pdist(point_cloud))
    n = len(point_cloud)

    if max_edge_length is None:
        max_edge_length = np.max(distances)

    # Union-Find for H0
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

    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= max_edge_length:
                edges.append((distances[i, j], i, j))
    edges.sort()

    # Track H0 births and deaths
    h0_birth = [0.0] * n
    h0_death = [np.inf] * n
    component_birth = {i: 0.0 for i in range(n)}

    for dist, i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            # Merge components - younger one dies
            birth_pi = component_birth.get(pi, 0.0)
            birth_pj = component_birth.get(pj, 0.0)

            if birth_pi < birth_pj:
                # pj dies
                h0_death[pj] = dist
            else:
                # pi dies
                h0_death[pi] = dist

            union(i, j)
            # Update component birth for merged component
            new_root = find(i)
            component_birth[new_root] = min(birth_pi, birth_pj)

    # Build H0 diagram (exclude infinite components)
    h0_pairs = []
    for i in range(n):
        if h0_death[i] < np.inf:
            h0_pairs.append((h0_birth[i], h0_death[i]))

    diagrams = [
        PersistenceDiagram(
            dimension=0,
            birth_times=np.array([p[0] for p in h0_pairs]) if h0_pairs else np.array([]),
            death_times=np.array([p[1] for p in h0_pairs]) if h0_pairs else np.array([])
        )
    ]

    # H1 and higher require more complex algorithms
    # Return empty diagrams for now
    for dim in range(1, max_dimension + 1):
        diagrams.append(PersistenceDiagram(
            dimension=dim,
            birth_times=np.array([]),
            death_times=np.array([])
        ))

    return diagrams

"""
Phase Space Reconstruction
==========================

Reconstructs the phase space from a 1D time series using
Takens' embedding theorem.

x(t) → [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]

Where:
    - τ is the time delay
    - m is the embedding dimension

The correlation dimension estimates the fractal dimension
of the reconstructed attractor.

References:
    Takens (1981) "Detecting strange attractors in turbulence"
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Dict


def compute(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1
) -> Dict[str, float]:
    """
    Reconstruct phase space and compute correlation dimension.

    Args:
        series: 1D numpy array of observations
        embedding_dim: Embedding dimension
        delay: Time delay

    Returns:
        dict with:
            - correlation_dimension: Estimated fractal dimension
            - n_vectors: Number of embedded vectors
            - mean_distance: Mean pairwise distance
    """
    n = len(series)
    n_vectors = n - (embedding_dim - 1) * delay

    if n_vectors < 20:
        return {
            'correlation_dimension': 0.0,
            'n_vectors': n_vectors,
            'mean_distance': 0.0
        }

    # Create embedded vectors
    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = series[i + j * delay]

    # Subsample for efficiency
    if n_vectors > 300:
        indices = np.linspace(0, n_vectors - 1, 300, dtype=int)
        embedded = embedded[indices]
        n_vectors = 300

    # Compute pairwise distances
    distances = pdist(embedded, 'euclidean')
    distances = distances[distances > 0]

    if len(distances) < 10:
        return {
            'correlation_dimension': 0.0,
            'n_vectors': n_vectors,
            'mean_distance': 0.0
        }

    mean_distance = np.mean(distances)

    # Correlation dimension via Grassberger-Procaccia
    # C(r) = lim (# pairs with d < r) / (total pairs)
    # D2 = lim log(C(r)) / log(r)

    # Use multiple radii
    r_min = np.percentile(distances, 5)
    r_max = np.percentile(distances, 95)

    if r_max <= r_min or r_min <= 0:
        return {
            'correlation_dimension': 0.0,
            'n_vectors': n_vectors,
            'mean_distance': float(mean_distance)
        }

    radii = np.exp(np.linspace(np.log(r_min), np.log(r_max), 20))
    correlations = []

    n_pairs = len(distances)
    for r in radii:
        c = np.sum(distances < r) / n_pairs
        if c > 0:
            correlations.append((np.log(r), np.log(c)))

    if len(correlations) < 5:
        return {
            'correlation_dimension': 0.0,
            'n_vectors': n_vectors,
            'mean_distance': float(mean_distance)
        }

    # Linear regression in log-log space
    log_r = np.array([c[0] for c in correlations])
    log_c = np.array([c[1] for c in correlations])

    # Use middle portion (more reliable)
    mid_start = len(log_r) // 4
    mid_end = 3 * len(log_r) // 4
    if mid_end > mid_start + 3:
        slope, _ = np.polyfit(log_r[mid_start:mid_end], log_c[mid_start:mid_end], 1)
    else:
        slope, _ = np.polyfit(log_r, log_c, 1)

    return {
        'correlation_dimension': float(max(0, slope)),
        'n_vectors': n_vectors,
        'mean_distance': float(mean_distance)
    }

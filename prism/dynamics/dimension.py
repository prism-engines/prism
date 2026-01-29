"""
Attractor Dimension Estimation

Computes fractal dimensions that characterize attractor complexity:
- D ~ 1: limit cycle (periodic orbit)
- D ~ 2: torus (quasiperiodic)
- D non-integer: strange attractor (chaos)

References:
    Grassberger, P. & Procaccia, I. (1983).
    "Characterization of strange attractors"
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist

from .reconstruction import embed_time_series, optimal_delay, optimal_embedding_dim


def correlation_dimension(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    r_range: tuple = (0.01, 1.0),
    n_radii: int = 20,
    max_pairs: int = 5000
) -> float:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.

    The correlation dimension D2 is defined by the scaling:
        C(r) ~ r^D2  as r -> 0

    where C(r) is the correlation sum (fraction of point pairs
    within distance r).

    Parameters
    ----------
    x : array
        Time series
    tau : int, optional
        Time delay for embedding
    dim : int, optional
        Embedding dimension
    r_range : tuple
        Range of radii as fraction of data std (min, max)
    n_radii : int
        Number of radius values to compute
    max_pairs : int
        Maximum pairs to use (for computational efficiency)

    Returns
    -------
    D2 : float
        Correlation dimension estimate

    Notes
    -----
    Typical values:
        - Lorenz attractor: ~2.05
        - Henon map: ~1.2
        - White noise: increases with embedding dimension

    Examples
    --------
    >>> # Lorenz attractor dimension ~ 2.05
    >>> x = lorenz_trajectory[:, 0]  # x-component
    >>> D2 = correlation_dimension(x)
    >>> 1.8 < D2 < 2.3
    True
    """
    x = np.asarray(x).flatten()

    if len(x) < 100:
        return np.nan

    # Auto-compute embedding parameters
    if tau is None:
        tau = optimal_delay(x, max_tau=min(50, len(x) // 20))
    if dim is None:
        dim = optimal_embedding_dim(x, tau, max_dim=min(8, len(x) // (5 * tau)))

    # Embed
    try:
        embedded = embed_time_series(x, tau, dim)
    except ValueError:
        return np.nan

    n = len(embedded)
    if n < 50:
        return np.nan

    # Subsample if too many points
    if n > int(np.sqrt(2 * max_pairs)):
        indices = np.random.choice(n, size=int(np.sqrt(2 * max_pairs)), replace=False)
        embedded = embedded[indices]
        n = len(embedded)

    # Compute pairwise distances
    distances = pdist(embedded)

    # Define radius range based on distance distribution
    std = np.std(distances)
    r_min = r_range[0] * std
    r_max = r_range[1] * std
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)

    # Correlation sum C(r) = fraction of pairs within distance r
    C = np.zeros(len(radii))
    for i, r in enumerate(radii):
        C[i] = np.mean(distances < r)

    # Find scaling region (avoid edge effects)
    valid = (C > 0.005) & (C < 0.5)

    if valid.sum() < 5:
        return np.nan

    log_r = np.log(radii[valid])
    log_C = np.log(C[valid])

    # Linear fit in log-log space
    try:
        slope, intercept = np.polyfit(log_r, log_C, 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    return float(slope)


def box_counting_dimension(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    n_scales: int = 15
) -> float:
    """
    Estimate box-counting (capacity) dimension.

    Count number of boxes N(r) of size r needed to cover the attractor:
        N(r) ~ r^(-D0)  as r -> 0

    Parameters
    ----------
    x : array
        Time series
    tau, dim : int, optional
        Embedding parameters
    n_scales : int
        Number of box sizes to try

    Returns
    -------
    D0 : float
        Box-counting dimension
    """
    x = np.asarray(x).flatten()

    if tau is None:
        tau = optimal_delay(x)
    if dim is None:
        dim = optimal_embedding_dim(x, tau)

    try:
        embedded = embed_time_series(x, tau, dim)
    except ValueError:
        return np.nan

    # Normalize to unit hypercube
    mins = embedded.min(axis=0)
    maxs = embedded.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (embedded - mins) / ranges

    # Count boxes at different scales
    box_sizes = np.logspace(-2, 0, n_scales)
    counts = []

    for size in box_sizes:
        # Discretize points to boxes
        boxes = (normalized / size).astype(int)
        # Count unique boxes
        unique_boxes = set(map(tuple, boxes))
        counts.append(len(unique_boxes))

    counts = np.array(counts)
    valid = counts > 1

    if valid.sum() < 5:
        return np.nan

    log_r = np.log(1 / box_sizes[valid])
    log_N = np.log(counts[valid])

    try:
        slope, _ = np.polyfit(log_r, log_N, 1)
    except:
        return np.nan

    return float(slope)


def kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    Estimate attractor dimension from Lyapunov spectrum.

    The Kaplan-Yorke dimension is:
        D_KY = j + (lambda_1 + ... + lambda_j) / |lambda_{j+1}|

    where j is the largest index such that the sum of the first j
    exponents is non-negative.

    Parameters
    ----------
    lyapunov_spectrum : array
        Lyapunov exponents in descending order

    Returns
    -------
    D_KY : float
        Kaplan-Yorke dimension estimate

    Notes
    -----
    This provides an upper bound on the information dimension.
    Requires accurate estimation of multiple Lyapunov exponents.
    """
    spectrum = np.asarray(lyapunov_spectrum)
    spectrum = np.sort(spectrum)[::-1]  # Ensure descending order

    if len(spectrum) == 0 or np.any(np.isnan(spectrum)):
        return np.nan

    cumsum = np.cumsum(spectrum)

    # Find j where cumsum becomes negative
    negative_idx = np.where(cumsum < 0)[0]

    if len(negative_idx) == 0:
        # All sums positive - dimension equals embedding dimension
        return float(len(spectrum))

    j = negative_idx[0]

    if j == 0:
        # First exponent already makes sum negative
        return 0.0

    # D_KY = j + sum(lambda_1..j) / |lambda_{j+1}|
    if abs(spectrum[j]) < 1e-10:
        return float(j)

    return float(j + cumsum[j-1] / abs(spectrum[j]))


def information_dimension(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    n_radii: int = 20
) -> float:
    """
    Estimate information dimension D1.

    Related to the entropy of the probability distribution on the attractor.
    D0 >= D1 >= D2 (box-counting >= information >= correlation)

    Parameters
    ----------
    x : array
        Time series
    tau, dim : int, optional
        Embedding parameters
    n_radii : int
        Number of radius values

    Returns
    -------
    D1 : float
        Information dimension
    """
    x = np.asarray(x).flatten()

    if tau is None:
        tau = optimal_delay(x)
    if dim is None:
        dim = optimal_embedding_dim(x, tau)

    try:
        embedded = embed_time_series(x, tau, dim)
    except ValueError:
        return np.nan

    n = len(embedded)
    if n < 50:
        return np.nan

    # Subsample for efficiency
    if n > 500:
        indices = np.random.choice(n, 500, replace=False)
        sample = embedded[indices]
    else:
        sample = embedded

    # Compute distances
    distances = cdist(sample, embedded)

    # Range of radii
    std = np.std(distances)
    radii = np.logspace(np.log10(0.01 * std), np.log10(std), n_radii)

    # Information entropy at each scale
    entropies = []
    for r in radii:
        # Probability = fraction of points within r
        probs = np.mean(distances < r, axis=1)
        probs = probs[probs > 0]  # Exclude zeros

        if len(probs) > 0:
            entropy = -np.mean(np.log(probs))
            entropies.append(entropy)
        else:
            entropies.append(np.nan)

    entropies = np.array(entropies)
    valid = ~np.isnan(entropies)

    if valid.sum() < 5:
        return np.nan

    log_r = np.log(radii[valid])
    H = entropies[valid]

    try:
        slope, _ = np.polyfit(log_r, H, 1)
    except:
        return np.nan

    return float(-slope)

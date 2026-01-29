"""
Dynamical Dimension Primitives (88)

Correlation dimension and related measures.
"""

import numpy as np
from typing import Tuple, Optional


def correlation_dimension(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    r_min: float = None,
    r_max: float = None,
    n_r: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int, optional
        Embedding dimension (default: auto)
    delay : int, optional
        Time delay (default: auto)
    r_min, r_max : float, optional
        Range of radii to test
    n_r : int
        Number of radii

    Returns
    -------
    tuple
        (D2, log_r, log_C)
        D2: Correlation dimension
        log_r: Log of radii tested
        log_C: Log of correlation sums

    Notes
    -----
    D2 = lim_{r→0} log(C(r)) / log(r)

    where C(r) = (2 / N(N-1)) Σ Θ(r - ||x_i - x_j||)

    D2 < embedding dimension suggests deterministic dynamics.
    D2 ≈ embedding dimension suggests noise or high-dimensional chaos.
    """
    signal = np.asarray(signal).flatten()

    # Auto-detect parameters
    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)

    # Embed
    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < 50:
        return np.nan, np.array([]), np.array([])

    # Compute pairwise distances
    dists = []
    sample_size = min(1000, n_points)
    sample_idx = np.random.choice(n_points, sample_size, replace=False)

    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            d = np.linalg.norm(embedded[sample_idx[i]] - embedded[sample_idx[j]])
            if d > 0:
                dists.append(d)

    if len(dists) < 10:
        return np.nan, np.array([]), np.array([])

    dists = np.array(dists)

    # Range of radii
    if r_min is None:
        r_min = np.percentile(dists, 1)
    if r_max is None:
        r_max = np.percentile(dists, 50)

    if r_min <= 0 or r_max <= r_min:
        return np.nan, np.array([]), np.array([])

    log_r = np.linspace(np.log(r_min), np.log(r_max), n_r)
    radii = np.exp(log_r)

    # Correlation sum
    log_C = np.zeros(n_r)
    n_pairs = len(dists)

    for i, r in enumerate(radii):
        count = np.sum(dists < r)
        C = count / n_pairs
        log_C[i] = np.log(C) if C > 0 else -np.inf

    # Fit slope in scaling region
    valid = np.isfinite(log_C)
    if np.sum(valid) < 3:
        return np.nan, log_r, log_C

    # Use middle portion of scaling region
    slope, _ = np.polyfit(log_r[valid], log_C[valid], 1)

    return float(slope), log_r, log_C


def correlation_integral(
    embedded: np.ndarray,
    r: float
) -> float:
    """
    Compute correlation integral for embedded trajectory.

    Parameters
    ----------
    embedded : np.ndarray
        Embedded trajectory (n_points x dimension)
    r : float
        Radius

    Returns
    -------
    float
        Correlation sum C(r)

    Notes
    -----
    C(r) = (2 / N(N-1)) Σ_{i<j} Θ(r - ||x_i - x_j||)

    Θ = Heaviside step function.
    """
    embedded = np.asarray(embedded)
    n = len(embedded)

    if n < 2:
        return 0.0

    count = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < r:
                count += 1
            total += 1

    return count / total if total > 0 else 0.0


def information_dimension(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    n_boxes: int = 20
) -> float:
    """
    Estimate information dimension using box-counting.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int, optional
        Embedding dimension
    delay : int, optional
        Time delay
    n_boxes : int
        Number of box sizes to test

    Returns
    -------
    float
        Information dimension D1

    Notes
    -----
    D1 = lim_{ε→0} Σ p_i log(p_i) / log(ε)

    where p_i = fraction of points in box i.

    D1 ≤ D0 (box-counting dimension) ≤ D2 (correlation dimension)
    """
    signal = np.asarray(signal).flatten()

    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)

    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < 50:
        return np.nan

    # Normalize to [0, 1]
    mins = np.min(embedded, axis=0)
    maxs = np.max(embedded, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (embedded - mins) / ranges

    # Test different box sizes
    log_eps = np.linspace(np.log(0.01), np.log(0.5), n_boxes)
    epsilons = np.exp(log_eps)

    log_info = np.zeros(n_boxes)

    for k, eps in enumerate(epsilons):
        # Count points in each box
        n_boxes_per_dim = int(np.ceil(1.0 / eps))
        box_indices = np.floor(normalized / eps).astype(int)
        box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)

        # Convert to single index
        multipliers = n_boxes_per_dim ** np.arange(dimension)
        flat_indices = np.sum(box_indices * multipliers, axis=1)

        # Count occupancy
        unique, counts = np.unique(flat_indices, return_counts=True)
        p = counts / n_points

        # Information entropy
        info = -np.sum(p * np.log(p + 1e-10))
        log_info[k] = info

    # Fit slope: D1 = -d(info)/d(log_eps)
    slope, _ = np.polyfit(log_eps, log_info, 1)
    D1 = -slope

    return float(D1)


# Helper functions

def _embed(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]
    return embedded


def _auto_delay(signal: np.ndarray) -> int:
    """Auto-detect delay using autocorrelation 1/e decay."""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    var = np.var(signal_centered)
    if var == 0:
        return 1

    for lag in range(1, n // 4):
        acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
        if acf < 1 / np.e:
            return lag

    return max(1, n // 10)


def kaplan_yorke_dimension(
    lyapunov_spectrum: np.ndarray
) -> float:
    """
    Compute Kaplan-Yorke dimension from Lyapunov spectrum.

    Parameters
    ----------
    lyapunov_spectrum : np.ndarray
        Lyapunov exponents sorted in descending order

    Returns
    -------
    float
        Kaplan-Yorke (Lyapunov) dimension

    Notes
    -----
    D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|

    where j is the largest index such that Σ_{i=1}^{j} λ_i >= 0.

    Physical interpretation:
    - D_KY estimates the dimension of the attractor
    - For chaotic systems: fractal dimension
    - D_KY = 0: fixed point
    - D_KY = 1: limit cycle
    - Non-integer: strange attractor (chaos)

    The sum of positive Lyapunov exponents equals the
    Kolmogorov-Sinai entropy (rate of information production).
    """
    spectrum = np.asarray(lyapunov_spectrum).flatten()

    if len(spectrum) == 0:
        return np.nan

    # Sort descending (should already be, but ensure)
    spectrum = np.sort(spectrum)[::-1]

    # Find j such that cumsum is still positive
    cumsum = np.cumsum(spectrum)

    # Find largest j where cumsum >= 0
    j = -1
    for i in range(len(cumsum)):
        if cumsum[i] >= 0:
            j = i
        else:
            break

    if j < 0:
        # No positive exponents - stable fixed point
        return 0.0

    if j >= len(spectrum) - 1:
        # All exponents sum to positive - return dimension
        return float(len(spectrum))

    # D_KY = j + 1 + cumsum[j] / |λ_{j+1}|
    # Note: j is 0-indexed, so j+1 is the "j" in the formula
    lambda_next = spectrum[j + 1]

    if lambda_next == 0:
        return float(j + 1)

    D_KY = (j + 1) + cumsum[j] / abs(lambda_next)

    return float(D_KY)


def _auto_dimension(signal: np.ndarray, delay: int) -> int:
    """Auto-detect embedding dimension."""
    # Use Cao's method simplified
    e_values = []

    for dim in range(1, 11):
        emb = _embed(signal, dim, delay)
        if len(emb) < 50:
            break

        # Compute mean distance to nearest neighbor
        sample = emb[np.random.choice(len(emb), min(200, len(emb)), replace=False)]
        dists = []
        for i in range(len(sample)):
            min_dist = np.inf
            for j in range(len(sample)):
                if i != j:
                    d = np.linalg.norm(sample[i] - sample[j])
                    if 0 < d < min_dist:
                        min_dist = d
            if min_dist < np.inf:
                dists.append(min_dist)

        if dists:
            e_values.append(np.mean(dists))

    # Find where E(d+1)/E(d) stabilizes
    if len(e_values) < 2:
        return 3

    for i in range(len(e_values) - 1):
        ratio = e_values[i + 1] / e_values[i] if e_values[i] > 0 else 1
        if abs(ratio - 1) < 0.1:
            return i + 2

    return min(5, len(e_values) + 1)

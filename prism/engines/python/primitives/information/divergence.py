"""
Information Divergence Primitives (113-114)

Cross-entropy, KL divergence, JS divergence.
"""

import numpy as np
from typing import Optional


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute cross-entropy H(P, Q).

    Parameters
    ----------
    p : np.ndarray
        True distribution (or data to estimate from)
    q : np.ndarray
        Approximate distribution (or data to estimate from)
    bins : int, optional
        Number of bins for continuous data
    base : float
        Logarithm base

    Returns
    -------
    float
        Cross-entropy H(P, Q)

    Notes
    -----
    H(P, Q) = -Σ p(x) log(q(x))
            = H(P) + D_KL(P || Q)

    Measures average number of bits needed to encode samples from P
    using a code optimized for Q.

    Cross-entropy ≥ entropy, with equality when P = Q.
    """
    p_dist = _to_distribution(p, bins)
    q_dist = _to_distribution(q, bins)

    # Align distributions
    n = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:n], q_dist[:n]

    # Avoid log(0)
    q_safe = np.where(q_dist > 0, q_dist, 1e-10)

    # Cross-entropy
    if base == np.e:
        ce = -np.sum(p_dist * np.log(q_safe))
    elif base == 2:
        ce = -np.sum(p_dist * np.log2(q_safe))
    else:
        ce = -np.sum(p_dist * np.log(q_safe)) / np.log(base)

    return float(ce)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).

    Parameters
    ----------
    p : np.ndarray
        True distribution (or data)
    q : np.ndarray
        Approximate distribution (or data)
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        KL divergence

    Notes
    -----
    D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
                 = H(P, Q) - H(P)

    Measures information lost when Q is used to approximate P.

    Properties:
    - D_KL ≥ 0, with equality iff P = Q
    - NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
    - Infinite if q(x) = 0 where p(x) > 0
    """
    p_dist = _to_distribution(p, bins)
    q_dist = _to_distribution(q, bins)

    n = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:n], q_dist[:n]

    # Only consider where p > 0
    mask = p_dist > 0
    if not np.any(mask):
        return 0.0

    # Check for infinite KL (q=0 where p>0)
    if np.any((p_dist > 0) & (q_dist == 0)):
        return np.inf

    p_safe = p_dist[mask]
    q_safe = q_dist[mask]

    # KL divergence
    if base == np.e:
        kl = np.sum(p_safe * np.log(p_safe / q_safe))
    elif base == 2:
        kl = np.sum(p_safe * np.log2(p_safe / q_safe))
    else:
        kl = np.sum(p_safe * np.log(p_safe / q_safe)) / np.log(base)

    return float(max(0, kl))  # Ensure non-negative due to numerical issues


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute Jensen-Shannon divergence.

    Parameters
    ----------
    p, q : np.ndarray
        Two distributions (or data)
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        JS divergence

    Notes
    -----
    D_JS(P || Q) = (D_KL(P || M) + D_KL(Q || M)) / 2
    where M = (P + Q) / 2

    Properties:
    - Symmetric: D_JS(P||Q) = D_JS(Q||P)
    - Bounded: 0 ≤ D_JS ≤ log(2) (in bits: 0 ≤ D_JS ≤ 1)
    - sqrt(D_JS) is a proper metric
    """
    p_dist = _to_distribution(p, bins)
    q_dist = _to_distribution(q, bins)

    n = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:n], q_dist[:n]

    # Mixture distribution
    m = (p_dist + q_dist) / 2

    # JS = average of KL divergences to mixture
    kl_pm = _kl_core(p_dist, m, base)
    kl_qm = _kl_core(q_dist, m, base)

    js = (kl_pm + kl_qm) / 2

    return float(max(0, js))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = None
) -> float:
    """
    Compute Hellinger distance.

    Parameters
    ----------
    p, q : np.ndarray
        Two distributions (or data)
    bins : int, optional
        Number of bins

    Returns
    -------
    float
        Hellinger distance in [0, 1]

    Notes
    -----
    H(P, Q) = (1/√2) √(Σ (√p(x) - √q(x))²)

    Properties:
    - Symmetric
    - Bounded: 0 ≤ H ≤ 1
    - H = 0 iff P = Q
    - H = 1 iff P and Q have disjoint support
    """
    p_dist = _to_distribution(p, bins)
    q_dist = _to_distribution(q, bins)

    n = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:n], q_dist[:n]

    h = np.sqrt(np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2)) / np.sqrt(2)

    return float(h)


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
    bins: int = None
) -> float:
    """
    Compute total variation distance.

    Parameters
    ----------
    p, q : np.ndarray
        Two distributions (or data)
    bins : int, optional
        Number of bins

    Returns
    -------
    float
        Total variation distance in [0, 1]

    Notes
    -----
    TV(P, Q) = (1/2) Σ |p(x) - q(x)|

    Maximum difference in probability of any event.
    """
    p_dist = _to_distribution(p, bins)
    q_dist = _to_distribution(q, bins)

    n = min(len(p_dist), len(q_dist))
    p_dist, q_dist = p_dist[:n], q_dist[:n]

    tv = np.sum(np.abs(p_dist - q_dist)) / 2

    return float(tv)


# Helper functions

def _to_distribution(
    data: np.ndarray,
    bins: int = None
) -> np.ndarray:
    """Convert data to probability distribution."""
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.array([1.0])

    # Check if already a distribution
    if np.all(data >= 0) and np.isclose(np.sum(data), 1.0):
        return data

    # Estimate from data
    if bins is None:
        bins = max(5, int(np.sqrt(len(data))))

    counts, _ = np.histogram(data, bins=bins)
    dist = counts / counts.sum()

    return dist


def _kl_core(p: np.ndarray, q: np.ndarray, base: float) -> float:
    """Core KL computation for aligned distributions."""
    mask = p > 0
    if not np.any(mask):
        return 0.0

    p_safe = p[mask]
    q_safe = np.maximum(q[mask], 1e-10)

    if base == np.e:
        return np.sum(p_safe * np.log(p_safe / q_safe))
    elif base == 2:
        return np.sum(p_safe * np.log2(p_safe / q_safe))
    else:
        return np.sum(p_safe * np.log(p_safe / q_safe)) / np.log(base)

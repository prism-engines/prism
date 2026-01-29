"""
Information Entropy Primitives (108-112)

Shannon, Renyi, Tsallis, joint, conditional entropy.
"""

import numpy as np
from typing import Tuple, Optional


def shannon_entropy(
    data: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute Shannon entropy.

    Parameters
    ----------
    data : np.ndarray
        Data (continuous: will be binned; discrete: used as-is if integer)
    bins : int, optional
        Number of bins for continuous data (default: sqrt(n))
    base : float
        Logarithm base (2 for bits, e for nats)

    Returns
    -------
    float
        Shannon entropy

    Notes
    -----
    H(X) = -Σ p(x) log(p(x))

    Measures uncertainty/information content.
    Higher entropy = more uncertain/unpredictable.
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return 0.0

    # Estimate probability distribution
    p = _estimate_probabilities(data, bins)

    # Remove zeros
    p = p[p > 0]

    # Shannon entropy
    if base == np.e:
        entropy = -np.sum(p * np.log(p))
    elif base == 2:
        entropy = -np.sum(p * np.log2(p))
    else:
        entropy = -np.sum(p * np.log(p)) / np.log(base)

    return float(entropy)


def renyi_entropy(
    data: np.ndarray,
    alpha: float = 2.0,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute Renyi entropy.

    Parameters
    ----------
    data : np.ndarray
        Input data
    alpha : float
        Order parameter (α ≠ 1; α=2 gives collision entropy)
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Renyi entropy of order α

    Notes
    -----
    H_α(X) = (1/(1-α)) log(Σ p(x)^α)

    Special cases:
    - α → 0: Max entropy (log of support size)
    - α → 1: Shannon entropy
    - α = 2: Collision entropy
    - α → ∞: Min entropy (determined by most likely event)
    """
    if alpha == 1:
        return shannon_entropy(data, bins, base)

    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return 0.0

    p = _estimate_probabilities(data, bins)
    p = p[p > 0]

    # Renyi entropy
    log_sum = np.log(np.sum(p ** alpha))

    if base == np.e:
        entropy = log_sum / (1 - alpha)
    elif base == 2:
        entropy = log_sum / (np.log(2) * (1 - alpha))
    else:
        entropy = log_sum / (np.log(base) * (1 - alpha))

    return float(entropy)


def tsallis_entropy(
    data: np.ndarray,
    q: float = 2.0,
    bins: int = None
) -> float:
    """
    Compute Tsallis entropy.

    Parameters
    ----------
    data : np.ndarray
        Input data
    q : float
        Entropic index (q ≠ 1)
    bins : int, optional
        Number of bins

    Returns
    -------
    float
        Tsallis entropy of order q

    Notes
    -----
    S_q(X) = (1 - Σ p(x)^q) / (q - 1)

    Non-additive entropy used in non-extensive statistical mechanics.
    q → 1: Shannon entropy
    q < 1: emphasizes rare events
    q > 1: emphasizes frequent events
    """
    if np.isclose(q, 1):
        return shannon_entropy(data, bins, base=np.e)

    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return 0.0

    p = _estimate_probabilities(data, bins)
    p = p[p > 0]

    # Tsallis entropy
    entropy = (1 - np.sum(p ** q)) / (q - 1)

    return float(entropy)


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute joint entropy of two variables.

    Parameters
    ----------
    x, y : np.ndarray
        Input variables
    bins : int, optional
        Number of bins per dimension
    base : float
        Logarithm base

    Returns
    -------
    float
        Joint entropy H(X, Y)

    Notes
    -----
    H(X, Y) = -Σ p(x, y) log(p(x, y))

    Properties:
    - H(X, Y) ≥ max(H(X), H(Y))
    - H(X, Y) ≤ H(X) + H(Y) (equality iff independent)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return 0.0

    if bins is None:
        bins = max(5, int(np.sqrt(len(x))))

    # Joint histogram
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist / hist.sum()
    p_xy = p_xy.flatten()
    p_xy = p_xy[p_xy > 0]

    # Joint entropy
    if base == np.e:
        entropy = -np.sum(p_xy * np.log(p_xy))
    elif base == 2:
        entropy = -np.sum(p_xy * np.log2(p_xy))
    else:
        entropy = -np.sum(p_xy * np.log(p_xy)) / np.log(base)

    return float(entropy)


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute conditional entropy H(X|Y).

    Parameters
    ----------
    x : np.ndarray
        Variable to condition
    y : np.ndarray
        Conditioning variable
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Conditional entropy H(X|Y)

    Notes
    -----
    H(X|Y) = H(X, Y) - H(Y)
           = -Σ p(x, y) log(p(x|y))

    Measures uncertainty in X given knowledge of Y.
    H(X|Y) ≤ H(X), with equality iff X and Y are independent.
    """
    h_xy = joint_entropy(x, y, bins, base)
    h_y = shannon_entropy(y, bins, base)

    return float(h_xy - h_y)


# Helper functions

def _estimate_probabilities(
    data: np.ndarray,
    bins: int = None
) -> np.ndarray:
    """Estimate probability distribution from data."""
    n = len(data)

    # Check if data is discrete (integers with few unique values)
    if np.issubdtype(data.dtype, np.integer):
        unique_vals = np.unique(data)
        if len(unique_vals) <= min(50, n // 2):
            # Use empirical frequencies
            counts = np.bincount(data - data.min())
            return counts / counts.sum()

    # Continuous data: use histogram
    if bins is None:
        bins = max(5, int(np.sqrt(n)))

    counts, _ = np.histogram(data, bins=bins)
    return counts / counts.sum()

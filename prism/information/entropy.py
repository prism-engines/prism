"""
Information-Theoretic Measures

Computes Shannon entropy, mutual information, and related quantities.
"""

import numpy as np
from typing import Optional


def shannon_entropy(
    x: np.ndarray,
    bins: int = 20,
    base: float = 2.0
) -> float:
    """
    Compute Shannon entropy using histogram estimation.

    H(X) = -sum p(x) log p(x)

    Parameters
    ----------
    x : array
        Data samples
    bins : int
        Number of histogram bins
    base : float
        Logarithm base (2 for bits, e for nats)

    Returns
    -------
    entropy : float
        Shannon entropy
    """
    x = np.asarray(x).flatten()

    if len(x) < 2:
        return 0.0

    hist, bin_edges = np.histogram(x, bins=bins)
    hist = hist / hist.sum()  # Normalize to probability
    hist = hist[hist > 0]  # Remove zeros

    if len(hist) == 0:
        return 0.0

    return -np.sum(hist * np.log(hist) / np.log(base))


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
    base: float = 2.0
) -> float:
    """
    Compute mutual information I(X; Y) using histogram estimation.

    I(X; Y) = H(X) + H(Y) - H(X, Y)

    Parameters
    ----------
    x, y : array
        Data samples (must be same length)
    bins : int
        Number of histogram bins per dimension
    base : float
        Logarithm base

    Returns
    -------
    mi : float
        Mutual information (>= 0)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 2:
        return 0.0

    # Joint distribution
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    hist_xy = hist_xy / hist_xy.sum()

    # Marginals
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)

    # I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                mi += hist_xy[i, j] * np.log(
                    hist_xy[i, j] / (hist_x[i] * hist_y[j])
                ) / np.log(base)

    return max(0.0, mi)


def kraskov_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute mutual information using Kraskov et al. (2004) k-NN estimator.

    More accurate for continuous variables than histogram method.

    Parameters
    ----------
    x, y : array
        Data samples
    k : int
        Number of nearest neighbors

    Returns
    -------
    mi : float
        Mutual information estimate
    """
    from scipy.special import digamma

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < k + 1:
        return 0.0

    # Reshape for distance computation
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xy = np.hstack([x, y])

    # Use Chebyshev (max) metric
    def chebyshev_dist(a, b):
        return np.max(np.abs(a - b), axis=-1)

    # For each point, find distance to k-th neighbor in joint space
    eps = np.zeros(n)
    for i in range(n):
        dists = chebyshev_dist(xy[i], xy)
        dists[i] = np.inf  # Exclude self
        eps[i] = np.sort(dists)[k - 1]  # k-th smallest

    # Count neighbors within eps in marginal spaces
    n_x = np.zeros(n, dtype=int)
    n_y = np.zeros(n, dtype=int)

    for i in range(n):
        n_x[i] = np.sum(np.abs(x - x[i]).flatten() <= eps[i]) - 1
        n_y[i] = np.sum(np.abs(y - y[i]).flatten() <= eps[i]) - 1

    # Kraskov formula
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)

    return max(0.0, mi)


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
    base: float = 2.0
) -> float:
    """
    Compute conditional entropy H(X|Y).

    H(X|Y) = H(X,Y) - H(Y)

    Parameters
    ----------
    x, y : array
        Data samples
    bins : int
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    h_x_given_y : float
        Conditional entropy
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 2:
        return 0.0

    # H(X,Y)
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    hist_xy = hist_xy / hist_xy.sum()
    h_xy = -np.sum(hist_xy[hist_xy > 0] * np.log(hist_xy[hist_xy > 0]) / np.log(base))

    # H(Y)
    h_y = shannon_entropy(y, bins, base)

    return h_xy - h_y


def normalized_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20
) -> float:
    """
    Compute normalized mutual information.

    NMI = 2 * I(X;Y) / (H(X) + H(Y))

    Returns value in [0, 1].
    """
    mi = mutual_information(x, y, bins)
    h_x = shannon_entropy(x, bins)
    h_y = shannon_entropy(y, bins)

    if h_x + h_y == 0:
        return 0.0

    return 2 * mi / (h_x + h_y)

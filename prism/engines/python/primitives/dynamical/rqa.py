"""
Dynamical RQA Primitives (89-95)

Recurrence Quantification Analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def recurrence_matrix(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    threshold: float = None,
    threshold_percentile: float = 10.0,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute recurrence matrix.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int
        Embedding dimension
    delay : int
        Time delay
    threshold : float, optional
        Recurrence threshold (if None, use percentile)
    threshold_percentile : float
        Percentile of distances for auto threshold
    metric : str
        Distance metric: 'euclidean', 'maximum', 'manhattan'

    Returns
    -------
    np.ndarray
        Binary recurrence matrix (n x n)

    Notes
    -----
    R_{ij} = Θ(ε - ||x_i - x_j||)

    where Θ is Heaviside step function, ε is threshold.
    """
    signal = np.asarray(signal).flatten()

    # Embed
    embedded = _embed(signal, dimension, delay)
    n = len(embedded)

    # Compute distance matrix
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                d = np.linalg.norm(embedded[i] - embedded[j])
            elif metric == 'maximum':
                d = np.max(np.abs(embedded[i] - embedded[j]))
            elif metric == 'manhattan':
                d = np.sum(np.abs(embedded[i] - embedded[j]))
            else:
                d = np.linalg.norm(embedded[i] - embedded[j])

            dist[i, j] = d
            dist[j, i] = d

    # Auto threshold
    if threshold is None:
        off_diag = dist[np.triu_indices(n, k=1)]
        threshold = np.percentile(off_diag, threshold_percentile)

    # Binary recurrence matrix
    R = (dist <= threshold).astype(int)

    return R


def recurrence_rate(
    R: np.ndarray
) -> float:
    """
    Compute recurrence rate from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix

    Returns
    -------
    float
        Recurrence rate in [0, 1]

    Notes
    -----
    RR = (1/N²) Σ R_{ij}

    Measures density of recurrence points.
    High RR: system revisits same states often.
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n == 0:
        return 0.0

    return float(np.sum(R) / (n * n))


def determinism(
    R: np.ndarray,
    min_line: int = 2
) -> float:
    """
    Compute determinism from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum diagonal line length

    Returns
    -------
    float
        Determinism in [0, 1]

    Notes
    -----
    DET = Σ_{l≥l_min} l * P(l) / Σ R_{ij}

    where P(l) = number of diagonal lines of length l.

    High DET: deterministic dynamics (predictable)
    Low DET: stochastic dynamics (unpredictable)
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n < min_line:
        return 0.0

    # Count diagonal lines
    diagonal_lengths = []

    for k in range(-(n - min_line), n - min_line + 1):
        diag = np.diag(R, k)
        lengths = _run_lengths(diag)
        diagonal_lengths.extend([l for l in lengths if l >= min_line])

    total_recurrence = np.sum(R)
    if total_recurrence == 0:
        return 0.0

    diag_points = sum(diagonal_lengths)
    return float(diag_points / total_recurrence)


def laminarity(
    R: np.ndarray,
    min_line: int = 2
) -> float:
    """
    Compute laminarity from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum vertical line length

    Returns
    -------
    float
        Laminarity in [0, 1]

    Notes
    -----
    LAM = Σ_{v≥v_min} v * P(v) / Σ R_{ij}

    where P(v) = number of vertical lines of length v.

    High LAM: laminar states (system stuck in certain states)
    Low LAM: chaotic transitions between states
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n < min_line:
        return 0.0

    # Count vertical lines
    vertical_lengths = []

    for j in range(n):
        col = R[:, j]
        lengths = _run_lengths(col)
        vertical_lengths.extend([l for l in lengths if l >= min_line])

    total_recurrence = np.sum(R)
    if total_recurrence == 0:
        return 0.0

    vert_points = sum(vertical_lengths)
    return float(vert_points / total_recurrence)


def trapping_time(
    R: np.ndarray,
    min_line: int = 2
) -> float:
    """
    Compute trapping time from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum vertical line length

    Returns
    -------
    float
        Mean trapping time

    Notes
    -----
    TT = Σ v * P(v) / Σ P(v)

    Average length of vertical lines.
    Measures how long system is "trapped" in a state.
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n < min_line:
        return 0.0

    # Collect vertical line lengths
    lengths = []

    for j in range(n):
        col = R[:, j]
        runs = _run_lengths(col)
        lengths.extend([l for l in runs if l >= min_line])

    if len(lengths) == 0:
        return 0.0

    return float(np.mean(lengths))


def entropy_rqa(
    R: np.ndarray,
    min_line: int = 2
) -> float:
    """
    Compute diagonal line entropy from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum diagonal line length

    Returns
    -------
    float
        Shannon entropy of diagonal line distribution

    Notes
    -----
    ENTR = -Σ p(l) log(p(l))

    where p(l) = P(l) / Σ P(l).

    Measures complexity of deterministic structure.
    High ENTR: diverse diagonal line lengths
    Low ENTR: uniform diagonal line lengths
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n < min_line:
        return 0.0

    # Collect diagonal line lengths
    lengths = []

    for k in range(-(n - min_line), n - min_line + 1):
        diag = np.diag(R, k)
        runs = _run_lengths(diag)
        lengths.extend([l for l in runs if l >= min_line])

    if len(lengths) == 0:
        return 0.0

    # Distribution of lengths
    unique, counts = np.unique(lengths, return_counts=True)
    p = counts / np.sum(counts)

    # Shannon entropy
    entropy = -np.sum(p * np.log(p + 1e-10))

    return float(entropy)


def max_diagonal_line(
    R: np.ndarray,
    min_line: int = 2
) -> int:
    """
    Find maximum diagonal line length in recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum line length to consider

    Returns
    -------
    int
        Maximum diagonal line length (L_max)

    Notes
    -----
    L_max is inversely related to largest Lyapunov exponent.
    Long L_max: predictable dynamics
    Short L_max: sensitive dependence (chaos)
    """
    R = np.asarray(R)
    n = R.shape[0]

    if n < min_line:
        return 0

    max_length = 0

    for k in range(-(n - min_line), n - min_line + 1):
        diag = np.diag(R, k)
        runs = _run_lengths(diag)
        if runs:
            max_length = max(max_length, max(runs))

    return max_length if max_length >= min_line else 0


def divergence_rqa(
    R: np.ndarray,
    min_line: int = 2
) -> float:
    """
    Compute divergence from recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Recurrence matrix
    min_line : int
        Minimum diagonal line length

    Returns
    -------
    float
        Divergence = 1 / L_max

    Notes
    -----
    DIV is related to largest Lyapunov exponent.
    High DIV: sensitive dependence, chaos
    Low DIV: stable, predictable dynamics
    """
    L_max = max_diagonal_line(R, min_line)

    if L_max == 0:
        return np.inf

    return 1.0 / L_max


def rqa_metrics(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    threshold: float = None,
    threshold_percentile: float = 10.0,
    min_line: int = 2
) -> Dict[str, float]:
    """
    Compute all RQA metrics at once.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int
        Embedding dimension
    delay : int
        Time delay
    threshold : float, optional
        Recurrence threshold
    threshold_percentile : float
        Percentile for auto threshold
    min_line : int
        Minimum line length

    Returns
    -------
    dict
        Dictionary of all RQA metrics
    """
    R = recurrence_matrix(
        signal, dimension, delay, threshold, threshold_percentile
    )

    return {
        'recurrence_rate': recurrence_rate(R),
        'determinism': determinism(R, min_line),
        'laminarity': laminarity(R, min_line),
        'trapping_time': trapping_time(R, min_line),
        'entropy': entropy_rqa(R, min_line),
        'max_diagonal': max_diagonal_line(R, min_line),
        'divergence': divergence_rqa(R, min_line),
    }


# Helper functions

def _embed(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    if n_points <= 0:
        return np.zeros((0, dimension))
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]
    return embedded


def _run_lengths(arr: np.ndarray) -> list:
    """Find lengths of consecutive 1s in binary array."""
    if len(arr) == 0:
        return []

    lengths = []
    current_length = 0

    for val in arr:
        if val:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
            current_length = 0

    if current_length > 0:
        lengths.append(current_length)

    return lengths

"""
Recurrence Quantification Analysis (RQA)

Analyzes recurrence plots to characterize dynamical systems:
- Determinism: predictability of the system
- Laminarity: presence of laminar (intermittent) states
- Entropy: complexity of deterministic structure

References:
    Marwan, N., et al. (2007). "Recurrence plots for the analysis
    of complex systems"
"""

import numpy as np
from scipy.spatial.distance import cdist

from .reconstruction import embed_time_series, optimal_delay, optimal_embedding_dim


def recurrence_matrix(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    threshold: float = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute recurrence matrix R[i,j] = 1 if ||x(i) - x(j)|| < threshold.

    The recurrence matrix visualizes times when the system revisits
    similar states in phase space.

    Parameters
    ----------
    x : array
        Time series
    tau : int, optional
        Time delay for embedding
    dim : int, optional
        Embedding dimension
    threshold : float, optional
        Distance threshold for recurrence (default: 10% of max distance)
    metric : str
        Distance metric (default: 'euclidean')

    Returns
    -------
    R : array, shape (n, n)
        Binary recurrence matrix

    Notes
    -----
    Memory usage is O(n^2). For very long time series, consider
    subsampling or using sparse representations.
    """
    x = np.asarray(x).flatten()

    if tau is None:
        tau = optimal_delay(x, max_tau=min(50, len(x) // 10))
    if dim is None:
        dim = optimal_embedding_dim(x, tau, max_dim=min(6, len(x) // (4 * tau)))

    embedded = embed_time_series(x, tau, dim)

    # Compute pairwise distances
    distances = cdist(embedded, embedded, metric=metric)

    if threshold is None:
        # Use 10% of maximum distance
        threshold = 0.1 * np.max(distances)

    return (distances < threshold).astype(np.int8)


def rqa_metrics(R: np.ndarray, min_line_length: int = 2) -> dict:
    """
    Compute Recurrence Quantification Analysis metrics.

    Parameters
    ----------
    R : array
        Binary recurrence matrix
    min_line_length : int
        Minimum length for diagonal/vertical lines

    Returns
    -------
    dict with:
        recurrence_rate : float
            Density of recurrence points (RR)
            Probability that a state recurs

        determinism : float
            Fraction of recurrent points in diagonal lines (DET)
            High = predictable, Low = stochastic

        avg_diagonal_length : float
            Average length of diagonal structures (L)
            Related to mean prediction time

        max_diagonal_length : int
            Maximum diagonal line length (Lmax)
            Related to longest predictable period

        entropy : float
            Shannon entropy of diagonal line distribution (ENTR)
            Complexity of deterministic structure

        laminarity : float
            Fraction of recurrent points in vertical lines (LAM)
            High = intermittency/laminar phases

        trapping_time : float
            Average length of vertical structures (TT)
            Mean duration of laminar states

        max_vertical_length : int
            Maximum vertical line length (Vmax)

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 20*np.pi, 1000))
    >>> R = recurrence_matrix(x)
    >>> metrics = rqa_metrics(R)
    >>> metrics['determinism'] > 0.9  # Periodic = high determinism
    True
    """
    n = len(R)

    # Recurrence rate (exclude main diagonal)
    total_points = n * n - n  # Exclude diagonal
    recurrent_points = np.sum(R) - np.trace(R)  # Exclude diagonal
    rr = recurrent_points / max(1, total_points)

    # Find diagonal lines (exclude main diagonal)
    diag_lengths = []
    for k in range(-n + min_line_length, n - min_line_length + 1):
        if k == 0:
            continue  # Skip main diagonal
        diag = np.diag(R, k)
        lengths = _find_line_lengths(diag, min_line_length)
        diag_lengths.extend(lengths)

    # Find vertical lines
    vert_lengths = []
    for col in range(n):
        lengths = _find_line_lengths(R[:, col], min_line_length)
        vert_lengths.extend(lengths)

    # Determinism: points in diagonal lines / total recurrent points
    if recurrent_points > 0 and len(diag_lengths) > 0:
        points_in_diags = sum(diag_lengths)
        det = min(1.0, points_in_diags / recurrent_points)
        avg_diag = np.mean(diag_lengths)
        max_diag = max(diag_lengths)

        # Entropy of diagonal length distribution
        unique, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / counts.sum()
        entr = -np.sum(probs * np.log(probs + 1e-10))
    else:
        det = 0.0
        avg_diag = 0.0
        max_diag = 0
        entr = 0.0

    # Laminarity: points in vertical lines / total recurrent points
    if recurrent_points > 0 and len(vert_lengths) > 0:
        points_in_verts = sum(vert_lengths)
        lam = min(1.0, points_in_verts / recurrent_points)
        trap = np.mean(vert_lengths)
        max_vert = max(vert_lengths)
    else:
        lam = 0.0
        trap = 0.0
        max_vert = 0

    return {
        'recurrence_rate': float(rr),
        'determinism': float(det),
        'avg_diagonal_length': float(avg_diag),
        'max_diagonal_length': int(max_diag),
        'entropy': float(entr),
        'laminarity': float(lam),
        'trapping_time': float(trap),
        'max_vertical_length': int(max_vert),
    }


def _find_line_lengths(arr: np.ndarray, min_length: int = 2) -> list:
    """Find lengths of consecutive runs of 1s in array."""
    lengths = []
    current_length = 0

    for val in arr:
        if val == 1:
            current_length += 1
        else:
            if current_length >= min_length:
                lengths.append(current_length)
            current_length = 0

    # Don't forget last run
    if current_length >= min_length:
        lengths.append(current_length)

    return lengths


def cross_recurrence_matrix(
    x: np.ndarray,
    y: np.ndarray,
    tau: int = None,
    dim: int = None,
    threshold: float = None
) -> np.ndarray:
    """
    Compute cross-recurrence matrix between two time series.

    CR[i,j] = 1 if ||x(i) - y(j)|| < threshold

    Useful for detecting coupling and synchronization between systems.

    Parameters
    ----------
    x, y : array
        Time series (can be different lengths)
    tau, dim : int, optional
        Embedding parameters (applied to both)
    threshold : float, optional
        Distance threshold

    Returns
    -------
    CR : array
        Cross-recurrence matrix
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if tau is None:
        tau = max(optimal_delay(x), optimal_delay(y))
    if dim is None:
        dim = max(optimal_embedding_dim(x, tau), optimal_embedding_dim(y, tau))

    embedded_x = embed_time_series(x, tau, dim)
    embedded_y = embed_time_series(y, tau, dim)

    distances = cdist(embedded_x, embedded_y)

    if threshold is None:
        threshold = 0.1 * np.max(distances)

    return (distances < threshold).astype(np.int8)


def joint_recurrence_matrix(
    x: np.ndarray,
    y: np.ndarray,
    tau: int = None,
    dim: int = None,
    threshold: float = None
) -> np.ndarray:
    """
    Compute joint recurrence matrix.

    JR[i,j] = 1 if x recurs AND y recurs at the same times.

    Detects times when both systems simultaneously revisit similar states.

    Parameters
    ----------
    x, y : array
        Time series (must be same length)
    tau, dim : int, optional
        Embedding parameters
    threshold : float, optional
        Distance threshold

    Returns
    -------
    JR : array
        Joint recurrence matrix
    """
    R_x = recurrence_matrix(x, tau, dim, threshold)
    R_y = recurrence_matrix(y, tau, dim, threshold)

    # Use shorter matrix if lengths differ
    n = min(len(R_x), len(R_y))

    return R_x[:n, :n] & R_y[:n, :n]

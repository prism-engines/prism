"""
Point Cloud Construction for Topological Analysis

Methods for constructing point clouds from time series data
for subsequent persistent homology computation.
"""

import numpy as np
from typing import Dict, Optional


def time_delay_embedding(
    x: np.ndarray,
    dim: int,
    tau: int
) -> np.ndarray:
    """
    Construct point cloud via time-delay embedding (Takens' theorem).

    Parameters
    ----------
    x : array, shape (n_samples,)
        Scalar time series
    dim : int
        Embedding dimension
    tau : int
        Time delay (in samples)

    Returns
    -------
    embedded : array, shape (n_samples - (dim-1)*tau, dim)
        Embedded trajectory in reconstructed phase space
    """
    x = np.asarray(x).flatten()
    n = len(x) - (dim - 1) * tau

    if n <= 0:
        raise ValueError(f"Time series too short for dim={dim}, tau={tau}")

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau : i * tau + n]
    return embedded


def sliding_window_embedding(
    x: np.ndarray,
    window_size: int,
    step: int = 1
) -> np.ndarray:
    """
    Construct point cloud where each point is a window of the signal.

    Good for periodic/quasi-periodic signals where the window captures
    one or more complete cycles.

    Parameters
    ----------
    x : array
        Time series
    window_size : int
        Size of each window
    step : int
        Step between windows

    Returns
    -------
    embedded : array, shape (n_windows, window_size)
        Each row is a window of the signal
    """
    x = np.asarray(x).flatten()
    n_windows = (len(x) - window_size) // step + 1

    if n_windows <= 0:
        raise ValueError(f"Time series too short for window_size={window_size}")

    embedded = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        embedded[i] = x[i * step : i * step + window_size]
    return embedded


def multivariate_point_cloud(
    signals: Dict[str, np.ndarray],
    method: str = 'direct'
) -> np.ndarray:
    """
    Construct point cloud from multiple signals.

    Parameters
    ----------
    signals : dict
        {signal_name: time_series_array}
    method : str
        'direct': Each time point is a point in R^n_signals
        'concatenated_embedding': Embed each signal, concatenate

    Returns
    -------
    point_cloud : array, shape (n_points, n_dims)
    """
    signal_list = list(signals.values())
    n_samples = min(len(s) for s in signal_list)

    if method == 'direct':
        return np.column_stack([s[:n_samples] for s in signal_list])

    elif method == 'concatenated_embedding':
        embeddings = []
        for s in signal_list:
            s = s[:n_samples]
            if len(s) > 30:
                emb = time_delay_embedding(s, dim=3, tau=max(1, len(s) // 30))
                embeddings.append(emb)
        if not embeddings:
            return np.column_stack([s[:n_samples] for s in signal_list])
        min_len = min(len(e) for e in embeddings)
        return np.hstack([e[:min_len] for e in embeddings])

    else:
        raise ValueError(f"Unknown method: {method}")


def subsample_point_cloud(
    point_cloud: np.ndarray,
    n_landmarks: int,
    method: str = 'random'
) -> np.ndarray:
    """
    Subsample a point cloud for computational efficiency.

    Parameters
    ----------
    point_cloud : array
        Original point cloud
    n_landmarks : int
        Number of points to keep
    method : str
        'random': Random subsampling
        'maxmin': Greedy max-min subsampling (better coverage)

    Returns
    -------
    subsampled : array
    """
    n = len(point_cloud)

    if n <= n_landmarks:
        return point_cloud

    if method == 'random':
        indices = np.random.choice(n, n_landmarks, replace=False)
        return point_cloud[indices]

    elif method == 'maxmin':
        # Greedy max-min subsampling
        indices = [np.random.randint(n)]

        for _ in range(1, n_landmarks):
            # Find point furthest from current landmarks
            dists = np.full(n, np.inf)
            for idx in indices:
                d = np.linalg.norm(point_cloud - point_cloud[idx], axis=1)
                dists = np.minimum(dists, d)
            dists[indices] = -np.inf  # Exclude already selected
            indices.append(np.argmax(dists))

        return point_cloud[indices]

    else:
        raise ValueError(f"Unknown method: {method}")

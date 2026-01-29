"""
Transfer Entropy Computation

T(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

Measures directional information flow from X to Y.
"""

import numpy as np
from typing import Tuple, List, Dict


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    history_length: int = 1,
    bins: int = 8
) -> float:
    """
    Compute transfer entropy T(source -> target).

    T(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Parameters
    ----------
    source : array
        Potential cause signal X
    target : array
        Potential effect signal Y
    lag : int
        Prediction horizon
    history_length : int
        How many past values to condition on
    bins : int
        Discretization bins

    Returns
    -------
    te : float
        Transfer entropy in bits
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    source, target = source[:n], target[:n]

    # Need enough data for lagged variables
    start = history_length
    end = n - lag

    if end <= start + 10:  # Need at least 10 samples
        return 0.0

    # Build lagged variables
    y_future = target[start + lag : end + lag]
    y_past = np.column_stack([target[start - i : end - i] for i in range(1, history_length + 1)])
    x_past = np.column_stack([source[start - i : end - i] for i in range(1, history_length + 1)])

    # Discretize
    def discretize(arr, bins):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        result = np.zeros_like(arr, dtype=int)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            edges = np.linspace(col.min() - 1e-10, col.max() + 1e-10, bins + 1)
            result[:, j] = np.clip(np.digitize(col, edges[:-1]) - 1, 0, bins - 1)
        return result

    y_future_d = discretize(y_future, bins).flatten()
    y_past_d = discretize(y_past, bins)
    x_past_d = discretize(x_past, bins)

    # Convert multi-column to single index
    def to_index(arr, bins):
        if arr.ndim == 1:
            return arr
        idx = np.zeros(len(arr), dtype=np.int64)
        for j in range(arr.shape[1]):
            idx = idx * bins + arr[:, j]
        return idx

    y_past_idx = to_index(y_past_d, bins)
    x_past_idx = to_index(x_past_d, bins)
    xy_past_idx = y_past_idx * (bins ** history_length) + x_past_idx

    # Count probabilities
    def joint_counts(idx1, idx2, n1, n2):
        counts = np.zeros((n1, n2), dtype=int)
        for i1, i2 in zip(idx1, idx2):
            if 0 <= i1 < n1 and 0 <= i2 < n2:
                counts[i1, i2] += 1
        return counts

    n_y = bins
    n_ypast = bins ** history_length
    n_xypast = bins ** (2 * history_length)

    # P(Y_future, Y_past)
    counts_yf_yp = joint_counts(y_future_d, y_past_idx, n_y, n_ypast)
    p_yf_yp = counts_yf_yp / counts_yf_yp.sum()

    # P(Y_past)
    p_yp = p_yf_yp.sum(axis=0)

    # P(Y_future, Y_past, X_past) - need 3D
    # Simplify: use combined xy_past index
    counts_yf_xyp = joint_counts(y_future_d, xy_past_idx, n_y, n_xypast)
    p_yf_xyp = counts_yf_xyp / max(1, counts_yf_xyp.sum())

    # P(Y_past, X_past)
    p_xyp = p_yf_xyp.sum(axis=0)

    # H(Y_future, Y_past)
    p_flat = p_yf_yp.flatten()
    h_yf_yp = -np.sum(p_flat[p_flat > 0] * np.log2(p_flat[p_flat > 0]))

    # H(Y_past)
    h_yp = -np.sum(p_yp[p_yp > 0] * np.log2(p_yp[p_yp > 0]))

    # H(Y_future | Y_past)
    h_yf_given_yp = h_yf_yp - h_yp

    # H(Y_future, Y_past, X_past)
    p_flat2 = p_yf_xyp.flatten()
    h_yf_xyp = -np.sum(p_flat2[p_flat2 > 0] * np.log2(p_flat2[p_flat2 > 0]))

    # H(Y_past, X_past)
    h_xyp = -np.sum(p_xyp[p_xyp > 0] * np.log2(p_xyp[p_xyp > 0]))

    # H(Y_future | Y_past, X_past)
    h_yf_given_xyp = h_yf_xyp - h_xyp

    # Transfer entropy
    te = h_yf_given_yp - h_yf_given_xyp

    return max(0.0, te)


def transfer_entropy_matrix(
    signals: Dict[str, np.ndarray],
    lag: int = 1,
    history_length: int = 1,
    bins: int = 8
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise transfer entropy matrix.

    Parameters
    ----------
    signals : dict
        {signal_name: time_series}
    lag : int
        Prediction horizon
    history_length : int
        History to condition on
    bins : int
        Discretization bins

    Returns
    -------
    te_matrix : array, shape (n_signals, n_signals)
        te_matrix[i, j] = T(signal_i -> signal_j)
    signal_names : list
        Names corresponding to matrix indices
    """
    names = list(signals.keys())
    n = len(names)

    te_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                te_matrix[i, j] = transfer_entropy(
                    signals[name_i],
                    signals[name_j],
                    lag=lag,
                    history_length=history_length,
                    bins=bins
                )

    return te_matrix, names


def effective_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    history_length: int = 1,
    bins: int = 8,
    n_surrogates: int = 20
) -> Tuple[float, float, bool]:
    """
    Compute effective transfer entropy with surrogate-based significance.

    Parameters
    ----------
    source, target : array
        Time series
    lag, history_length, bins : int
        TE parameters
    n_surrogates : int
        Number of surrogate tests

    Returns
    -------
    te : float
        Transfer entropy
    te_eff : float
        Effective TE (= te - mean(surrogates))
    significant : bool
        Whether TE is significant (p < 0.05)
    """
    te = transfer_entropy(source, target, lag, history_length, bins)

    # Surrogate test: shuffle source to break temporal structure
    surrogate_tes = []
    for _ in range(n_surrogates):
        source_shuffled = np.random.permutation(source)
        te_surr = transfer_entropy(source_shuffled, target, lag, history_length, bins)
        surrogate_tes.append(te_surr)

    mean_surr = np.mean(surrogate_tes)
    std_surr = np.std(surrogate_tes)

    te_eff = te - mean_surr
    significant = te > mean_surr + 2 * std_surr  # ~95% confidence

    return te, te_eff, significant

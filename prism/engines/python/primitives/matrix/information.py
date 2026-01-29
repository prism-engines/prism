"""
Matrix Information Primitives (61-63)

Mutual information, transfer entropy, and Granger causality matrices.
"""

import numpy as np
from typing import Tuple


def mutual_information_matrix(
    signals: np.ndarray,
    n_bins: int = 10
) -> np.ndarray:
    """
    Compute mutual information between all signal pairs.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_signals)
    n_bins : int
        Number of bins for histogram estimation

    Returns
    -------
    np.ndarray
        Mutual information matrix of shape (n_signals, n_signals)
        mi[i,j] = mutual information between signal i and j
        Units: bits (using log2)

    Notes
    -----
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Properties:
    - Symmetric: mi[i,j] = mi[j,i]
    - Non-negative: mi[i,j] >= 0
    - mi[i,i] = H(X) (entropy of signal i)
    - mi[i,j] = 0 iff signals are independent

    Physical interpretation:
    Unlike correlation, MI captures NONLINEAR relationships.
    High MI means knowing one signal tells you about the other,
    even if the relationship is not linear.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape
    mi_matrix = np.zeros((n_signals, n_signals))

    # Discretize signals
    binned = np.zeros((n_samples, n_signals), dtype=int)
    for i in range(n_signals):
        sig = signals[:, i]
        edges = np.linspace(np.nanmin(sig), np.nanmax(sig) + 1e-10, n_bins + 1)
        binned[:, i] = np.clip(np.digitize(sig, edges) - 1, 0, n_bins - 1)

    def entropy(x):
        """Compute entropy of discrete variable."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def joint_entropy(x, y):
        """Compute joint entropy of two discrete variables."""
        xy = np.column_stack([x, y])
        _, counts = np.unique(xy, axis=0, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))

    # Compute pairwise MI
    for i in range(n_signals):
        h_i = entropy(binned[:, i])
        mi_matrix[i, i] = h_i  # Self MI = entropy

        for j in range(i + 1, n_signals):
            h_j = entropy(binned[:, j])
            h_ij = joint_entropy(binned[:, i], binned[:, j])

            mi = h_i + h_j - h_ij
            mi_matrix[i, j] = max(0, mi)
            mi_matrix[j, i] = mi_matrix[i, j]

    return mi_matrix


def transfer_entropy_matrix(
    signals: np.ndarray,
    lag: int = 1,
    n_bins: int = 8
) -> np.ndarray:
    """
    Compute transfer entropy between all signal pairs.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_signals)
    lag : int
        Time lag for source signal
    n_bins : int
        Number of bins for histogram estimation

    Returns
    -------
    np.ndarray
        Transfer entropy matrix of shape (n_signals, n_signals)
        te[i,j] = transfer entropy from signal i TO signal j
        NOTE: NOT symmetric! te[i,j] != te[j,i] in general

    Notes
    -----
    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    "How much does knowing X's past reduce uncertainty about
    Y's future, beyond what Y's own past tells us?"

    Properties:
    - NOT symmetric: direction matters
    - Non-negative: te[i,j] >= 0
    - te[i,j] > te[j,i] suggests i drives j

    Physical interpretation:
    This is a CAUSAL matrix. High te[i,j] means signal i
    provides information about signal j's future.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape
    te_matrix = np.zeros((n_signals, n_signals))

    if n_samples < lag + 10:
        return te_matrix

    # Discretize signals
    binned = np.zeros((n_samples, n_signals), dtype=int)
    for i in range(n_signals):
        sig = signals[:, i]
        edges = np.linspace(np.nanmin(sig), np.nanmax(sig) + 1e-10, n_bins + 1)
        binned[:, i] = np.clip(np.digitize(sig, edges) - 1, 0, n_bins - 1)

    def entropy_hist(*args):
        """Joint entropy from multiple discrete variables."""
        combined = np.column_stack(args)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        probs = counts / len(args[0])
        return -np.sum(probs * np.log2(probs + 1e-10))

    n = n_samples - lag - 1

    for source in range(n_signals):
        for target in range(n_signals):
            if source == target:
                continue

            # Build time series
            y_future = binned[lag + 1:, target][:n]
            y_past = binned[lag:-1, target][:n]
            x_past = binned[:-lag-1, source][:n]

            # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
            # H(A|B) = H(A,B) - H(B)
            h_yf_yp = entropy_hist(y_future, y_past) - entropy_hist(y_past)
            h_yf_yp_xp = entropy_hist(y_future, y_past, x_past) - entropy_hist(y_past, x_past)

            te_matrix[source, target] = max(0, h_yf_yp - h_yf_yp_xp)

    return te_matrix


def granger_matrix(
    signals: np.ndarray,
    max_lag: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Granger causality between all signal pairs.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_signals)
    max_lag : int
        Maximum lag to test

    Returns
    -------
    f_matrix : np.ndarray
        F-statistic matrix of shape (n_signals, n_signals)
        f[i,j] = F-statistic for "signal i Granger-causes signal j"
    p_matrix : np.ndarray
        p-value matrix of shape (n_signals, n_signals)
        p[i,j] = p-value for the test

    Notes
    -----
    Tests whether past values of signal i help predict signal j
    beyond signal j's own past.

    Properties:
    - NOT symmetric: direction matters
    - Low p-value (< 0.05) = significant causal relationship
    - F-statistic magnitude indicates strength

    Physical interpretation:
    Linear causal relationships. Use alongside transfer entropy:
    - Granger: linear causality
    - TE: nonlinear causality
    """
    from scipy import stats

    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape
    f_matrix = np.zeros((n_signals, n_signals))
    p_matrix = np.ones((n_signals, n_signals))

    if n_samples < max_lag + 10:
        return f_matrix, p_matrix

    for source in range(n_signals):
        for target in range(n_signals):
            if source == target:
                continue

            try:
                f_stat, p_value, best_lag = _granger_test(
                    signals[:, source],
                    signals[:, target],
                    max_lag
                )
                f_matrix[source, target] = f_stat
                p_matrix[source, target] = p_value
            except Exception:
                # If test fails, leave as default
                pass

    return f_matrix, p_matrix


def _granger_test(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int
) -> Tuple[float, float, int]:
    """Run Granger causality test from source to target."""
    from scipy import stats

    n = len(target)

    best_f = 0.0
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        # Restricted model: Y ~ Y_past
        y = target[max_lag:]
        X_r = np.column_stack([
            np.ones(len(y)),
            *[target[max_lag - i : n - i] for i in range(1, lag + 1)]
        ])

        # Unrestricted model: Y ~ Y_past + X_past
        X_u = np.column_stack([
            X_r,
            *[source[max_lag - i : n - i] for i in range(1, lag + 1)]
        ])

        if len(y) < X_u.shape[1] + 2:
            continue

        try:
            # Fit models
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]

            ssr_r = np.sum((y - X_r @ beta_r) ** 2)
            ssr_u = np.sum((y - X_u @ beta_u) ** 2)

            # F-test
            df1 = lag
            df2 = len(y) - X_u.shape[1]

            if df2 <= 0 or ssr_u <= 0:
                continue

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_value < best_p:
                best_p = p_value
                best_f = f_stat
                best_lag = lag

        except Exception:
            continue

    return best_f, best_p, best_lag

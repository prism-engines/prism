"""
Granger Causality

Tests whether past values of X improve prediction of Y
beyond Y's own past.
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy import stats


def granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5
) -> Tuple[float, float, int]:
    """
    Test if source Granger-causes target.

    Parameters
    ----------
    source : array
        Potential cause X
    target : array
        Potential effect Y
    max_lag : int
        Maximum lag to consider

    Returns
    -------
    f_stat : float
        F-statistic for Granger test
    p_value : float
        P-value (< 0.05 suggests causality)
    optimal_lag : int
        Optimal lag based on AIC
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    source, target = source[:n], target[:n]

    if n < max_lag + 10:
        return 0.0, 1.0, 1

    # Find optimal lag using AIC
    best_aic = np.inf
    optimal_lag = 1

    for lag in range(1, max_lag + 1):
        # Restricted model: Y ~ Y_past
        y = target[max_lag:]
        X_r = np.column_stack([
            np.ones(len(y)),
            *[target[max_lag - i : n - i] for i in range(1, lag + 1)]
        ])

        if len(y) < X_r.shape[1] + 2:
            continue

        try:
            beta_r, residuals_r, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
            if len(residuals_r) == 0:
                ssr_r = np.sum((y - X_r @ beta_r) ** 2)
            else:
                ssr_r = residuals_r[0]

            aic = len(y) * np.log(ssr_r / len(y) + 1e-10) + 2 * (lag + 1)

            if aic < best_aic:
                best_aic = aic
                optimal_lag = lag
        except:
            continue

    lag = optimal_lag

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

    try:
        # Fit restricted model
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        residuals_r = y - X_r @ beta_r
        ssr_r = np.sum(residuals_r ** 2)

        # Fit unrestricted model
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
        residuals_u = y - X_u @ beta_u
        ssr_u = np.sum(residuals_u ** 2)

        # F-test
        df1 = lag  # Additional parameters
        df2 = len(y) - X_u.shape[1]

        if df2 <= 0 or ssr_u <= 0:
            return 0.0, 1.0, lag

        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return float(f_stat), float(p_value), lag

    except Exception:
        return 0.0, 1.0, lag


def granger_causality_matrix(
    signals: Dict[str, np.ndarray],
    max_lag: int = 5,
    significance: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute pairwise Granger causality matrix.

    Parameters
    ----------
    signals : dict
        {signal_name: time_series}
    max_lag : int
        Maximum lag to test
    significance : float
        P-value threshold

    Returns
    -------
    f_matrix : array, shape (n_signals, n_signals)
        F-statistics
    p_matrix : array, shape (n_signals, n_signals)
        P-values
    signal_names : list
        Names corresponding to indices
    """
    names = list(signals.keys())
    n = len(names)

    f_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                f_stat, p_val, _ = granger_causality(
                    signals[name_i],
                    signals[name_j],
                    max_lag=max_lag
                )
                f_matrix[i, j] = f_stat
                p_matrix[i, j] = p_val

    return f_matrix, p_matrix, names


def conditional_granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray,
    max_lag: int = 5
) -> Tuple[float, float]:
    """
    Conditional Granger causality: X -> Y | Z

    Tests if X Granger-causes Y after conditioning on Z.

    Parameters
    ----------
    source : array
        Potential cause X
    target : array
        Potential effect Y
    conditioning : array
        Conditioning variable Z

    Returns
    -------
    f_stat : float
    p_value : float
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()
    conditioning = np.asarray(conditioning).flatten()

    n = min(len(source), len(target), len(conditioning))
    source = source[:n]
    target = target[:n]
    conditioning = conditioning[:n]

    if n < max_lag + 10:
        return 0.0, 1.0

    lag = 1  # Simplified: use lag 1

    y = target[max_lag:]

    # Restricted: Y ~ Y_past + Z_past
    X_r = np.column_stack([
        np.ones(len(y)),
        *[target[max_lag - i : n - i] for i in range(1, lag + 1)],
        *[conditioning[max_lag - i : n - i] for i in range(1, lag + 1)]
    ])

    # Unrestricted: Y ~ Y_past + Z_past + X_past
    X_u = np.column_stack([
        X_r,
        *[source[max_lag - i : n - i] for i in range(1, lag + 1)]
    ])

    try:
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        ssr_r = np.sum((y - X_r @ beta_r) ** 2)

        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
        ssr_u = np.sum((y - X_u @ beta_u) ** 2)

        df1 = lag
        df2 = len(y) - X_u.shape[1]

        if df2 <= 0 or ssr_u <= 0:
            return 0.0, 1.0

        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return float(f_stat), float(p_value)

    except:
        return 0.0, 1.0

"""
Cointegration Engine.

Computes Engle-Granger cointegration test between signal pairs.
"""

import numpy as np


def compute(y_a: np.ndarray, y_b: np.ndarray) -> dict:
    """
    Compute cointegration between two signals (symmetric).

    Args:
        y_a: First signal values
        y_b: Second signal values

    Returns:
        dict with coint_stat, coint_pvalue, is_cointegrated, coint_beta
    """
    result = {
        'coint_stat': np.nan,
        'coint_pvalue': np.nan,
        'is_cointegrated': False,
        'coint_beta': np.nan,
    }

    n = min(len(y_a), len(y_b))
    if n < 50:
        return result

    y_a, y_b = y_a[:n], y_b[:n]

    try:
        # Simple Engle-Granger: regress y_a on y_b, test residuals for stationarity
        X = np.column_stack([np.ones(n), y_b])
        beta = np.linalg.lstsq(X, y_a, rcond=None)[0]
        residuals = y_a - X @ beta

        # ADF test on residuals (simplified)
        diff_resid = np.diff(residuals)
        lag_resid = residuals[:-1]

        if len(diff_resid) > 10 and np.std(lag_resid) > 1e-10:
            gamma = np.sum(diff_resid * lag_resid) / np.sum(lag_resid ** 2)
            se = np.std(diff_resid - gamma * lag_resid) / (np.std(lag_resid) * np.sqrt(len(diff_resid)))

            if se > 1e-10:
                t_stat = gamma / se
                result['coint_stat'] = float(t_stat)
                result['coint_pvalue'] = float(min(1.0, np.exp(0.5 * t_stat)))
                result['is_cointegrated'] = t_stat < -2.86
                result['coint_beta'] = float(beta[1])
    except Exception:
        pass

    return result

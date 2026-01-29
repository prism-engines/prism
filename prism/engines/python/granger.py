"""
Granger Causality Engine.

Computes Granger causality between signal pairs.
"""

import numpy as np
from scipy import stats


def compute(y_source: np.ndarray, y_target: np.ndarray, max_lag: int = 10) -> dict:
    """
    Compute Granger causality from source to target.

    Args:
        y_source: Source signal values
        y_target: Target signal values
        max_lag: Maximum lag to test (default 10)

    Returns:
        dict with granger_fstat, granger_pvalue, granger_lag, granger_significant
    """
    n = min(len(y_source), len(y_target))
    if n < 50:
        return {
            'granger_fstat': np.nan,
            'granger_pvalue': np.nan,
            'granger_lag': 0,
            'granger_significant': False
        }

    y_source, y_target = y_source[:n], y_target[:n]
    max_lag = min(max_lag, n // 5)

    best_f, best_p, best_lag = 0, 1.0, 1
    for lag in range(1, max_lag + 1):
        try:
            X_r = np.column_stack([y_target[i:n-lag+i] for i in range(lag)])
            y_dep = y_target[lag:]
            if len(y_dep) < lag + 5:
                continue
            X_u = np.column_stack([X_r] + [y_source[i:n-lag+i] for i in range(lag)])

            beta_r = np.linalg.lstsq(X_r, y_dep, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y_dep, rcond=None)[0]

            ssr_r = np.sum((y_dep - X_r @ beta_r) ** 2)
            ssr_u = np.sum((y_dep - X_u @ beta_u) ** 2)

            df1, df2 = lag, len(y_dep) - 2 * lag - 1
            if df2 > 0 and ssr_u > 0:
                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                if f_stat > best_f:
                    best_f, best_p, best_lag = f_stat, p_value, lag
        except Exception:
            continue

    return {
        'granger_fstat': float(best_f),
        'granger_pvalue': float(best_p),
        'granger_lag': int(best_lag),
        'granger_significant': bool(best_p < 0.05)
    }

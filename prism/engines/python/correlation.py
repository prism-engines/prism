"""
Correlation Engine.

Computes Pearson correlation between signal pairs.
"""

import numpy as np


def compute(y_a: np.ndarray, y_b: np.ndarray) -> dict:
    """
    Compute Pearson correlation between two signals.

    Args:
        y_a: First signal values
        y_b: Second signal values

    Returns:
        dict with 'correlation', 'correlation_abs', 'n_points'
    """
    n = min(len(y_a), len(y_b))
    if n < 10:
        return {
            'correlation': np.nan,
            'correlation_abs': np.nan,
            'n_points': n
        }

    y_a, y_b = y_a[:n], y_b[:n]

    try:
        corr = np.corrcoef(y_a, y_b)[0, 1]
        return {
            'correlation': float(corr) if not np.isnan(corr) else np.nan,
            'correlation_abs': float(abs(corr)) if not np.isnan(corr) else np.nan,
            'n_points': n
        }
    except Exception:
        return {
            'correlation': np.nan,
            'correlation_abs': np.nan,
            'n_points': n
        }

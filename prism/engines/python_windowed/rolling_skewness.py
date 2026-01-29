"""
Rolling Skewness Engine.

Skewness over sliding window.
"""

import numpy as np
from scipy import stats


def compute(y: np.ndarray, window: int = 100) -> dict:
    """
    Compute rolling skewness.

    Args:
        y: Signal values
        window: Window size

    Returns:
        dict with 'rolling_skewness' array
    """
    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_skewness': result}

    for i in range(window, n):
        chunk = y[i-window:i]
        result[i] = stats.skew(chunk)

    return {'rolling_skewness': result}

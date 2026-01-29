"""
Rolling Pulsation Engine.

Pulsation index over sliding window.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 100) -> dict:
    """
    Compute rolling pulsation index.

    Args:
        y: Signal values
        window: Window size

    Returns:
        dict with 'rolling_pulsation' array
    """
    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_pulsation': result}

    for i in range(window, n):
        chunk = y[i-window:i]
        mean_val = np.mean(chunk)
        range_val = np.max(chunk) - np.min(chunk)
        result[i] = range_val / (abs(mean_val) + 1e-10)

    return {'rolling_pulsation': result}

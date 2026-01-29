"""
Rolling Range Engine.

Max - Min over sliding window.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 100) -> dict:
    """
    Compute rolling range.

    Args:
        y: Signal values
        window: Window size

    Returns:
        dict with 'rolling_range', 'rolling_max', 'rolling_min' arrays
    """
    n = len(y)
    rolling_range = np.full(n, np.nan)
    rolling_max = np.full(n, np.nan)
    rolling_min = np.full(n, np.nan)

    if n < window:
        return {
            'rolling_range': rolling_range,
            'rolling_max': rolling_max,
            'rolling_min': rolling_min
        }

    for i in range(window, n):
        chunk = y[i-window:i]
        max_val = np.max(chunk)
        min_val = np.min(chunk)
        rolling_max[i] = max_val
        rolling_min[i] = min_val
        rolling_range[i] = max_val - min_val

    return {
        'rolling_range': rolling_range,
        'rolling_max': rolling_max,
        'rolling_min': rolling_min
    }

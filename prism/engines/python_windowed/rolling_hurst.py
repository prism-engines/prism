"""
Rolling Hurst Engine.

Computes Hurst exponent over rolling windows.
"""

import numpy as np
from ..python import hurst


def compute(y: np.ndarray, window: int = 100, stride: int = 1) -> dict:
    """
    Compute rolling Hurst exponent.

    Args:
        y: Signal values
        window: Window size
        stride: Step size between windows

    Returns:
        dict with 'rolling_hurst' array
    """
    n = len(y)
    if n < window:
        return {'rolling_hurst': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        h = hurst.compute(chunk)
        # Assign to all points in this window (or just the last)
        result[i + window - 1] = h['hurst']

    return {'rolling_hurst': result}

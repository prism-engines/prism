"""
Rolling Kurtosis Engine.

Computes kurtosis over rolling windows.
"""

import numpy as np
from scipy import stats


def compute(y: np.ndarray, window: int = 50, stride: int = 1) -> dict:
    """
    Compute rolling kurtosis.

    Args:
        y: Signal values
        window: Window size
        stride: Step size between windows

    Returns:
        dict with 'rolling_kurtosis' array
    """
    n = len(y)
    if n < window:
        return {'rolling_kurtosis': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = stats.kurtosis(chunk, fisher=True)

    return {'rolling_kurtosis': result}

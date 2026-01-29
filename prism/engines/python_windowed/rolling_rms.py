"""
Rolling RMS Engine.

Computes RMS over rolling windows.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 50, stride: int = 1) -> dict:
    """
    Compute rolling RMS.

    Args:
        y: Signal values
        window: Window size
        stride: Step size between windows

    Returns:
        dict with 'rolling_rms' array
    """
    n = len(y)
    if n < window:
        return {'rolling_rms': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = np.sqrt(np.mean(chunk ** 2))

    return {'rolling_rms': result}

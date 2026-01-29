"""
Rolling Crest Factor Engine.

Crest factor over sliding window - detects impulsive events.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 100) -> dict:
    """
    Compute rolling crest factor.

    Args:
        y: Signal values
        window: Window size

    Returns:
        dict with 'rolling_crest_factor' array
    """
    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_crest_factor': result}

    for i in range(window, n):
        chunk = y[i-window:i]
        rms = np.sqrt(np.mean(chunk ** 2))
        peak = np.max(np.abs(chunk))
        result[i] = peak / (rms + 1e-10)

    return {'rolling_crest_factor': result}

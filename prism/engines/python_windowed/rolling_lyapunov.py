"""
Rolling Lyapunov Engine.

Lyapunov exponent over sliding window.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 500, lag: int = 1) -> dict:
    """
    Compute rolling Lyapunov exponent estimate.

    Args:
        y: Signal values
        window: Window size
        lag: Time lag for divergence calculation

    Returns:
        dict with 'rolling_lyapunov' array
    """
    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_lyapunov': result}

    for i in range(window, n):
        chunk = y[i-window:i]

        # Simple divergence-based estimate
        diffs = np.abs(np.diff(chunk))
        if len(diffs) > lag:
            initial = diffs[:-lag] + 1e-10
            final = diffs[lag:]
            ratios = final / initial
            ratios = ratios[ratios > 0]
            if len(ratios) > 0:
                result[i] = np.mean(np.log(ratios)) / lag

    return {'rolling_lyapunov': result}

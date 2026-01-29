"""
Rolling Volatility Engine.

Computes volatility (realized variance) over rolling windows.
"""

import numpy as np


def compute(y: np.ndarray, window: int = 50, stride: int = 1) -> dict:
    """
    Compute rolling volatility.

    Args:
        y: Signal values
        window: Window size
        stride: Step size between windows

    Returns:
        dict with 'rolling_volatility' array
    """
    n = len(y)
    if n < window:
        return {'rolling_volatility': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        returns = np.diff(chunk)
        if len(returns) > 0:
            result[i + window - 1] = np.std(returns)

    return {'rolling_volatility': result}

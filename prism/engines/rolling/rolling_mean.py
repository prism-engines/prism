"""
Rolling Mean Engine.

Computes mean over rolling windows.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling mean.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_mean' array
    """
    params = params or {}
    window = params.get('window', 50)
    # Cheap engine (O(n)): stride=1 is acceptable default
    stride = params.get('stride', 1)

    n = len(y)
    if n < window:
        return {'rolling_mean': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = np.mean(chunk)

    return {'rolling_mean': result}

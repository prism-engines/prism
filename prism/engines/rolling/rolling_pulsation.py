"""
Rolling Pulsation Engine.

Pulsation index over sliding window.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling pulsation index.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_pulsation' array
    """
    params = params or {}
    window = params.get('window', 100)
    # Cheap engine (O(n)): stride=1 is acceptable default
    stride = params.get('stride', 1)

    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_pulsation': result}

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        mean_val = np.mean(chunk)
        range_val = np.max(chunk) - np.min(chunk)
        result[i + window - 1] = range_val / (abs(mean_val) + 1e-10)

    return {'rolling_pulsation': result}

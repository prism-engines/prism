"""
Rolling Range Engine.

Max - Min over sliding window.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling range.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_range', 'rolling_max', 'rolling_min' arrays
    """
    params = params or {}
    window = params.get('window', 100)
    # Cheap engine (O(n)): stride=1 is acceptable default
    stride = params.get('stride', 1)

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

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        max_val = np.max(chunk)
        min_val = np.min(chunk)
        rolling_max[i + window - 1] = max_val
        rolling_min[i + window - 1] = min_val
        rolling_range[i + window - 1] = max_val - min_val

    return {
        'rolling_range': rolling_range,
        'rolling_max': rolling_max,
        'rolling_min': rolling_min
    }

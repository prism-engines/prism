"""
Rolling Skewness Engine.

Skewness over sliding window.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from scipy import stats


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling skewness.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_skewness' array
    """
    params = params or {}
    window = params.get('window', 100)
    # Moderate cost: default to window//4 if not specified
    stride = params.get('stride', max(1, window // 4))

    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_skewness': result}

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = stats.skew(chunk)

    return {'rolling_skewness': result}

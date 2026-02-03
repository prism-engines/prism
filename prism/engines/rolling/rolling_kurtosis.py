"""
Rolling Kurtosis Engine.

Computes kurtosis over rolling windows.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from scipy import stats


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling kurtosis.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_kurtosis' array
    """
    params = params or {}
    window = params.get('window', 50)
    # Moderate cost: default to window//4 if not specified
    stride = params.get('stride', max(1, window // 4))

    n = len(y)
    if n < window:
        return {'rolling_kurtosis': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = stats.kurtosis(chunk, fisher=True)

    return {'rolling_kurtosis': result}

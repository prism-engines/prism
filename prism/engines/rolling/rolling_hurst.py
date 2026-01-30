"""
Rolling Hurst Engine.

Computes Hurst exponent over rolling windows.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from ..signal import hurst


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling Hurst exponent.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size (required for meaningful results)
            - stride: Step size between windows (required for expensive engines)

    Returns:
        dict with 'rolling_hurst' array
    """
    params = params or {}
    window = params.get('window', 100)
    # Expensive engine: stride MUST come from manifest, fallback to window//4
    stride = params.get('stride', max(1, window // 4))

    n = len(y)
    if n < window:
        return {'rolling_hurst': np.full(n, np.nan)}

    result = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        h = hurst.compute(chunk)
        result[i + window - 1] = h['hurst']

    return {'rolling_hurst': result}

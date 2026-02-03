"""
Rolling Volatility Engine.

Computes volatility (realized variance) over rolling windows.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling volatility.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_volatility' array
    """
    params = params or {}
    window = params.get('window', 50)
    # Moderate cost: default to window//4 if not specified
    stride = params.get('stride', max(1, window // 4))

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

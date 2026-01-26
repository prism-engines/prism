"""
Rolling Volatility Engine
=========================

Computes rolling volatility statistics for volatility clustering detection.

Metrics:
    - rolling_std_ratio: Max / Min rolling std (clustering indicator)
    - rolling_std_max: Maximum rolling std
    - rolling_std_min: Minimum rolling std
    - rolling_std_mean: Mean rolling std

High rolling_std_ratio = volatility clustering (ARCH effects)
Low rolling_std_ratio = homoskedastic (constant volatility)
"""

import numpy as np
from typing import Dict, Any


def compute(
    series: np.ndarray,
    window_size: int = None,
) -> Dict[str, Any]:
    """
    Compute rolling volatility statistics.

    Args:
        series: 1D numpy array of observations
        window_size: Window for rolling std (default: n/20)

    Returns:
        dict with:
            - rolling_std_ratio: Max / Min rolling std
            - rolling_std_max: Maximum rolling std
            - rolling_std_min: Minimum rolling std
            - rolling_std_mean: Mean rolling std
            - rolling_std_series: The rolling std array
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 20:
        return {
            'rolling_std_ratio': 1.0,
            'rolling_std_max': 0.0,
            'rolling_std_min': 0.0,
            'rolling_std_mean': 0.0,
            'rolling_std_series': np.array([]),
        }

    if window_size is None:
        window_size = max(10, n // 20)

    # Compute rolling std
    rolling_std = np.array([
        np.std(series[max(0, i - window_size):i + 1])
        for i in range(len(series))
    ])

    # Filter out zeros for ratio calculation
    nonzero_std = rolling_std[rolling_std > 1e-10]

    if len(nonzero_std) == 0:
        return {
            'rolling_std_ratio': 1.0,
            'rolling_std_max': 0.0,
            'rolling_std_min': 0.0,
            'rolling_std_mean': 0.0,
            'rolling_std_series': rolling_std,
        }

    rolling_std_max = float(np.max(nonzero_std))
    rolling_std_min = float(np.min(nonzero_std))
    rolling_std_mean = float(np.mean(nonzero_std))

    # Ratio indicates volatility clustering
    # High ratio = volatility varies a lot over time
    if rolling_std_min > 1e-10:
        rolling_std_ratio = rolling_std_max / rolling_std_min
    else:
        rolling_std_ratio = 1.0

    return {
        'rolling_std_ratio': float(rolling_std_ratio),
        'rolling_std_max': rolling_std_max,
        'rolling_std_min': rolling_std_min,
        'rolling_std_mean': rolling_std_mean,
        'rolling_std_series': rolling_std,
    }

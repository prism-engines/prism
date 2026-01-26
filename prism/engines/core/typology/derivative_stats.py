"""
Derivative Statistics Engine
============================

Computes derivative-based statistics for the Signal Typology framework.

Metrics:
    - derivative_mean: Mean rate of change
    - derivative_std: Volatility of rate of change
    - derivative_kurtosis: Spikiness of changes
    - zero_crossing_rate: How often direction changes

High kurtosis + low mean = spiky signal
High zero crossing rate = oscillatory signal
"""

import numpy as np
from typing import Dict, Any


def compute(series: np.ndarray) -> Dict[str, Any]:
    """
    Compute derivative statistics.

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - derivative_mean: Mean of first derivative
            - derivative_std: Std of first derivative
            - derivative_kurtosis: Kurtosis of first derivative
            - zero_crossing_rate: Rate of sign changes in derivative
    """
    series = np.asarray(series).flatten()

    if len(series) < 2:
        return {
            'derivative_mean': 0.0,
            'derivative_std': 0.0,
            'derivative_kurtosis': 3.0,
            'zero_crossing_rate': 0.0,
        }

    # First derivative
    derivative = np.diff(series)

    # Basic stats
    derivative_mean = float(np.mean(derivative))
    derivative_std = float(np.std(derivative))

    # Kurtosis (excess kurtosis, normal = 0)
    if derivative_std > 1e-10:
        standardized = (derivative - derivative_mean) / derivative_std
        derivative_kurtosis = float(np.mean(standardized ** 4))
    else:
        derivative_kurtosis = 3.0  # Normal distribution kurtosis

    # Zero crossing rate
    signs = np.sign(derivative)
    # Handle zeros in sign
    signs[signs == 0] = 1
    sign_changes = np.sum(signs[1:] != signs[:-1])
    zero_crossing_rate = float(sign_changes / len(derivative)) if len(derivative) > 0 else 0.0

    return {
        'derivative_mean': derivative_mean,
        'derivative_std': derivative_std,
        'derivative_kurtosis': derivative_kurtosis,
        'zero_crossing_rate': zero_crossing_rate,
    }

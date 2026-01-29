"""
Rate of Change Engine.

Computes mean and max rate of change.
Used for temperature ramp detection, pressure transients.
"""

import numpy as np


def compute(y: np.ndarray, I: np.ndarray = None) -> dict:
    """
    Compute rate of change metrics.

    Args:
        y: Signal values
        I: Index/time values (optional, assumes uniform if None)

    Returns:
        dict with mean_rate, max_rate, min_rate, rate_std
    """
    if len(y) < 3:
        return {
            'mean_rate': np.nan,
            'max_rate': np.nan,
            'min_rate': np.nan,
            'rate_std': np.nan
        }

    if I is None:
        dy = np.diff(y)
    else:
        dy = np.gradient(y, I)

    return {
        'mean_rate': float(np.mean(dy)),
        'max_rate': float(np.max(dy)),
        'min_rate': float(np.min(dy)),
        'rate_std': float(np.std(dy))
    }

"""
Kurtosis Engine.

Computes the kurtosis (4th moment) of a signal.
"""

import numpy as np
from scipy import stats


def compute(y: np.ndarray) -> dict:
    """
    Compute kurtosis of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'kurtosis' key (Fisher definition: 0 for normal)
    """
    if len(y) < 4:
        return {'kurtosis': np.nan}

    return {'kurtosis': float(stats.kurtosis(y, fisher=True))}

"""
Skewness Engine.

Computes the skewness (3rd moment) of a signal.
"""

import numpy as np
from scipy import stats


def compute(y: np.ndarray) -> dict:
    """
    Compute skewness of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'skewness' key
    """
    if len(y) < 3:
        return {'skewness': np.nan}

    return {'skewness': float(stats.skew(y))}

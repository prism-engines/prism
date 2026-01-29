"""
Peak Engine.

Computes the peak (maximum absolute) value of a signal.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute peak value of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'peak' and 'peak_to_peak' keys
    """
    if len(y) < 1:
        return {'peak': np.nan, 'peak_to_peak': np.nan}

    return {
        'peak': float(np.max(np.abs(y))),
        'peak_to_peak': float(np.max(y) - np.min(y))
    }

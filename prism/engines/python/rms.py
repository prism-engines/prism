"""
RMS (Root Mean Square) Engine.

Computes the RMS value of a signal.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute RMS of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'rms' key
    """
    if len(y) < 1:
        return {'rms': np.nan}

    return {'rms': float(np.sqrt(np.mean(y ** 2)))}

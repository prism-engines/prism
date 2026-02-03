"""
Crest Factor Engine.

Computes the crest factor (peak / RMS) of a signal.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute crest factor of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'crest_factor' key
    """
    if len(y) < 1:
        return {'crest_factor': np.nan}

    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))

    return {'crest_factor': float(peak / rms) if rms > 0 else np.nan}

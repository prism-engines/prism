"""
Rolling Entropy Engine.

Computes entropy over rolling windows.
"""

import numpy as np
from ..python import entropy


def compute(y: np.ndarray, window: int = 100, stride: int = 1) -> dict:
    """
    Compute rolling entropy.

    Args:
        y: Signal values
        window: Window size
        stride: Step size between windows

    Returns:
        dict with 'rolling_sample_entropy', 'rolling_permutation_entropy' arrays
    """
    n = len(y)
    if n < window:
        return {
            'rolling_sample_entropy': np.full(n, np.nan),
            'rolling_permutation_entropy': np.full(n, np.nan)
        }

    sample_ent = np.full(n, np.nan)
    perm_ent = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        e = entropy.compute(chunk)
        sample_ent[i + window - 1] = e['sample_entropy']
        perm_ent[i + window - 1] = e['permutation_entropy']

    return {
        'rolling_sample_entropy': sample_ent,
        'rolling_permutation_entropy': perm_ent
    }

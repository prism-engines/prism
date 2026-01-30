"""
Rolling Entropy Engine.

Computes entropy over rolling windows.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from ..signal import entropy


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling entropy.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size (required for meaningful results)
            - stride: Step size between windows (required for expensive engines)

    Returns:
        dict with 'rolling_sample_entropy', 'rolling_permutation_entropy' arrays
    """
    params = params or {}
    window = params.get('window', 100)
    # Expensive engine (O(n²)): stride MUST come from manifest, fallback to window//4
    stride = params.get('stride', max(1, window // 4))

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

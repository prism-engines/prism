"""
Stability Engine.

Computes stability state at every observation point.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, dy: np.ndarray, d2y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute stability state at each point.

    Args:
        y: Signal values
        dy: First derivative values
        d2y: Second derivative values
        params: Parameters from manifest:
            - dy_threshold_pct: Threshold as % of dy std (default: 1%)

    Returns:
        dict with 'stability_state' and 'is_locally_stable' arrays
    """
    params = params or {}
    dy_threshold_pct = params.get('dy_threshold_pct', 0.01)

    n = len(y)
    if n == 0:
        return {
            'stability_state': np.array([], dtype=object),
            'is_locally_stable': np.array([], dtype=bool)
        }

    dy_threshold = dy_threshold_pct * (np.std(dy) + 1e-10)

    states = np.empty(n, dtype=object)
    stable = np.ones(n, dtype=bool)

    for i in range(n):
        if abs(dy[i]) < dy_threshold:
            if d2y[i] > 0:
                states[i] = 'stable_min'
                stable[i] = True
            elif d2y[i] < 0:
                states[i] = 'unstable_max'
                stable[i] = False
            else:
                states[i] = 'saddle'
                stable[i] = False
        elif dy[i] > 0:
            states[i] = 'increasing'
            stable[i] = True
        else:
            states[i] = 'decreasing'
            stable[i] = True

    return {
        'stability_state': states,
        'is_locally_stable': stable
    }

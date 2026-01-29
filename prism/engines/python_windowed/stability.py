"""
Stability Engine.

Computes stability state at every observation point.
"""

import numpy as np


def compute(y: np.ndarray, dy: np.ndarray, d2y: np.ndarray) -> dict:
    """
    Compute stability state at each point.

    Args:
        y: Signal values
        dy: First derivative values
        d2y: Second derivative values

    Returns:
        dict with 'stability_state' and 'is_locally_stable' arrays
    """
    n = len(y)
    if n == 0:
        return {
            'stability_state': np.array([], dtype=object),
            'is_locally_stable': np.array([], dtype=bool)
        }

    dy_threshold = 0.01 * (np.std(dy) + 1e-10)

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

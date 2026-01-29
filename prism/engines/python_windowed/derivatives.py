"""
Derivatives Engine.

Computes derivatives at every observation point.
"""

import numpy as np


def compute(y: np.ndarray, I: np.ndarray) -> dict:
    """
    Compute derivatives of signal.

    Args:
        y: Signal values
        I: Index values (time, cycle, etc.)

    Returns:
        dict with 'dy', 'd2y', 'd3y', 'curvature' arrays
    """
    if len(y) < 3:
        n = len(y)
        return {
            'dy': np.zeros(n),
            'd2y': np.zeros(n),
            'd3y': np.zeros(n),
            'curvature': np.zeros(n)
        }

    dy = np.gradient(y, I)
    d2y = np.gradient(dy, I)
    d3y = np.gradient(d2y, I)
    curvature = np.abs(d2y) / ((1 + dy**2) ** 1.5 + 1e-10)

    return {
        'dy': dy,
        'd2y': d2y,
        'd3y': d3y,
        'curvature': curvature
    }

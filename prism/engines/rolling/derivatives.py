"""
Derivatives Engine.

Computes derivatives at every observation point.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, I: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute derivatives of signal.

    Args:
        y: Signal values
        I: Index values (time, cycle, etc.)
        params: Parameters from manifest (currently unused, reserved for future)

    Returns:
        dict with 'dy', 'd2y', 'd3y', 'curvature' arrays
    """
    params = params or {}
    # Reserved for future params like smoothing, method, etc.

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

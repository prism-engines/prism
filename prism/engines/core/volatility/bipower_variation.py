"""
Bipower Variation
=================

Jump-robust volatility measure.

BV = (π/2) × Σ |r_t| × |r_{t-1}|

The difference between realized variance and bipower variation
isolates the contribution of jumps to total volatility:

    Jump² = max(0, RV² - BV²)

References:
    Barndorff-Nielsen & Shephard (2004)
    "Power and Bipower Variation with Stochastic Volatility and Jumps"
"""

import numpy as np
from typing import Dict


def compute(series: np.ndarray) -> Dict[str, float]:
    """
    Compute bipower variation (jump-robust volatility).

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - bipower: Bipower variation estimate
            - bipower_vol: Square root of bipower variation
            - jump_component: Estimated jump contribution
            - jump_ratio: Jump / RV ratio
    """
    if len(series) < 3:
        return {
            'bipower': 0.0,
            'bipower_vol': 0.0,
            'jump_component': 0.0,
            'jump_ratio': 0.0
        }

    returns = np.diff(series)
    abs_returns = np.abs(returns)

    if len(abs_returns) < 2:
        return {
            'bipower': 0.0,
            'bipower_vol': 0.0,
            'jump_component': 0.0,
            'jump_ratio': 0.0
        }

    # Bipower variation: (π/2) × Σ|r_t| × |r_{t-1}|
    bipower = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])
    bipower_vol = np.sqrt(bipower)

    # Realized variance for comparison
    realized_variance = np.sum(returns ** 2)

    # Jump component: RV - BV (if positive)
    jump_squared = max(0, realized_variance - bipower)
    jump_component = np.sqrt(jump_squared)

    # Jump ratio
    realized_vol = np.sqrt(realized_variance)
    jump_ratio = jump_component / realized_vol if realized_vol > 0 else 0.0

    return {
        'bipower': float(bipower),
        'bipower_vol': float(bipower_vol),
        'jump_component': float(jump_component),
        'jump_ratio': float(jump_ratio)
    }

"""
CUSUM and Level Shift Engine
=============================

Detects structural changes via CUSUM (Cumulative Sum) and level shifts.

Metrics:
    - cusum_max: Maximum cumulative deviation from mean
    - cusum_crossings: Number of threshold violations
    - level_shift_count: Number of detected level shifts
    - level_shift_magnitude_mean: Mean magnitude of level shifts

High cusum_max = sustained deviation from mean
High level_shift_count = step-like structure
"""

import numpy as np
from typing import Dict, Any, List


def compute(
    series: np.ndarray,
    level_shift_window: int = None,
    level_shift_threshold_std: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute CUSUM and level shift metrics.

    Args:
        series: 1D numpy array of observations
        level_shift_window: Window size for level shift detection (default: n/50)
        level_shift_threshold_std: Std multiplier for level shift threshold

    Returns:
        dict with:
            - cusum_max: Maximum absolute CUSUM value
            - cusum_crossings: Number of threshold crossings
            - level_shift_count: Number of detected level shifts
            - level_shift_magnitude_mean: Mean magnitude of shifts
            - level_shift_locations: List of shift locations
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 10:
        return {
            'cusum_max': 0.0,
            'cusum_crossings': 0,
            'level_shift_count': 0,
            'level_shift_magnitude_mean': 0.0,
            'level_shift_locations': [],
        }

    # CUSUM computation
    mean_val = np.mean(series)
    cusum = np.cumsum(series - mean_val)
    cusum_max = float(np.max(np.abs(cusum)))

    # CUSUM threshold crossings
    # Threshold grows with sqrt(n) - standard control chart approach
    std_val = np.std(series)
    threshold = level_shift_threshold_std * std_val * np.sqrt(np.arange(1, n + 1))
    cusum_crossings = int(np.sum(np.abs(cusum) > threshold))

    # Level shift detection
    if level_shift_window is None:
        level_shift_window = max(10, n // 50)

    shift_locations: List[int] = []
    shift_magnitudes: List[float] = []

    for i in range(level_shift_window, n - level_shift_window):
        left_mean = np.mean(series[i - level_shift_window:i])
        right_mean = np.mean(series[i:i + level_shift_window])
        diff = abs(right_mean - left_mean)

        if diff > level_shift_threshold_std * std_val:
            # Check if this is a new shift (not too close to previous)
            if not shift_locations or i - shift_locations[-1] >= level_shift_window:
                shift_locations.append(i)
                shift_magnitudes.append(diff)

    level_shift_count = len(shift_locations)
    level_shift_magnitude_mean = float(np.mean(shift_magnitudes)) if shift_magnitudes else 0.0

    return {
        'cusum_max': cusum_max,
        'cusum_crossings': cusum_crossings,
        'level_shift_count': level_shift_count,
        'level_shift_magnitude_mean': level_shift_magnitude_mean,
        'level_shift_locations': shift_locations,
    }

"""
Permutation Entropy
===================

Bandt-Pompe permutation entropy measures the complexity of a time series
by analyzing the ordinal patterns in the data.

Normalized to [0, 1] where:
    - 0: Completely deterministic
    - 1: Completely random

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Bandt & Pompe (2002) "Permutation Entropy: A Natural Complexity Measure"
"""

import os
import numpy as np
from itertools import permutations
import math
from typing import Dict, Any, Optional

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    order: int = 3,
    delay: int = 1,
) -> Dict[str, Any]:
    """
    Compute permutation entropy (Bandt & Pompe, 2002).

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        order: Embedding dimension (pattern length)
        delay: Time delay between pattern elements

    Returns:
        mode='static': {'entropy': float, 'n_patterns': int}
        mode='windowed': {'entropy': array, 'n_patterns': array, 't': array, ...}
        mode='point': {'entropy': float, 'n_patterns': int, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, order, delay)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, order, delay)
    elif mode == 'point':
        return _compute_point(series, t, window_size, order, delay)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> Dict[str, Any]:
    """Compute permutation entropy on entire signal."""
    n = len(series)

    if n < order * delay + 1:
        return {'entropy': 1.0, 'n_patterns': 0}

    factorial_order = math.factorial(order)

    # All possible permutations
    all_patterns = list(permutations(range(order)))
    pattern_counts = {p: 0 for p in all_patterns}

    n_patterns = 0
    for i in range(n - (order - 1) * delay):
        # Extract embedded vector
        indices = [i + j * delay for j in range(order)]
        values = series[indices]

        # Get ordinal pattern (rank of each value)
        pattern = tuple(np.argsort(np.argsort(values)))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        n_patterns += 1

    if n_patterns == 0:
        return {'entropy': 1.0, 'n_patterns': 0, 'n_unique_patterns': 0, 'pattern_coverage': 0.0}

    # Compute entropy
    probs = np.array([c / n_patterns for c in pattern_counts.values() if c > 0])
    entropy = -np.sum(probs * np.log(probs))

    # Normalize by maximum entropy (log of factorial(order))
    max_entropy = np.log(factorial_order)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    # Confidence metrics
    n_unique = sum(1 for c in pattern_counts.values() if c > 0)
    pattern_coverage = n_unique / factorial_order  # What fraction of possible patterns observed?

    result = {
        'entropy': float(normalized),
        'n_patterns': n_patterns,
        'n_unique_patterns': n_unique,
        'pattern_coverage': float(pattern_coverage),  # Confidence metric: sampling coverage
    }

    if VERBOSE:
        result['entropy_unnormalized'] = float(entropy)
        result['max_entropy'] = float(max_entropy)
        result['order'] = order
        result['delay'] = delay
        result['possible_patterns'] = factorial_order
        # Distribution uniformity (how evenly distributed are the patterns)
        if n_unique > 1:
            expected_uniform = 1.0 / n_unique
            observed_probs = [c / n_patterns for c in pattern_counts.values() if c > 0]
            uniformity = 1.0 - np.std(observed_probs) / expected_uniform
            result['pattern_uniformity'] = float(max(0, uniformity))

    return result


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    order: int = 3,
    delay: int = 1,
) -> Dict[str, Any]:
    """Compute permutation entropy over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'entropy': np.array([]),
            'n_patterns': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    entropy_values = []
    n_patterns_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, order, delay)

        t_values.append(start + window_size // 2)
        entropy_values.append(result['entropy'])
        n_patterns_values.append(result['n_patterns'])

    return {
        'entropy': np.array(entropy_values),
        'n_patterns': np.array(n_patterns_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    order: int = 3,
    delay: int = 1,
) -> Dict[str, Any]:
    """Compute permutation entropy at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    if len(window) < order * delay + 1:
        return {
            'entropy': 1.0,
            'n_patterns': 0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, order, delay)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result

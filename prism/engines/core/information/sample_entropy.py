"""
Sample Entropy
==============

Sample entropy (SampEn) measures the complexity of a time series
by quantifying the probability that similar patterns remain similar.

Lower values indicate more self-similarity (regularity).
Higher values indicate more complexity.

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Richman & Moorman (2000) "Physiological time-series analysis"
"""

import os
import numpy as np
from typing import Dict, Any, Optional

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    m: int = 2,
    r: float = None,
) -> Dict[str, Any]:
    """
    Compute sample entropy.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)

    Returns:
        mode='static': {'entropy': float, 'matches_m': int, 'matches_m1': int}
        mode='windowed': {'entropy': array, 'matches_m': array, 't': array, ...}
        mode='point': {'entropy': float, 'matches_m': int, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, m, r)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, m, r)
    elif mode == 'point':
        return _compute_point(series, t, window_size, m, r)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(
    series: np.ndarray,
    m: int = 2,
    r: float = None,
) -> Dict[str, Any]:
    """Compute sample entropy on entire signal."""
    n = len(series)

    if r is None:
        r = 0.2 * np.std(series)

    if r == 0 or n < m + 2:
        return {'entropy': 0.0, 'matches_m': 0, 'matches_m1': 0}

    def count_matches(template_length):
        count = 0
        for i in range(n - template_length):
            for j in range(i + 1, n - template_length):
                # Check if templates match within tolerance
                diff = np.abs(series[i:i+template_length] - series[j:j+template_length])
                if np.all(diff <= r):
                    count += 1
        return count

    # Count matches for m and m+1
    a = count_matches(m)
    b = count_matches(m + 1)

    if a == 0 or b == 0:
        return {'entropy': 0.0, 'matches_m': a, 'matches_m1': b, 'match_ratio': 0.0}

    entropy = -np.log(b / a)
    match_ratio = b / a  # Conditional probability of continued similarity

    result = {
        'entropy': float(entropy),
        'matches_m': a,
        'matches_m1': b,
        'match_ratio': float(match_ratio),  # Confidence metric: stability of pattern matches
    }

    if VERBOSE:
        # Total possible pairs
        n_pairs = (n - m) * (n - m - 1) // 2
        n_pairs_m1 = (n - m - 1) * (n - m - 2) // 2
        result['match_density_m'] = float(a / n_pairs) if n_pairs > 0 else 0.0
        result['match_density_m1'] = float(b / n_pairs_m1) if n_pairs_m1 > 0 else 0.0
        result['tolerance'] = float(r)
        result['n_samples'] = n

    return result


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    m: int = 2,
    r: float = None,
) -> Dict[str, Any]:
    """Compute sample entropy over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'entropy': np.array([]),
            'matches_m': np.array([]),
            'matches_m1': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    entropy_values = []
    matches_m_values = []
    matches_m1_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        # Compute r per window if not specified globally
        window_r = r if r is not None else 0.2 * np.std(window)
        result = _compute_static(window, m, window_r)

        t_values.append(start + window_size // 2)
        entropy_values.append(result['entropy'])
        matches_m_values.append(result['matches_m'])
        matches_m1_values.append(result['matches_m1'])

    return {
        'entropy': np.array(entropy_values),
        'matches_m': np.array(matches_m_values),
        'matches_m1': np.array(matches_m1_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    m: int = 2,
    r: float = None,
) -> Dict[str, Any]:
    """Compute sample entropy at specific time t."""
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

    if len(window) < m + 2:
        return {
            'entropy': 0.0,
            'matches_m': 0,
            'matches_m1': 0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    # Compute r for this window if not specified
    window_r = r if r is not None else 0.2 * np.std(window)
    result = _compute_static(window, m, window_r)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result

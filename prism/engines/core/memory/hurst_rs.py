"""
Hurst Exponent via Rescaled Range (R/S) Analysis
================================================

Classical method for estimating the Hurst exponent.
Measures long-term memory in time series.

Interpretation:
    H < 0.5: Anti-persistent (mean-reverting)
    H = 0.5: Random walk (no memory)
    H > 0.5: Persistent (trending)

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Hurst (1951) "Long-term storage capacity of reservoirs"
"""

import os
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    min_window: int = 10,
) -> Dict[str, Any]:
    """
    Compute Hurst exponent using R/S (rescaled range) analysis.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        min_window: Minimum internal window size for R/S analysis

    Returns:
        mode='static': {'hurst_exponent': float, 'confidence': float}
        mode='windowed': {'hurst_exponent': array, 'confidence': array, 't': array, ...}
        mode='point': {'hurst_exponent': float, 'confidence': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, min_window)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, min_window)
    elif mode == 'point':
        return _compute_point(series, t, window_size, min_window)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray, min_window: int = 10) -> Dict[str, float]:
    """Compute Hurst exponent on entire signal."""
    n = len(series)

    if n < min_window * 2:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    # Window sizes (powers of 2 that fit)
    max_k = int(np.floor(np.log2(n / min_window)))
    if max_k < 2:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    window_sizes = [int(n / (2**k)) for k in range(max_k + 1)]
    window_sizes = [w for w in window_sizes if w >= min_window]

    rs_values = []

    for ws in window_sizes:
        n_windows = n // ws
        rs_list = []

        for i in range(n_windows):
            window = series[i * ws:(i + 1) * ws]

            # Mean-adjusted cumulative sum
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)

            # Range and standard deviation
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(window, ddof=1)

            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((ws, np.mean(rs_list)))

    if len(rs_values) < 2:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    # Log-log regression
    log_n = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)

    result = {
        'hurst_exponent': float(np.clip(slope, 0, 1)),
        'confidence': float(r_value ** 2),  # Backwards compatible
        # Full confidence metrics
        'hurst_r2': float(r_value ** 2),
        'hurst_p_value': float(p_value),
        'hurst_std_err': float(std_err),
        'hurst_intercept': float(intercept),
    }

    if VERBOSE:
        result['n_scales'] = len(rs_values)
        result['log_window_sizes'] = log_n.tolist()
        result['log_rs_values'] = log_rs.tolist()

    return result


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    min_window: int = 10,
) -> Dict[str, Any]:
    """Compute Hurst exponent over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'hurst_exponent': np.array([]),
            'confidence': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    hurst_values = []
    confidence_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, min_window)

        t_values.append(start + window_size // 2)
        hurst_values.append(result['hurst_exponent'])
        confidence_values.append(result['confidence'])

    return {
        'hurst_exponent': np.array(hurst_values),
        'confidence': np.array(confidence_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    min_window: int = 10,
) -> Dict[str, Any]:
    """Compute Hurst exponent at specific time t."""
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

    if len(window) < min_window * 2:
        return {
            'hurst_exponent': 0.5,
            'confidence': 0.0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, min_window)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result

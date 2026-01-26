"""
Hurst Exponent via Detrended Fluctuation Analysis (DFA)
=======================================================

DFA is more robust to non-stationarity than R/S analysis.
Estimates the Hurst exponent H where:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H ≈ 0.5: Random walk
    - H > 0.5: Persistent (trending)

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Peng et al. (1994) "Mosaic organization of DNA nucleotides"
"""

import os
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional

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
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        min_window: Minimum window size for DFA analysis

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


def _compute_static(series: np.ndarray, min_window: int = 10) -> Dict[str, Any]:
    """Compute Hurst exponent via DFA on entire signal."""
    n = len(series)

    if n < min_window * 4:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    # Integrate the series (cumulative sum of deviations from mean)
    y = np.cumsum(series - np.mean(series))

    # Window sizes (logarithmically spaced)
    max_window = n // 4
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5)

    if len(window_sizes) < 3:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    fluctuations = []

    for ws in window_sizes:
        n_windows = n // ws
        f2_list = []

        for i in range(n_windows):
            segment = y[i * ws:(i + 1) * ws]

            # Fit linear trend
            x = np.arange(ws)
            slope, intercept = np.polyfit(x, segment, 1)
            trend = slope * x + intercept

            # Fluctuation (RMS of detrended segment)
            f2 = np.mean((segment - trend) ** 2)
            f2_list.append(f2)

        if f2_list:
            fluctuations.append((ws, np.sqrt(np.mean(f2_list))))

    if len(fluctuations) < 3:
        return {'hurst_exponent': 0.5, 'confidence': 0.0}

    # Log-log regression
    log_n = np.log([x[0] for x in fluctuations])
    log_f = np.log([x[1] for x in fluctuations])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_f)

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
        result['n_scales'] = len(fluctuations)
        result['log_window_sizes'] = log_n.tolist()
        result['log_fluctuations'] = log_f.tolist()

    return result


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    min_window: int = 10,
) -> Dict[str, Any]:
    """Compute Hurst via DFA over rolling windows."""
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
    """Compute Hurst via DFA at specific time t."""
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

    if len(window) < min_window * 4:
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

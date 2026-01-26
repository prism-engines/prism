"""
Realized Volatility
===================

Actual observed volatility from sum of squared returns.

RV = √(Σ r²_t)

This is the "true" volatility as observed in the data,
unlike GARCH which is a model of volatility.

The difference between realized vol and model vol is informative.

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    annualize: bool = False,
) -> Dict[str, Any]:
    """
    Compute realized volatility.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        annualize: If True, annualize assuming 252 trading days

    Returns:
        mode='static': {'realized_vol': float, 'realized_variance': float, ...}
        mode='windowed': {'realized_vol': array, 'realized_variance': array, 't': array, ...}
        mode='point': {'realized_vol': float, 'realized_variance': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, annualize)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, annualize)
    elif mode == 'point':
        return _compute_point(series, t, window_size, annualize)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray, annualize: bool = False) -> Dict[str, Any]:
    """Compute realized volatility on entire signal."""
    if len(series) < 2:
        return {
            'realized_vol': 0.0,
            'realized_variance': 0.0,
            'n_returns': 0
        }

    returns = np.diff(series)
    n = len(returns)

    realized_variance = np.sum(returns ** 2)
    realized_vol = np.sqrt(realized_variance)

    if annualize:
        # Annualize (assuming 252 trading days)
        realized_vol *= np.sqrt(252 / n)
        realized_variance *= (252 / n)

    return {
        'realized_vol': float(realized_vol),
        'realized_variance': float(realized_variance),
        'n_returns': n
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    annualize: bool = False,
) -> Dict[str, Any]:
    """Compute realized volatility over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'realized_vol': np.array([]),
            'realized_variance': np.array([]),
            'n_returns': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    vol_values = []
    var_values = []
    n_returns_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, annualize)

        t_values.append(start + window_size // 2)
        vol_values.append(result['realized_vol'])
        var_values.append(result['realized_variance'])
        n_returns_values.append(result['n_returns'])

    return {
        'realized_vol': np.array(vol_values),
        'realized_variance': np.array(var_values),
        'n_returns': np.array(n_returns_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    annualize: bool = False,
) -> Dict[str, Any]:
    """Compute realized volatility at specific time t."""
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

    if len(window) < 2:
        return {
            'realized_vol': 0.0,
            'realized_variance': 0.0,
            'n_returns': 0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, annualize)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result

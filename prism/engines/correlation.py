"""
Correlation Engine.

Computes Pearson correlation and cross-correlation between signal pairs.
"""

import numpy as np
from typing import Dict


def compute(y_a: np.ndarray, y_b: np.ndarray, max_lag: int = None) -> Dict[str, float]:
    """
    Compute correlation between two signals.

    Args:
        y_a: First signal values
        y_b: Second signal values
        max_lag: Maximum lag for cross-correlation (default: n//4)

    Returns:
        dict with:
            - correlation: Pearson correlation at lag 0
            - correlation_abs: Absolute Pearson correlation
            - max_xcorr: Maximum cross-correlation value
            - lag_at_max_xcorr: Lag where max cross-correlation occurs
            - n_points: Number of points used
    """
    result = {
        'correlation': np.nan,
        'correlation_abs': np.nan,
        'max_xcorr': np.nan,
        'lag_at_max_xcorr': np.nan,
        'n_points': 0
    }

    # Handle arrays and NaN values
    y_a = np.asarray(y_a).flatten()
    y_b = np.asarray(y_b).flatten()

    # Align lengths
    n = min(len(y_a), len(y_b))
    if n < 10:
        result['n_points'] = n
        return result

    y_a, y_b = y_a[:n], y_b[:n]

    # Remove pairs with NaN in either signal
    valid_mask = ~(np.isnan(y_a) | np.isnan(y_b))
    y_a = y_a[valid_mask]
    y_b = y_b[valid_mask]
    n = len(y_a)

    if n < 10:
        result['n_points'] = n
        return result

    result['n_points'] = n

    # Check for constant signals
    std_a = np.std(y_a)
    std_b = np.std(y_b)
    if std_a < 1e-10 or std_b < 1e-10:
        result['correlation'] = 0.0
        result['correlation_abs'] = 0.0
        return result

    try:
        # Pearson correlation at lag 0
        corr = np.corrcoef(y_a, y_b)[0, 1]
        if not np.isnan(corr):
            result['correlation'] = float(corr)
            result['correlation_abs'] = float(abs(corr))

        # Cross-correlation with lags
        if max_lag is None:
            max_lag = n // 4

        max_lag = min(max_lag, n - 1)

        if max_lag > 0:
            # Normalize signals
            y_a_norm = (y_a - np.mean(y_a)) / std_a
            y_b_norm = (y_b - np.mean(y_b)) / std_b

            # Full cross-correlation
            xcorr = np.correlate(y_a_norm, y_b_norm, mode='full') / n

            # Extract window around lag 0
            mid = len(xcorr) // 2
            start = max(0, mid - max_lag)
            end = min(len(xcorr), mid + max_lag + 1)

            xcorr_window = xcorr[start:end]
            lags = np.arange(start - mid, end - mid)

            # Find maximum absolute cross-correlation
            max_idx = np.argmax(np.abs(xcorr_window))
            result['max_xcorr'] = float(xcorr_window[max_idx])
            result['lag_at_max_xcorr'] = int(lags[max_idx])

    except Exception:
        pass

    return result

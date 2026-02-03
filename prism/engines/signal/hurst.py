"""
Hurst Exponent Engine.

Computes the Hurst exponent via R/S analysis.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute Hurst exponent of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'hurst' and 'hurst_r2' keys
    """
    n = len(y)
    if n < 20:
        return {'hurst': np.nan, 'hurst_r2': np.nan}

    max_k = min(n // 2, 100)
    sizes = [int(n / k) for k in range(2, max_k) if n / k >= 8]
    sizes = sorted(set(sizes))[:20]

    if len(sizes) < 3:
        return {'hurst': np.nan, 'hurst_r2': np.nan}

    rs_values = []
    for size in sizes:
        n_chunks = n // size
        rs_chunk = []
        for i in range(n_chunks):
            chunk = y[i*size:(i+1)*size]
            mean = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_values.append((np.log(size), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return {'hurst': np.nan, 'hurst_r2': np.nan}

    x = np.array([v[0] for v in rs_values])
    y_rs = np.array([v[1] for v in rs_values])
    slope, intercept = np.polyfit(x, y_rs, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y_rs - y_pred) ** 2)
    ss_tot = np.sum((y_rs - np.mean(y_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {'hurst': float(slope), 'hurst_r2': float(r2)}

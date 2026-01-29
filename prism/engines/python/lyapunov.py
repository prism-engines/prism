"""
Lyapunov Exponent Engine.

Computes the largest Lyapunov exponent for chaos detection.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute Lyapunov exponent of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'lyapunov' and 'is_chaotic' keys
    """
    n = len(y)
    if n < 100:
        return {'lyapunov': np.nan, 'is_chaotic': False}

    y_sub = y[::max(1, n//5000)]
    n = len(y_sub)

    embed_dim, delay = 3, max(1, n // 50)
    m = n - (embed_dim - 1) * delay
    if m < 20:
        return {'lyapunov': np.nan, 'is_chaotic': False}

    embedded = np.zeros((m, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = y_sub[i*delay:i*delay+m]

    lyap_sum, count = 0, 0
    for i in range(min(100, m - 10)):
        dists = np.sum((embedded[i+1:] - embedded[i])**2, axis=1)
        if len(dists) == 0:
            continue
        j = np.argmin(dists) + i + 1
        if j < m - 1:
            d0 = np.sqrt(dists[j - i - 1])
            d1 = np.linalg.norm(embedded[min(i+1, m-1)] - embedded[min(j+1, m-1)])
            if d0 > 1e-10 and d1 > 1e-10:
                lyap_sum += np.log(d1 / d0)
                count += 1

    lyap = lyap_sum / count if count > 0 else np.nan
    return {
        'lyapunov': float(lyap) if not np.isnan(lyap) else np.nan,
        'is_chaotic': bool(lyap > 0.01) if not np.isnan(lyap) else False
    }

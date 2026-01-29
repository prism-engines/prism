"""
Attractor Engine.

Computes attractor properties via phase space reconstruction.
"""

import numpy as np
from scipy.spatial.distance import pdist


def compute(y: np.ndarray) -> dict:
    """
    Compute attractor properties of signal.

    Args:
        y: Signal values

    Returns:
        dict with embedding_dim, correlation_dim, attractor_type, delay
    """
    n = len(y)
    if n < 100:
        return {
            'embedding_dim': np.nan,
            'correlation_dim': np.nan,
            'attractor_type': 'unknown',
            'delay': np.nan
        }

    # Estimate delay from first zero crossing of autocorrelation
    autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
    autocorr = autocorr[len(autocorr)//2:] / (autocorr[len(autocorr)//2] + 1e-10)
    delay = 1
    for i in range(1, min(len(autocorr), n // 4)):
        if autocorr[i] <= 0:
            delay = i
            break

    embed_dim = min(5, n // (delay * 5))
    if embed_dim < 2:
        embed_dim = 2

    m = n - (embed_dim - 1) * delay
    if m < 50:
        return {
            'embedding_dim': embed_dim,
            'correlation_dim': np.nan,
            'attractor_type': 'unknown',
            'delay': delay
        }

    embedded = np.zeros((m, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = y[i*delay:i*delay+m]

    sample_size = min(500, m)
    indices = np.random.choice(m, sample_size, replace=False) if m > 500 else np.arange(m)
    dists = pdist(embedded[indices])
    dists = dists[dists > 0]

    if len(dists) < 100:
        return {
            'embedding_dim': embed_dim,
            'correlation_dim': np.nan,
            'attractor_type': 'unknown',
            'delay': delay
        }

    radii = np.percentile(dists, [10, 20, 30, 40, 50])
    log_r, log_c = [], []
    for r in radii:
        c = np.sum(dists < r) / len(dists)
        if c > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(c))

    corr_dim = np.nan
    if len(log_r) >= 3:
        corr_dim, _ = np.polyfit(log_r, log_c, 1)

    if np.isnan(corr_dim):
        atype = 'unknown'
    elif corr_dim < 1.5:
        atype = 'fixed_point'
    elif corr_dim < 2.5:
        atype = 'limit_cycle'
    elif corr_dim < 3.5:
        atype = 'torus'
    else:
        atype = 'strange'

    return {
        'embedding_dim': int(embed_dim),
        'correlation_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
        'attractor_type': atype,
        'delay': int(delay)
    }

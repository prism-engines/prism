"""
Attractor Engine.

Computes attractor properties via phase space reconstruction.
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Dict, Union


def compute(y: np.ndarray, embedding_dim: int = None, delay: int = None) -> Dict[str, Union[float, int, str]]:
    """
    Compute attractor properties of signal.

    Args:
        y: Signal values
        embedding_dim: Embedding dimension (auto-detected if None)
        delay: Time delay (auto-detected if None)

    Returns:
        dict with embedding_dim, correlation_dim, attractor_type, delay
    """
    result = {
        'embedding_dim': np.nan,
        'correlation_dim': np.nan,
        'attractor_type': 'unknown',
        'delay': np.nan
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 100:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['attractor_type'] = 'fixed_point'
        result['correlation_dim'] = 0.0
        return result

    try:
        # Estimate delay from first zero crossing of autocorrelation
        if delay is None:
            y_centered = y - np.mean(y)
            autocorr = np.correlate(y_centered, y_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_max = autocorr[0]
            if autocorr_max > 0:
                autocorr = autocorr / autocorr_max
            else:
                autocorr = np.zeros_like(autocorr)

            delay = 1
            for i in range(1, min(len(autocorr), n // 4)):
                if autocorr[i] <= 0:
                    delay = i
                    break

        # Determine embedding dimension
        if embedding_dim is None:
            embedding_dim = min(5, max(2, n // (delay * 5)))

        # Check if we have enough points for embedding
        m = n - (embedding_dim - 1) * delay
        if m < 50:
            return {
                'embedding_dim': int(embedding_dim),
                'correlation_dim': np.nan,
                'attractor_type': 'unknown',
                'delay': int(delay)
            }

        # Create embedded matrix
        embedded = np.zeros((m, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = y[i*delay:i*delay+m]

        # Sample for efficiency
        sample_size = min(500, m)
        if m > 500:
            indices = np.random.choice(m, sample_size, replace=False)
        else:
            indices = np.arange(m)

        dists = pdist(embedded[indices])
        dists = dists[dists > 0]

        if len(dists) < 100:
            return {
                'embedding_dim': int(embedding_dim),
                'correlation_dim': np.nan,
                'attractor_type': 'unknown',
                'delay': int(delay)
            }

        # Correlation dimension via Grassberger-Procaccia
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

        # Classify attractor type
        if np.isnan(corr_dim):
            atype = 'unknown'
        elif corr_dim < 0.5:
            atype = 'fixed_point'
        elif corr_dim < 1.5:
            atype = 'limit_cycle'
        elif corr_dim < 2.5:
            atype = 'torus'
        else:
            atype = 'strange'

        result = {
            'embedding_dim': int(embedding_dim),
            'correlation_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
            'attractor_type': atype,
            'delay': int(delay)
        }

    except Exception:
        pass

    return result

"""
Lyapunov Exponent Engine.

Computes the largest Lyapunov exponent using Rosenstein's algorithm.
ENGINES computes only - no classification.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray) -> Dict[str, Any]:
    """
    Compute Lyapunov exponent of signal.

    Uses Rosenstein's algorithm which tracks divergence of nearby
    trajectories in reconstructed phase space.

    Args:
        y: Signal values (raw 1D time series — embedding handled by primitive)

    Returns:
        dict with:
            - 'lyapunov': Largest Lyapunov exponent (number only)
            - 'embedding_dim': Used embedding dimension
            - 'embedding_tau': Used time delay
            - 'confidence': Confidence in estimate (0-1)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # Hard math floor: need enough for embedding + divergence tracking.
    # Dispatch layer already gates on config.yaml min_window (64).
    if n < 30:
        return _empty_result()

    # Pass raw signal to lyapunov_rosenstein — it handles embedding internally.
    # Do NOT pre-embed; the primitive expects a 1D time series.
    from engines.primitives.dynamical.lyapunov import lyapunov_rosenstein

    lyap, divergence, iterations = lyapunov_rosenstein(y)

    if np.isnan(lyap):
        return _empty_result()

    # iterations is np.arange(max_iter) — use its length for confidence
    n_iters = len(iterations) if hasattr(iterations, '__len__') else 0
    confidence = min(1.0, n_iters / 100) if n_iters > 0 else 0.0

    return {
        'lyapunov': float(lyap),
        'embedding_dim': np.nan,
        'embedding_tau': np.nan,
        'confidence': float(confidence),
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'lyapunov': np.nan,
        'embedding_dim': np.nan,
        'embedding_tau': np.nan,
        'confidence': np.nan,
    }

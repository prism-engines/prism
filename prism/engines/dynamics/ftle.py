"""
Finite-Time Lyapunov Exponent (FTLE) Engine.

Computes trajectory-dependent stability using FTLE.
Unlike global Lyapunov, FTLE varies with position on the attractor.

PRISM computes numbers. ORTHON interprets:
    - High FTLE regions = sensitive (Lagrangian Coherent Structures)
    - Low FTLE regions = stable
    - Rising FTLE = approaching sensitive region
"""

import numpy as np
from typing import Dict, Any, Optional

from prism.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)
from prism.primitives.dynamical.ftle import (
    ftle_local_linearization,
    compute_cauchy_green_tensor,
    detect_lcs_ridges,
)


def compute(
    y: np.ndarray,
    time_horizon: int = 10,
    min_samples: int = 100,
    emb_dim: Optional[int] = None,
    emb_tau: Optional[int] = None,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Compute FTLE from time series.

    Args:
        y: Signal values
        time_horizon: Steps to track divergence
        min_samples: Minimum samples required
        emb_dim: Embedding dimension (auto if None)
        emb_tau: Embedding delay (auto if None)
        n_neighbors: Neighbors for local linearization

    Returns:
        dict with ftle, ftle_mean, ftle_std, ftle_max, confidence
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < min_samples:
        return _empty_result()

    try:
        # Auto-detect embedding parameters
        if emb_tau is None:
            emb_tau = optimal_delay(y, max_lag=min(100, n // 10))
        if emb_dim is None:
            emb_dim = optimal_dimension(y, emb_tau, max_dim=10)

        # Embed signal
        embedded = time_delay_embedding(y, dimension=emb_dim, delay=emb_tau)

        if len(embedded) < time_horizon + 20:
            return _empty_result()

        # Compute FTLE
        ftle, confidence = ftle_local_linearization(
            embedded,
            time_horizon=time_horizon,
            n_neighbors=n_neighbors,
        )

        # Filter valid values
        valid = ~np.isnan(ftle)
        if np.sum(valid) < 5:
            return _empty_result()

        ftle_valid = ftle[valid]
        conf_valid = confidence[valid]

        return {
            'ftle': ftle,
            'ftle_mean': float(np.mean(ftle_valid)),
            'ftle_std': float(np.std(ftle_valid)),
            'ftle_max': float(np.max(ftle_valid)),
            'ftle_min': float(np.min(ftle_valid)),
            'ftle_current': float(ftle_valid[-1]) if len(ftle_valid) > 0 else None,
            'embedding_dim': emb_dim,
            'embedding_tau': emb_tau,
            'time_horizon': time_horizon,
            'confidence': float(np.mean(conf_valid)),
            'n_valid': int(np.sum(valid)),
        }

    except Exception:
        return _empty_result()


def compute_with_geometry(
    y: np.ndarray,
    time_horizon: int = 10,
    min_samples: int = 100,
    emb_dim: Optional[int] = None,
    emb_tau: Optional[int] = None,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Compute FTLE with Cauchy-Green tensor geometry.

    Returns additional geometric information about stretching.
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < min_samples:
        return _empty_result_extended()

    try:
        # Auto-detect embedding parameters
        if emb_tau is None:
            emb_tau = optimal_delay(y, max_lag=min(100, n // 10))
        if emb_dim is None:
            emb_dim = optimal_dimension(y, emb_tau, max_dim=10)

        # Embed signal
        embedded = time_delay_embedding(y, dimension=emb_dim, delay=emb_tau)

        if len(embedded) < time_horizon + 20:
            return _empty_result_extended()

        # Compute Cauchy-Green tensor
        eigenvalues, eigenvectors, ftle = compute_cauchy_green_tensor(
            embedded,
            time_horizon=time_horizon,
            n_neighbors=n_neighbors,
        )

        # Detect LCS ridges
        is_lcs = detect_lcs_ridges(ftle, embedded)

        valid = ~np.isnan(ftle)
        if np.sum(valid) < 5:
            return _empty_result_extended()

        ftle_valid = ftle[valid]

        # Compute anisotropy from eigenvalue ratios
        anisotropy = np.full(len(ftle), np.nan)
        for i in range(len(eigenvalues)):
            if not np.any(np.isnan(eigenvalues[i])) and eigenvalues[i][0] > 0:
                # Ratio of largest to second largest
                if len(eigenvalues[i]) > 1 and eigenvalues[i][1] > 0:
                    anisotropy[i] = eigenvalues[i][0] / eigenvalues[i][1]

        return {
            'ftle': ftle,
            'ftle_mean': float(np.mean(ftle_valid)),
            'ftle_std': float(np.std(ftle_valid)),
            'ftle_max': float(np.max(ftle_valid)),
            'ftle_min': float(np.min(ftle_valid)),
            'ftle_current': float(ftle_valid[-1]) if len(ftle_valid) > 0 else None,
            'cauchy_green_eigenvalues': eigenvalues,
            'stretching_anisotropy': anisotropy,
            'stretching_anisotropy_mean': float(np.nanmean(anisotropy)),
            'is_lcs': is_lcs,
            'n_lcs_points': int(np.sum(is_lcs)),
            'lcs_fraction': float(np.sum(is_lcs) / len(is_lcs)) if len(is_lcs) > 0 else 0.0,
            'embedding_dim': emb_dim,
            'embedding_tau': emb_tau,
            'time_horizon': time_horizon,
            'n_valid': int(np.sum(valid)),
        }

    except Exception:
        return _empty_result_extended()


def compute_rolling(
    y: np.ndarray,
    window: int = 200,
    stride: int = 20,
    time_horizon: int = 10,
    min_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling FTLE statistics.

    Args:
        y: Signal values
        window: Rolling window size
        stride: Step size
        time_horizon: FTLE time horizon
        min_samples: Min samples per window

    Returns:
        dict with rolling FTLE statistics
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n < window or window < min_samples:
        return {
            'rolling_ftle_mean': np.full(n, np.nan),
            'rolling_ftle_max': np.full(n, np.nan),
            'rolling_ftle_std': np.full(n, np.nan),
        }

    ftle_mean = np.full(n, np.nan)
    ftle_max = np.full(n, np.nan)
    ftle_std = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result = compute(chunk, time_horizon=time_horizon, min_samples=min_samples)

        idx = i + window - 1
        if result['ftle_mean'] is not None:
            ftle_mean[idx] = result['ftle_mean']
            ftle_max[idx] = result['ftle_max']
            ftle_std[idx] = result['ftle_std']

    return {
        'rolling_ftle_mean': ftle_mean,
        'rolling_ftle_max': ftle_max,
        'rolling_ftle_std': ftle_std,
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'ftle': None,
        'ftle_mean': None,
        'ftle_std': None,
        'ftle_max': None,
        'ftle_min': None,
        'ftle_current': None,
        'embedding_dim': None,
        'embedding_tau': None,
        'time_horizon': None,
        'confidence': 0.0,
        'n_valid': 0,
    }


def _empty_result_extended() -> Dict[str, Any]:
    """Return empty extended result."""
    result = _empty_result()
    result.update({
        'cauchy_green_eigenvalues': None,
        'stretching_anisotropy': None,
        'stretching_anisotropy_mean': None,
        'is_lcs': None,
        'n_lcs_points': 0,
        'lcs_fraction': 0.0,
    })
    return result

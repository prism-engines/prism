"""
Jacobian Engine - Wolf (1985) Algorithm

Computes Jacobian matrix and eigenvalues along embedded trajectory.

This complements Rosenstein-based Lyapunov by providing:
- Local stability analysis
- Eigenvalue spectrum
- Bifurcation detection capabilities

Reference:
    Wolf, Swift, Swinney & Vastano (1985)
    "Determining Lyapunov exponents from a time series"
"""

import numpy as np
from scipy import linalg
from typing import Dict, Any, Optional
import os

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'


def compute(
    signal: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    n_neighbors: int = 5,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate Jacobian via local linear fit in embedding space.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    embedding_dim : int
        Embedding dimension (m)
    tau : int
        Time delay for embedding
    n_neighbors : int
        Number of neighbors for local linear fit
    dt : float
        Time step for derivative estimation

    Returns
    -------
    dict with scalar summary statistics (always)
    plus detailed arrays if PRISM_VERBOSE=1
    """
    # Clean input
    signal = np.asarray(signal, dtype=float).flatten()
    signal = signal[np.isfinite(signal)]

    # Validate input
    min_length = (embedding_dim - 1) * tau + n_neighbors + 10
    if len(signal) < min_length:
        return _null_result(f"Insufficient data: need {min_length}, have {len(signal)}")

    # Build delay embedding
    n_points = len(signal) - (embedding_dim - 1) * tau
    embedded = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = signal[i * tau : i * tau + n_points]

    # Compute Jacobian at multiple points along trajectory
    jacobians = []
    eigenvalues_list = []

    # Sample points (not every point - computationally expensive)
    n_samples = min(100, max(10, n_points // 10))
    sample_indices = np.linspace(
        n_neighbors,
        n_points - 2,
        n_samples
    ).astype(int)

    for idx in sample_indices:
        J = _estimate_local_jacobian(embedded, idx, n_neighbors, dt)
        if J is not None:
            jacobians.append(J)
            eigs = linalg.eigvals(J)
            eigenvalues_list.append(eigs)

    if len(jacobians) == 0:
        return _null_result("Could not estimate Jacobian at any point")

    # Aggregate eigenvalues across trajectory
    all_eigenvalues = np.array(eigenvalues_list)
    mean_eigenvalues = np.mean(all_eigenvalues, axis=0)

    # Compute summary statistics
    real_parts = np.real(mean_eigenvalues)
    imag_parts = np.imag(mean_eigenvalues)

    max_real = float(np.max(real_parts))
    min_real = float(np.min(real_parts))
    mean_real = float(np.mean(real_parts))

    # Stability indicators
    n_positive = int(np.sum(real_parts > 0))
    n_negative = int(np.sum(real_parts < 0))

    # Jacobian summary (mean across trajectory)
    J_mean = np.mean(jacobians, axis=0)

    # Compute trace and determinant safely
    try:
        trace_val = float(np.trace(J_mean))
    except Exception:
        trace_val = np.nan

    try:
        det_val = float(linalg.det(J_mean))
    except Exception:
        det_val = np.nan

    try:
        frob_norm = float(linalg.norm(J_mean, 'fro'))
    except Exception:
        frob_norm = np.nan

    result = {
        # Eigenvalue summaries (always returned)
        'jacobian_max_eigenvalue_real': max_real,
        'jacobian_min_eigenvalue_real': min_real,
        'jacobian_mean_eigenvalue_real': mean_real,
        'jacobian_n_positive_eigenvalues': n_positive,
        'jacobian_n_negative_eigenvalues': n_negative,

        # Matrix summaries
        'jacobian_trace': trace_val,
        'jacobian_determinant': det_val,
        'jacobian_frobenius_norm': frob_norm,

        # Stability classification
        'jacobian_stability': 'unstable' if max_real > 0.01 else ('stable' if max_real < -0.01 else 'marginal'),
        'jacobian_has_oscillation': bool(np.any(np.abs(imag_parts) > 1e-10)),

        # Confidence metrics
        'jacobian_n_samples': len(jacobians),
        'jacobian_eigenvalue_std': float(np.std(np.real(all_eigenvalues))),

        # Embedding parameters used
        'embedding_dim': embedding_dim,
        'tau': tau,
        'n_neighbors': n_neighbors,

        'status': 'success',
        'error': None
    }

    # Verbose output
    if VERBOSE:
        result.update({
            'jacobian_eigenvalues_real': real_parts.tolist(),
            'jacobian_eigenvalues_imag': imag_parts.tolist(),
            'jacobian_matrix_mean': J_mean.tolist(),
            'jacobian_all_eigenvalues_real': np.real(all_eigenvalues).tolist(),
        })

    return result


def _estimate_local_jacobian(
    embedded: np.ndarray,
    idx: int,
    n_neighbors: int,
    dt: float
) -> Optional[np.ndarray]:
    """
    Estimate Jacobian at point idx using local linear fit.

    Uses least-squares fit: x(t+dt) = J @ x(t) + c
    """
    try:
        n_points = embedded.shape[0]

        # Get reference point
        x0 = embedded[idx]

        # Find nearest neighbors (excluding self and immediate successor)
        distances = np.linalg.norm(embedded - x0, axis=1)
        distances[idx] = np.inf
        if idx + 1 < n_points:
            distances[idx + 1] = np.inf

        # Get valid neighbor indices (must have a successor)
        valid_mask = np.arange(n_points) < n_points - 1
        distances[~valid_mask] = np.inf

        neighbor_idx = np.argsort(distances)[:n_neighbors]

        # Check we have enough valid neighbors
        if len(neighbor_idx) < n_neighbors:
            return None

        # Build system for least squares
        # We want: x(t+1) - x(t) ≈ J @ (x(t) - x0)
        X = embedded[neighbor_idx] - x0  # deviations from reference
        Y = embedded[neighbor_idx + 1] - embedded[neighbor_idx]  # time derivatives

        # Check for degenerate cases
        if np.linalg.matrix_rank(X) < min(X.shape):
            return None

        # Solve Y = X @ J.T  =>  J.T = lstsq(X, Y)
        J_T, residuals, rank, s = linalg.lstsq(X, Y)
        J = J_T.T / dt

        # Validate result
        if not np.all(np.isfinite(J)):
            return None

        return J

    except Exception:
        return None


def _null_result(error_msg: str) -> Dict[str, Any]:
    """Return null result with error message."""
    return {
        'jacobian_max_eigenvalue_real': np.nan,
        'jacobian_min_eigenvalue_real': np.nan,
        'jacobian_mean_eigenvalue_real': np.nan,
        'jacobian_n_positive_eigenvalues': 0,
        'jacobian_n_negative_eigenvalues': 0,
        'jacobian_trace': np.nan,
        'jacobian_determinant': np.nan,
        'jacobian_frobenius_norm': np.nan,
        'jacobian_stability': 'unknown',
        'jacobian_has_oscillation': False,
        'jacobian_n_samples': 0,
        'jacobian_eigenvalue_std': np.nan,
        'embedding_dim': None,
        'tau': None,
        'n_neighbors': None,
        'status': 'failed',
        'error': error_msg
    }


# Alias for consistency with other engines
compute_jacobian = compute

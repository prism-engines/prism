"""
Lyapunov Exponent Estimation

Computes Lyapunov exponents which characterize the rate of separation
of infinitesimally close trajectories:
    - lambda > 0: chaotic (exponential divergence)
    - lambda ~ 0: periodic/quasiperiodic
    - lambda < 0: stable fixed point (convergence)

References:
    Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
    "A practical method for calculating largest Lyapunov exponents
    from small data sets."
"""

import numpy as np
from scipy.spatial import KDTree

from .reconstruction import embed_time_series, optimal_delay, optimal_embedding_dim


def largest_lyapunov_exponent(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    min_tsep: int = None,
    max_iter: int = None
) -> float:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.

    The algorithm tracks the divergence of nearby trajectories over time.
    The slope of log(divergence) vs time gives the Lyapunov exponent.

    Parameters
    ----------
    x : array
        Time series (1D)
    tau : int, optional
        Time delay for embedding (auto-computed if None)
    dim : int, optional
        Embedding dimension (auto-computed if None)
    min_tsep : int, optional
        Minimum temporal separation for neighbors (avoids correlated points)
    max_iter : int, optional
        Maximum iterations for divergence tracking

    Returns
    -------
    lambda_max : float
        Largest Lyapunov exponent (per sample)
        > 0: chaotic
        ~ 0: periodic/quasiperiodic
        < 0: stable fixed point

    Notes
    -----
    Typical values:
        - Lorenz system: ~0.9
        - Periodic signal: ~0
        - White noise: >1

    Examples
    --------
    >>> # Periodic signal should have Lyapunov ~ 0
    >>> x = np.sin(np.linspace(0, 50*np.pi, 2000))
    >>> lyap = largest_lyapunov_exponent(x)
    >>> abs(lyap) < 0.2
    True
    """
    x = np.asarray(x).flatten()

    if len(x) < 50:
        return np.nan

    # Auto-compute embedding parameters
    if tau is None:
        tau = optimal_delay(x, max_tau=min(100, len(x) // 10))
    if dim is None:
        dim = optimal_embedding_dim(x, tau, max_dim=min(10, len(x) // (3 * tau)))
    if min_tsep is None:
        min_tsep = max(tau * dim, 10)

    # Embed the time series
    try:
        embedded = embed_time_series(x, tau, dim)
    except ValueError:
        return np.nan

    n_points = len(embedded)
    if n_points < 2 * min_tsep:
        return np.nan

    if max_iter is None:
        max_iter = min(n_points // 4, 200)

    # Build KD-tree for nearest neighbor search
    tree = KDTree(embedded)

    # Track divergence of nearest neighbors over time
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n_points - max_iter):
        # Find neighbors, excluding those within temporal window
        dists, indices = tree.query(embedded[i], k=min(n_points, 20))

        # Find first neighbor outside temporal window
        nn_idx = None
        for j, idx in enumerate(indices):
            if abs(idx - i) > min_tsep:
                nn_idx = idx
                break

        if nn_idx is None:
            continue

        # Track divergence over time
        for k in range(max_iter):
            i_k = i + k
            nn_k = nn_idx + k

            if i_k >= n_points or nn_k >= n_points:
                break

            dist = np.linalg.norm(embedded[i_k] - embedded[nn_k])
            if dist > 1e-10:  # Avoid log(0)
                divergence[k] += np.log(dist)
                counts[k] += 1

    # Average divergence curve
    valid = counts > 10  # Need sufficient samples
    if not np.any(valid):
        return np.nan

    divergence[valid] /= counts[valid]

    # Find linear region (typically early part of curve)
    # Look for region with consistent slope
    linear_end = min(max_iter // 3, 50)
    valid_range = np.where(valid[:linear_end])[0]

    if len(valid_range) < 5:
        return np.nan

    # Fit linear region
    try:
        slope, intercept = np.polyfit(valid_range, divergence[valid_range], 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    # Convert from per-iteration to per-sample rate
    return float(slope / tau)


def lyapunov_spectrum(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    n_exponents: int = None
) -> np.ndarray:
    """
    Estimate Lyapunov spectrum (multiple exponents).

    The full spectrum [lambda_1, lambda_2, ..., lambda_n] characterizes:
    - lambda_1: rate of divergence along most unstable direction
    - Sum of all lambdas: rate of phase space volume change
    - Number of positive lambdas: dimension of unstable manifold

    Parameters
    ----------
    x : array
        Time series
    tau : int, optional
        Time delay
    dim : int, optional
        Embedding dimension
    n_exponents : int, optional
        Number of exponents to compute (default: dim)

    Returns
    -------
    spectrum : array
        Lyapunov exponents in descending order [lambda_1, lambda_2, ...]

    Notes
    -----
    Full spectrum estimation is computationally expensive and less
    reliable than largest exponent for high dimensions.
    Currently returns only the largest exponent as a single-element array.

    For full spectrum, consider using specialized packages like nolds.
    """
    # For now, compute only largest exponent
    # Full spectrum (Wolf's algorithm) is more complex
    lambda_max = largest_lyapunov_exponent(x, tau, dim)

    if np.isnan(lambda_max):
        return np.array([np.nan])

    return np.array([lambda_max])


def lyapunov_from_jacobian(
    trajectory: np.ndarray,
    jacobian_func: callable,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute Lyapunov spectrum from known system dynamics (Jacobian).

    This is more accurate than time series methods when the
    governing equations are known.

    Parameters
    ----------
    trajectory : array, shape (n_points, dim)
        System trajectory
    jacobian_func : callable
        Function that returns Jacobian matrix at a point
    dt : float
        Time step

    Returns
    -------
    spectrum : array
        Lyapunov exponents
    """
    n_points, dim = trajectory.shape

    # Initialize orthonormal vectors
    Q = np.eye(dim)
    exponents = np.zeros(dim)

    for i in range(n_points):
        # Get Jacobian at current point
        J = jacobian_func(trajectory[i])

        # Evolve tangent vectors
        Q = J @ Q

        # QR decomposition to re-orthonormalize
        Q, R = np.linalg.qr(Q)

        # Accumulate exponents from diagonal of R
        exponents += np.log(np.abs(np.diag(R)))

    # Average over trajectory
    exponents /= (n_points * dt)

    return np.sort(exponents)[::-1]

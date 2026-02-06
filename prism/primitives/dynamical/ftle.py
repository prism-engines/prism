"""
Finite-Time Lyapunov Exponent (FTLE) Primitives.

FTLE measures trajectory-dependent stability over finite time horizons.
Unlike global Lyapunov which averages over the attractor, FTLE captures
local, time-varying sensitivity to perturbations.

Mathematical basis:
    FTLE(x₀, T) = (1/T) * ln(σ_max(Φ_T))

    where Φ_T is the flow map gradient (deformation tensor) and
    σ_max is its largest singular value.

Key insight: FTLE varies with position on the attractor.
Systems have "sensitive" and "stable" regions that change over time.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.spatial import KDTree


def ftle_local_linearization(
    trajectory: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = 10,
    epsilon: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FTLE using local linearization of the flow.

    This method estimates the deformation gradient Φ by fitting local
    linear models at each point. Suitable for real-time computation.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension)
    time_horizon : int
        Number of steps over which to measure stretching (T)
    n_neighbors : int
        Number of neighbors for local linearization
    epsilon : float, optional
        Neighborhood radius (auto if None)

    Returns
    -------
    ftle : np.ndarray
        FTLE values at each valid point
    confidence : np.ndarray
        Confidence scores (0-1) based on fit quality

    Notes
    -----
    FTLE > 0: Local exponential divergence (sensitive region)
    FTLE ≈ 0: Neutral stability
    FTLE < 0: Local exponential convergence (stable region)

    The FTLE field identifies Lagrangian Coherent Structures (LCS)
    which are ridges of maximum stretching.
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        raise ValueError("Trajectory must be 2D (n_points, dimension)")

    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    if n_valid < 10:
        return np.full(n_points, np.nan), np.full(n_points, 0.0)

    # Build KDTree for neighbor search
    tree = KDTree(trajectory)

    # Auto epsilon if not provided
    if epsilon is None:
        # Use 5th percentile of pairwise distances
        sample_idx = np.random.choice(n_points, min(100, n_points), replace=False)
        dists = []
        for i in sample_idx:
            for j in sample_idx:
                if i != j:
                    dists.append(np.linalg.norm(trajectory[i] - trajectory[j]))
        epsilon = np.percentile(dists, 20) if dists else 1.0

    ftle = np.full(n_points, np.nan)
    confidence = np.full(n_points, 0.0)

    for i in range(n_valid):
        # Find neighbors at time i
        indices = tree.query_ball_point(trajectory[i], epsilon)

        # Filter: need neighbors that are still valid at time i+T
        valid_neighbors = [j for j in indices if j != i and j + time_horizon < n_points]

        if len(valid_neighbors) < n_neighbors:
            # Fall back to k-nearest if not enough in radius
            _, indices = tree.query(trajectory[i], k=n_neighbors + 1)
            valid_neighbors = [j for j in indices[1:] if j + time_horizon < n_points]

        if len(valid_neighbors) < dim + 1:
            continue

        # Compute deformation gradient via least squares
        # At time 0: x_j - x_i = δx_0
        # At time T: x_j(T) - x_i(T) = δx_T
        # We want: δx_T ≈ Φ @ δx_0

        delta_x0 = np.array([trajectory[j] - trajectory[i] for j in valid_neighbors])
        delta_xT = np.array([trajectory[j + time_horizon] - trajectory[i + time_horizon]
                            for j in valid_neighbors])

        # Least squares: delta_xT = delta_x0 @ Phi.T
        try:
            Phi_T, residuals, rank, s = np.linalg.lstsq(delta_x0, delta_xT, rcond=None)
            Phi = Phi_T.T  # (dim x dim) deformation gradient

            # Compute FTLE from largest singular value
            singular_values = np.linalg.svd(Phi, compute_uv=False)
            sigma_max = singular_values[0]

            if sigma_max > 0:
                ftle[i] = (1.0 / time_horizon) * np.log(sigma_max)

            # Confidence based on fit quality
            if residuals.size > 0 and np.sum(delta_xT**2) > 0:
                r2 = 1 - np.sum(residuals) / np.sum(delta_xT**2)
                confidence[i] = max(0, min(1, r2))
            else:
                confidence[i] = 0.5  # Unknown fit quality

        except (np.linalg.LinAlgError, ValueError):
            continue

    return ftle, confidence


def ftle_direct_perturbation(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    time_horizon: int = 10,
    perturbation: float = 1e-6,
    n_perturbations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FTLE by directly perturbing initial conditions.

    Uses numerical perturbations rather than local linearization.
    More accurate but requires surrogate trajectory generation.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int
        Embedding dimension
    delay : int
        Time delay for embedding
    time_horizon : int
        Number of steps to track divergence
    perturbation : float
        Size of initial perturbation (δ)
    n_perturbations : int
        Number of random perturbation directions

    Returns
    -------
    ftle : np.ndarray
        FTLE estimate at each point
    stretching_directions : np.ndarray
        Principal stretching direction at each point
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # Embed signal
    n_points = n - (dimension - 1) * delay
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]

    n_valid = n_points - time_horizon
    ftle = np.full(n_points, np.nan)
    stretching_dirs = np.full((n_points, dimension), np.nan)

    for i in range(n_valid):
        # Generate random perturbation directions
        max_stretch = 0.0
        best_dir = np.zeros(dimension)

        for _ in range(n_perturbations):
            # Random unit vector
            direction = np.random.randn(dimension)
            direction /= np.linalg.norm(direction)

            # Perturbed initial condition
            x_pert = embedded[i] + perturbation * direction

            # Find nearest point in trajectory to perturbed state
            dists = np.linalg.norm(embedded - x_pert, axis=1)
            nearest_idx = np.argmin(dists)

            if nearest_idx + time_horizon >= n_points:
                continue

            # Measure divergence after time_horizon steps
            x_final = embedded[i + time_horizon]
            x_pert_final = embedded[nearest_idx + time_horizon]

            final_dist = np.linalg.norm(x_final - x_pert_final)

            if final_dist > max_stretch:
                max_stretch = final_dist
                best_dir = direction

        if max_stretch > 0 and perturbation > 0:
            ftle[i] = (1.0 / time_horizon) * np.log(max_stretch / perturbation)
            stretching_dirs[i] = best_dir

    return ftle, stretching_dirs


def compute_cauchy_green_tensor(
    trajectory: np.ndarray,
    time_horizon: int,
    n_neighbors: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the right Cauchy-Green deformation tensor at each point.

    The Cauchy-Green tensor C = Φᵀ @ Φ captures the stretching
    without rotation. Its eigenvalues give the principal stretches.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension)
    time_horizon : int
        Time steps for flow map
    n_neighbors : int
        Neighbors for local estimation

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of C at each point (n_points, dimension)
    eigenvectors : np.ndarray
        Eigenvectors of C at each point (n_points, dimension, dimension)
    ftle : np.ndarray
        FTLE from largest eigenvalue

    Notes
    -----
    The FTLE is: λ = (1/2T) * ln(λ_max(C))
    where λ_max(C) = σ_max²(Φ)
    """
    trajectory = np.asarray(trajectory)
    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    eigenvalues = np.full((n_points, dim), np.nan)
    eigenvectors = np.full((n_points, dim, dim), np.nan)
    ftle = np.full(n_points, np.nan)

    tree = KDTree(trajectory)

    for i in range(n_valid):
        # Find neighbors
        _, indices = tree.query(trajectory[i], k=n_neighbors + 1)
        valid_neighbors = [j for j in indices[1:] if j + time_horizon < n_points]

        if len(valid_neighbors) < dim + 1:
            continue

        # Compute deformation gradient
        delta_x0 = np.array([trajectory[j] - trajectory[i] for j in valid_neighbors])
        delta_xT = np.array([trajectory[j + time_horizon] - trajectory[i + time_horizon]
                            for j in valid_neighbors])

        try:
            Phi_T, _, _, _ = np.linalg.lstsq(delta_x0, delta_xT, rcond=None)
            Phi = Phi_T.T

            # Cauchy-Green tensor: C = Φᵀ @ Φ
            C = Phi.T @ Phi

            # Eigendecomposition (C is symmetric positive semi-definite)
            evals, evecs = np.linalg.eigh(C)

            # Sort descending
            idx = np.argsort(evals)[::-1]
            eigenvalues[i] = evals[idx]
            eigenvectors[i] = evecs[:, idx]

            # FTLE from largest eigenvalue
            if evals[idx[0]] > 0:
                ftle[i] = (1.0 / (2 * time_horizon)) * np.log(evals[idx[0]])

        except (np.linalg.LinAlgError, ValueError):
            continue

    return eigenvalues, eigenvectors, ftle


def detect_lcs_ridges(
    ftle_field: np.ndarray,
    trajectory: np.ndarray,
    threshold_percentile: float = 90,
) -> np.ndarray:
    """
    Detect Lagrangian Coherent Structures as ridges of FTLE field.

    LCS are material lines that organize the flow. They appear as
    ridges (local maxima in some direction) of the FTLE field.

    Parameters
    ----------
    ftle_field : np.ndarray
        FTLE values at trajectory points
    trajectory : np.ndarray
        Embedded trajectory
    threshold_percentile : float
        Percentile threshold for ridge detection

    Returns
    -------
    is_lcs : np.ndarray
        Boolean mask indicating LCS points
    """
    ftle_field = np.asarray(ftle_field)
    valid_mask = ~np.isnan(ftle_field)

    if np.sum(valid_mask) < 10:
        return np.zeros_like(ftle_field, dtype=bool)

    # Threshold on FTLE magnitude
    threshold = np.nanpercentile(ftle_field, threshold_percentile)
    high_ftle = ftle_field > threshold

    # Ridge detection: point is ridge if it's a local maximum
    # in at least one direction
    tree = KDTree(trajectory)
    is_ridge = np.zeros_like(ftle_field, dtype=bool)

    for i in np.where(high_ftle)[0]:
        if np.isnan(ftle_field[i]):
            continue

        # Check if local maximum among neighbors
        _, indices = tree.query(trajectory[i], k=10)
        neighbor_ftle = ftle_field[indices[1:]]  # Exclude self
        valid_neighbors = ~np.isnan(neighbor_ftle)

        if np.sum(valid_neighbors) > 0:
            if ftle_field[i] >= np.nanmax(neighbor_ftle):
                is_ridge[i] = True

    return is_ridge

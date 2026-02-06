"""
Saddle Point Detection Primitives.

Saddle points are unstable equilibria in phase space where the Jacobian
has both positive and negative real eigenvalues (in continuous systems)
or eigenvalues inside and outside the unit circle (in discrete systems).

Near saddle points, systems exhibit:
- Sensitive dependence on initial conditions
- Transient dynamics along stable/unstable manifolds
- Critical transitions if the system crosses the separatrix

Mathematical basis:
    Saddle point x*: f(x*) = 0 and Df(x*) has mixed-sign eigenvalues

For time series, we detect saddle proximity from:
    1. Low velocity (near equilibrium)
    2. Mixed Jacobian eigenvalue signs
    3. High FTLE (divergence nearby)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.spatial import KDTree


def estimate_jacobian_local(
    trajectory: np.ndarray,
    point_idx: int,
    n_neighbors: int = None,
) -> np.ndarray:
    """
    Estimate local Jacobian from trajectory using linear regression.

    The Jacobian J is estimated from the map x_{t+1} = f(x_t) by fitting:
        x_{j+1} - x_{i+1} ≈ J @ (x_j - x_i)

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension)
    point_idx : int
        Index of point to estimate Jacobian at
    n_neighbors : int
        Number of neighbors for estimation

    Returns
    -------
    jacobian : np.ndarray
        Estimated Jacobian matrix (dim x dim)
    """
    trajectory = np.asarray(trajectory)
    n_points, dim = trajectory.shape

    if n_neighbors is None:
        n_neighbors = 2 * dim + 1

    if point_idx >= n_points - 1:
        return np.full((dim, dim), np.nan)

    # Find neighbors
    tree = KDTree(trajectory)
    _, indices = tree.query(trajectory[point_idx], k=n_neighbors + 1)

    # Filter out self and points at end
    valid = [j for j in indices[1:] if j < n_points - 1]

    if len(valid) < dim + 1:
        return np.full((dim, dim), np.nan)

    # Build regression matrices
    X = np.array([trajectory[j] - trajectory[point_idx] for j in valid])
    Y = np.array([trajectory[j + 1] - trajectory[point_idx + 1] for j in valid])

    try:
        J_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return J_T.T
    except np.linalg.LinAlgError:
        return np.full((dim, dim), np.nan)


def classify_jacobian_eigenvalues(
    jacobian: np.ndarray,
) -> Dict[str, Any]:
    """
    Classify fixed point type from Jacobian eigenvalues.

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix

    Returns
    -------
    dict with:
        eigenvalues: Complex eigenvalues
        eigenvalues_real: Real parts
        eigenvalues_imag: Imaginary parts
        n_positive: Count of positive real eigenvalues
        n_negative: Count of negative real eigenvalues
        n_zero: Count of near-zero real eigenvalues
        is_saddle: True if mixed signs (discrete: mixed inside/outside unit circle)
        is_stable: True if all eigenvalues have negative real part
        is_unstable: True if any eigenvalue has positive real part
        stability_type: 'saddle', 'stable_node', 'stable_focus', 'unstable_node', etc.
    """
    jacobian = np.asarray(jacobian)

    if np.any(np.isnan(jacobian)):
        return {
            'eigenvalues': np.array([]),
            'eigenvalues_real': np.array([]),
            'eigenvalues_imag': np.array([]),
            'n_positive': 0,
            'n_negative': 0,
            'n_zero': 0,
            'is_saddle': False,
            'is_stable': False,
            'is_unstable': False,
            'stability_type': 'unknown',
        }

    try:
        eigenvalues = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        return {
            'eigenvalues': np.array([]),
            'eigenvalues_real': np.array([]),
            'eigenvalues_imag': np.array([]),
            'n_positive': 0,
            'n_negative': 0,
            'n_zero': 0,
            'is_saddle': False,
            'is_stable': False,
            'is_unstable': False,
            'stability_type': 'unknown',
        }

    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    # Tolerance for "zero"
    tol = 1e-6

    n_positive = np.sum(real_parts > tol)
    n_negative = np.sum(real_parts < -tol)
    n_zero = np.sum(np.abs(real_parts) <= tol)

    # For discrete maps, use unit circle criterion
    magnitudes = np.abs(eigenvalues)
    n_outside_unit = np.sum(magnitudes > 1 + tol)
    n_inside_unit = np.sum(magnitudes < 1 - tol)

    # Classification
    has_complex = np.any(np.abs(imag_parts) > tol)

    # Saddle: mixed positive/negative (continuous) or mixed inside/outside unit circle (discrete)
    is_saddle = (n_positive > 0 and n_negative > 0) or (n_outside_unit > 0 and n_inside_unit > 0)

    # Stability
    is_stable = n_positive == 0 and n_outside_unit == 0
    is_unstable = n_positive > 0 or n_outside_unit > 0

    # Classify type
    if is_saddle:
        stability_type = 'saddle'
    elif is_stable:
        if has_complex:
            stability_type = 'stable_focus'
        else:
            stability_type = 'stable_node'
    elif n_positive > 0 and n_negative == 0:
        if has_complex:
            stability_type = 'unstable_focus'
        else:
            stability_type = 'unstable_node'
    elif n_zero > 0:
        stability_type = 'center_manifold'
    else:
        stability_type = 'unknown'

    return {
        'eigenvalues': eigenvalues,
        'eigenvalues_real': real_parts,
        'eigenvalues_imag': imag_parts,
        'n_positive': int(n_positive),
        'n_negative': int(n_negative),
        'n_zero': int(n_zero),
        'is_saddle': is_saddle,
        'is_stable': is_stable,
        'is_unstable': is_unstable,
        'stability_type': stability_type,
    }


def detect_saddle_points(
    trajectory: np.ndarray,
    velocity_threshold: float = 0.1,
    n_neighbors: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Detect saddle point regions in a trajectory.

    Saddle points are identified by:
    1. Low velocity (near equilibrium)
    2. Jacobian with mixed-sign eigenvalues

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension)
    velocity_threshold : float
        Normalized velocity threshold for "near equilibrium"
        (relative to max velocity)
    n_neighbors : int
        Neighbors for Jacobian estimation

    Returns
    -------
    saddle_score : np.ndarray
        Saddle proximity score at each point (0-1)
    velocity : np.ndarray
        Velocity magnitude at each point
    saddle_info : list
        Detailed eigenvalue information at each point
    """
    trajectory = np.asarray(trajectory)
    n_points, dim = trajectory.shape

    if n_neighbors is None:
        n_neighbors = 2 * dim + 1

    # Compute velocity (finite differences)
    velocity = np.full(n_points, np.nan)
    for i in range(n_points - 1):
        velocity[i] = np.linalg.norm(trajectory[i + 1] - trajectory[i])

    # Normalize velocity
    max_vel = np.nanmax(velocity)
    if max_vel > 0:
        velocity_normalized = velocity / max_vel
    else:
        velocity_normalized = np.zeros_like(velocity)

    # Low velocity mask
    low_velocity = velocity_normalized < velocity_threshold

    # Compute saddle score
    saddle_score = np.zeros(n_points)
    saddle_info = []

    for i in range(n_points - 1):
        info = {'point_idx': i}

        # Estimate Jacobian
        J = estimate_jacobian_local(trajectory, i, n_neighbors)
        info['jacobian'] = J

        # Classify eigenvalues
        eig_info = classify_jacobian_eigenvalues(J)
        info.update(eig_info)

        # Saddle score: combination of low velocity and mixed eigenvalues
        velocity_factor = 1.0 - velocity_normalized[i] if not np.isnan(velocity_normalized[i]) else 0
        eigenvalue_factor = 1.0 if eig_info['is_saddle'] else 0.0

        # Also consider unstable points with some directions stable
        if eig_info['n_positive'] > 0 and eig_info['n_negative'] > 0:
            eigenvalue_factor = 1.0
        elif eig_info['is_unstable'] and not eig_info['is_saddle']:
            eigenvalue_factor = 0.5

        saddle_score[i] = velocity_factor * eigenvalue_factor

        saddle_info.append(info)

    # Add empty info for last point
    saddle_info.append({'point_idx': n_points - 1, 'stability_type': 'unknown'})

    return saddle_score, velocity, saddle_info


def compute_separatrix_distance(
    trajectory: np.ndarray,
    saddle_indices: np.ndarray,
    stable_direction: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate distance to separatrix (stable manifold of saddle).

    The separatrix separates basins of attraction. Crossing it
    leads to a different attractor (tipping).

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory
    saddle_indices : np.ndarray
        Indices of saddle points
    stable_direction : np.ndarray, optional
        Estimated stable eigenvector of saddle

    Returns
    -------
    distance : np.ndarray
        Estimated distance to separatrix at each point
    """
    trajectory = np.asarray(trajectory)
    n_points = len(trajectory)

    if len(saddle_indices) == 0:
        return np.full(n_points, np.nan)

    # Use saddle points as approximation of separatrix location
    saddle_points = trajectory[saddle_indices]

    # Build KDTree of saddle points
    tree = KDTree(saddle_points)

    # Query distance to nearest saddle
    distances, _ = tree.query(trajectory, k=1)

    return distances


def compute_basin_stability(
    trajectory: np.ndarray,
    saddle_score: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """
    Estimate basin stability from saddle proximity over time.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory
    saddle_score : np.ndarray
        Saddle proximity score (from detect_saddle_points)
    window : int
        Rolling window for stability estimation

    Returns
    -------
    stability : np.ndarray
        Basin stability score (0-1, higher = more stable)
    """
    n_points = len(saddle_score)
    stability = np.full(n_points, np.nan)

    for i in range(window, n_points):
        window_scores = saddle_score[i - window:i]
        valid = ~np.isnan(window_scores)

        if np.sum(valid) > 0:
            # Stability is inverse of mean saddle proximity
            stability[i] = 1.0 - np.mean(window_scores[valid])

    return stability

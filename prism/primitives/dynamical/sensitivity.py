"""
Trajectory Sensitivity Primitives.

Computes which variables (signals) are most sensitive at the current
state and time. Unlike global importance measures, trajectory sensitivity
varies with position on the attractor.

Mathematical basis:
    Sensitivity_i(x, t) = |∂FTLE/∂x_i| or ||J[:, i]||

Key insight: The "most important" variable changes depending on
where the system is in state space.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.spatial import KDTree


def compute_variable_sensitivity(
    trajectory: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sensitivity of each variable to perturbations.

    At each point, measures how much each variable contributes
    to trajectory divergence.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory (n_points, dimension)
    time_horizon : int
        Time steps for measuring divergence
    n_neighbors : int
        Neighbors for local estimation

    Returns
    -------
    sensitivity : np.ndarray
        Sensitivity score per variable at each point (n_points, dim)
    rank : np.ndarray
        Variable rank at each point (1 = most sensitive)
    """
    trajectory = np.asarray(trajectory)
    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    sensitivity = np.full((n_points, dim), np.nan)
    rank = np.full((n_points, dim), np.nan)

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
            Phi = Phi_T.T  # Deformation gradient

            # Sensitivity: column norm of Φ
            # High norm = perturbations in that direction get amplified
            for d in range(dim):
                sensitivity[i, d] = np.linalg.norm(Phi[:, d])

            # Rank variables (1 = most sensitive)
            sorted_idx = np.argsort(-sensitivity[i])  # Descending
            for r, idx in enumerate(sorted_idx):
                rank[i, idx] = r + 1

        except (np.linalg.LinAlgError, ValueError):
            continue

    return sensitivity, rank


def compute_directional_sensitivity(
    trajectory: np.ndarray,
    direction: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    Compute sensitivity along a specific direction.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory
    direction : np.ndarray
        Direction vector (normalized internally)
    time_horizon : int
        Time steps for divergence
    n_neighbors : int
        Neighbors for estimation

    Returns
    -------
    sensitivity : np.ndarray
        Directional sensitivity at each point
    """
    trajectory = np.asarray(trajectory)
    direction = np.asarray(direction).flatten()
    direction = direction / np.linalg.norm(direction)

    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    sensitivity = np.full(n_points, np.nan)
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

            # Directional sensitivity: ||Φ @ direction||
            amplified = Phi @ direction
            sensitivity[i] = np.linalg.norm(amplified)

        except (np.linalg.LinAlgError, ValueError):
            continue

    return sensitivity


def compute_sensitivity_evolution(
    sensitivity: np.ndarray,
    window: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Analyze how variable sensitivity evolves over time.

    Parameters
    ----------
    sensitivity : np.ndarray
        Sensitivity scores (n_points, dim)
    window : int
        Rolling window size

    Returns
    -------
    dict with:
        mean_sensitivity: Rolling mean per variable
        sensitivity_std: Rolling std per variable
        dominant_variable: Index of most sensitive variable per window
        sensitivity_entropy: Entropy of sensitivity distribution (low = one dominant)
    """
    n_points, dim = sensitivity.shape

    mean_sens = np.full((n_points, dim), np.nan)
    std_sens = np.full((n_points, dim), np.nan)
    dominant_var = np.full(n_points, np.nan)
    entropy = np.full(n_points, np.nan)

    for i in range(window, n_points):
        window_data = sensitivity[i - window:i]
        valid = np.all(~np.isnan(window_data), axis=1)

        if np.sum(valid) < 5:
            continue

        valid_data = window_data[valid]

        for d in range(dim):
            mean_sens[i, d] = np.mean(valid_data[:, d])
            std_sens[i, d] = np.std(valid_data[:, d])

        # Dominant variable
        mean_at_i = mean_sens[i]
        if not np.all(np.isnan(mean_at_i)):
            dominant_var[i] = np.nanargmax(mean_at_i)

        # Sensitivity entropy (normalized distribution)
        total = np.nansum(mean_at_i)
        if total > 0:
            p = mean_at_i / total
            p = p[p > 0]  # Remove zeros
            if len(p) > 1:
                entropy[i] = -np.sum(p * np.log(p)) / np.log(dim)

    return {
        'mean_sensitivity': mean_sens,
        'sensitivity_std': std_sens,
        'dominant_variable': dominant_var,
        'sensitivity_entropy': entropy,
    }


def detect_sensitivity_transitions(
    sensitivity: np.ndarray,
    rank: np.ndarray,
    window: int = 20,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Detect transitions in which variable is most sensitive.

    Parameters
    ----------
    sensitivity : np.ndarray
        Sensitivity scores (n_points, dim)
    rank : np.ndarray
        Variable ranks (n_points, dim)
    window : int
        Window for transition detection

    Returns
    -------
    transition_points : np.ndarray
        Indices where dominant variable changes
    transitions : list
        Details about each transition
    """
    n_points, dim = sensitivity.shape

    # Compute dominant variable at each point
    dominant = np.full(n_points, -1)
    for i in range(n_points):
        if not np.all(np.isnan(sensitivity[i])):
            dominant[i] = np.nanargmax(sensitivity[i])

    # Find transitions
    transition_points = []
    transitions = []

    prev_dominant = dominant[0]
    for i in range(1, n_points):
        if dominant[i] != prev_dominant and dominant[i] >= 0 and prev_dominant >= 0:
            transition_points.append(i)
            transitions.append({
                'index': i,
                'from_variable': int(prev_dominant),
                'to_variable': int(dominant[i]),
                'sensitivity_before': sensitivity[i - 1].copy() if i > 0 else None,
                'sensitivity_after': sensitivity[i].copy(),
            })
        if dominant[i] >= 0:
            prev_dominant = dominant[i]

    return np.array(transition_points), transitions


def compute_influence_matrix(
    trajectory: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    Compute time-averaged influence matrix.

    Entry (i, j) shows how much variable j influences the future of variable i.

    Parameters
    ----------
    trajectory : np.ndarray
        Embedded trajectory
    time_horizon : int
        Time steps for influence
    n_neighbors : int
        Neighbors for estimation

    Returns
    -------
    influence : np.ndarray
        Influence matrix (dim x dim), normalized rows
    """
    trajectory = np.asarray(trajectory)
    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    influence_sum = np.zeros((dim, dim))
    count = 0

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

            # Influence: |Φ_{ij}| shows how j influences i
            influence_sum += np.abs(Phi)
            count += 1

        except (np.linalg.LinAlgError, ValueError):
            continue

    if count > 0:
        influence = influence_sum / count
        # Normalize rows
        row_sums = np.sum(influence, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        influence = influence / row_sums
        return influence
    else:
        return np.full((dim, dim), np.nan)

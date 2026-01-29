"""
Information Decomposition Primitives (114-116)

Partial Information Decomposition: redundancy, synergy, unique information.
"""

import numpy as np
from typing import List, Dict


def partial_information_decomposition(
    source_1: np.ndarray,
    source_2: np.ndarray,
    target: np.ndarray,
    n_bins: int = 8
) -> Dict[str, float]:
    """
    Partial Information Decomposition: separate redundancy, synergy, and unique info.

    Parameters
    ----------
    source_1 : np.ndarray
        First source signal
    source_2 : np.ndarray
        Second source signal
    target : np.ndarray
        Target signal
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Dictionary with:
        - 'redundancy': Information both sources provide
        - 'unique_1': Information only source_1 provides
        - 'unique_2': Information only source_2 provides
        - 'synergy': Information only both together provide
        - 'total': Total information about target

    Notes
    -----
    PID (Williams & Beer, 2010) decomposes the information that
    (X₁, X₂) provide about Y into four atoms:

    I(X₁,X₂ ; Y) = Redundancy + Unique₁ + Unique₂ + Synergy

    - Redundancy: Information available from EITHER source alone
    - Unique₁: Information ONLY X₁ provides
    - Unique₂: Information ONLY X₂ provides
    - Synergy: Information available ONLY when both sources known

    Physical interpretation:

    REDUNDANCY: "Both pressure and flow tell me about temperature"
    → Sensors are measuring related aspects
    → Can lose one sensor without losing info

    UNIQUE: "Only vibration tells me about bearing state"
    → This sensor has irreplaceable information

    SYNERGY: "Pressure AND flow together tell me about efficiency,
              but neither alone does"
    → EMERGENCE! The whole is greater than the sum of parts
    → This is what pairwise analysis MISSES
    """
    source_1 = np.asarray(source_1).flatten()
    source_2 = np.asarray(source_2).flatten()
    target = np.asarray(target).flatten()

    # Align
    n = min(len(source_1), len(source_2), len(target))
    source_1 = source_1[:n]
    source_2 = source_2[:n]
    target = target[:n]

    # Remove NaN
    valid = ~(np.isnan(source_1) | np.isnan(source_2) | np.isnan(target))
    source_1 = source_1[valid]
    source_2 = source_2[valid]
    target = target[valid]

    if len(source_1) < 20:
        return {
            'redundancy': 0.0,
            'unique_1': 0.0,
            'unique_2': 0.0,
            'synergy': 0.0,
            'total': 0.0,
        }

    # Compute mutual informations
    I_1_T = _mutual_information(source_1, target, n_bins)
    I_2_T = _mutual_information(source_2, target, n_bins)

    # Joint mutual information I(X1,X2 ; T)
    combined = np.column_stack([source_1, source_2])
    I_12_T = _joint_mutual_information(combined, target, n_bins)

    # Compute PID using minimum mutual information (MMI) as redundancy measure
    # This is an approximation (exact PID is computationally complex)

    # Redundancy (MMI): min(I(X1;T), I(X2;T))
    redundancy = min(I_1_T, I_2_T)

    # Unique information
    unique_1 = max(0, I_1_T - redundancy)
    unique_2 = max(0, I_2_T - redundancy)

    # Synergy: What's left
    # I(X1,X2;T) = Redundancy + Unique_1 + Unique_2 + Synergy
    synergy = I_12_T - redundancy - unique_1 - unique_2
    synergy = max(0, synergy)  # Ensure non-negative

    return {
        'redundancy': float(redundancy),
        'unique_1': float(unique_1),
        'unique_2': float(unique_2),
        'synergy': float(synergy),
        'total': float(I_12_T),
    }


def redundancy(
    sources: List[np.ndarray],
    target: np.ndarray,
    n_bins: int = 8
) -> float:
    """
    Compute redundancy among multiple sources about a target.

    Parameters
    ----------
    sources : list of np.ndarray
        List of source signals
    target : np.ndarray
        Target signal
    n_bins : int
        Number of bins

    Returns
    -------
    float
        Redundancy in bits

    Notes
    -----
    Uses minimum mutual information (MMI) as redundancy measure.

    Redundancy = min over all sources of I(source ; target)

    Physical interpretation:
    "Information that ALL sources provide"

    High redundancy:
    - Sensors are measuring the same thing
    - Can lose some sensors without losing info
    - May indicate opportunity for sensor reduction

    Low redundancy:
    - Sensors provide unique information
    - All sensors are necessary
    """
    if not sources:
        return 0.0

    target = np.asarray(target).flatten()

    mi_values = []
    for source in sources:
        source = np.asarray(source).flatten()

        # Align
        n = min(len(source), len(target))
        mi = _mutual_information(source[:n], target[:n], n_bins)
        mi_values.append(mi)

    return float(min(mi_values))


def synergy(
    sources: List[np.ndarray],
    target: np.ndarray,
    n_bins: int = 8
) -> float:
    """
    Compute synergy among multiple sources about a target.

    Parameters
    ----------
    sources : list of np.ndarray
        List of source signals
    target : np.ndarray
        Target signal
    n_bins : int
        Number of bins

    Returns
    -------
    float
        Synergy in bits

    Notes
    -----
    Synergy = I(all sources ; target) - Σ I(source_i ; target) + (k-1) × Redundancy

    This is an approximation based on the inclusion-exclusion principle.

    Physical interpretation:
    "Information available ONLY when all sources are known together"

    THE WHOLE IS GREATER THAN THE SUM OF PARTS.

    High synergy indicates EMERGENCE:
    - Individual signals don't predict target
    - But COMBINATION of signals does
    - This is what pairwise analysis MISSES

    Examples of synergistic systems:
    - XOR gate: Neither input alone predicts output
    - Chemical reactions: Reactants alone don't predict product
    - System failures: Multiple precursors combine
    """
    if len(sources) < 2:
        return 0.0

    target = np.asarray(target).flatten()

    # Align all sources
    n = min(len(target), *[len(np.asarray(s).flatten()) for s in sources])
    sources_aligned = [np.asarray(s).flatten()[:n] for s in sources]
    target = target[:n]

    # Compute individual MIs
    individual_mi = sum(_mutual_information(s, target, n_bins) for s in sources_aligned)

    # Compute joint MI (all sources together)
    combined = np.column_stack(sources_aligned)
    joint_mi = _joint_mutual_information(combined, target, n_bins)

    # Compute redundancy
    red = redundancy(sources, target[:n], n_bins)

    # Synergy estimate
    k = len(sources)
    syn = joint_mi - individual_mi + (k - 1) * red

    return float(max(0, syn))


def information_atoms(
    sources: List[np.ndarray],
    target: np.ndarray,
    n_bins: int = 8
) -> Dict[str, float]:
    """
    Compute all information atoms for multiple sources.

    Parameters
    ----------
    sources : list of np.ndarray
        List of source signals
    target : np.ndarray
        Target signal
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Dictionary with redundancy, synergy, total MI, and emergence ratio
    """
    if len(sources) < 2:
        return {
            'redundancy': 0.0,
            'synergy': 0.0,
            'total_mi': 0.0,
            'emergence_ratio': 0.0,
        }

    target = np.asarray(target).flatten()
    n = min(len(target), *[len(np.asarray(s).flatten()) for s in sources])
    sources_aligned = [np.asarray(s).flatten()[:n] for s in sources]
    target = target[:n]

    # Compute metrics
    red = redundancy(sources_aligned, target, n_bins)
    syn = synergy(sources_aligned, target, n_bins)

    combined = np.column_stack(sources_aligned)
    total_mi = _joint_mutual_information(combined, target, n_bins)

    emergence_ratio = syn / total_mi if total_mi > 0 else 0.0

    return {
        'redundancy': float(red),
        'synergy': float(syn),
        'total_mi': float(total_mi),
        'emergence_ratio': float(emergence_ratio),
    }


# Helper functions

def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute mutual information between two arrays."""
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        return 0.0

    # Discretize
    def discretize(s):
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return np.zeros(len(s), dtype=int)
        return np.digitize(s, np.linspace(s_min, s_max + 1e-10, n_bins + 1)[:-1]) - 1

    x_d = discretize(x)
    y_d = discretize(y)

    # Compute entropies
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    def joint_entropy(arr1, arr2):
        combined = np.column_stack([arr1, arr2])
        _, counts = np.unique(combined, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    h_x = entropy(x_d)
    h_y = entropy(y_d)
    h_xy = joint_entropy(x_d, y_d)

    mi = h_x + h_y - h_xy
    return max(0, mi)


def _joint_mutual_information(combined: np.ndarray, target: np.ndarray, n_bins: int) -> float:
    """Compute mutual information between combined sources and target."""
    # Remove NaN rows
    valid = ~(np.any(np.isnan(combined), axis=1) | np.isnan(target))
    combined = combined[valid]
    target = target[valid]

    if len(target) < 10:
        return 0.0

    # Discretize combined sources
    def discretize_nd(arr, bins):
        state = np.zeros(len(arr), dtype=int)
        for col in range(arr.shape[1]):
            col_data = arr[:, col]
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                col_d = np.digitize(col_data, np.linspace(col_min, col_max + 1e-10, bins + 1)[:-1]) - 1
            else:
                col_d = np.zeros(len(col_data), dtype=int)
            state = state * bins + col_d
        return state

    combined_d = discretize_nd(combined, n_bins)

    # Discretize target
    t_min, t_max = target.min(), target.max()
    if t_max > t_min:
        target_d = np.digitize(target, np.linspace(t_min, t_max + 1e-10, n_bins + 1)[:-1]) - 1
    else:
        target_d = np.zeros(len(target), dtype=int)

    # Compute entropies
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    def joint_entropy(arr1, arr2):
        combined_arr = np.column_stack([arr1, arr2])
        _, counts = np.unique(combined_arr, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    h_combined = entropy(combined_d)
    h_target = entropy(target_d)
    h_joint = joint_entropy(combined_d, target_d)

    mi = h_combined + h_target - h_joint
    return max(0, mi)

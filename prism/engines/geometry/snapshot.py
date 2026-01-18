"""
Geometry Snapshot
=================

Compute system geometry at a single timestamp from Laplace fields.
"""

import numpy as np
from typing import Dict, List

from prism.core.signals.types import LaplaceField, GeometrySnapshot
from prism.engines.geometry.coupling import compute_coupling_matrix
from prism.engines.geometry.divergence import compute_divergence
from prism.engines.geometry.modes import discover_modes


def compute_geometry_at_t(
    fields: Dict[str, LaplaceField],
    t: float,
    s_values: np.ndarray = None,
) -> GeometrySnapshot:
    """
    Compute full geometry snapshot at timestamp t.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        t: Timestamp to compute geometry at
        s_values: Laplace s-values for comparison (if None, use from fields)

    Returns:
        GeometrySnapshot with coupling, divergence, modes
    """
    signal_ids = sorted(fields.keys())
    n_signals = len(signal_ids)

    if n_signals == 0:
        return GeometrySnapshot(
            timestamp=t,
            coupling_matrix=np.array([[]]),
            divergence=0.0,
            mode_labels=np.array([]),
            mode_coherence=np.array([]),
            signal_ids=[],
        )

    # Get s_values from first field if not provided
    if s_values is None:
        s_values = list(fields.values())[0].s_values

    # Get F(s) for each signal at time t
    field_vectors = np.zeros((n_signals, len(s_values)), dtype=np.complex128)

    for i, sid in enumerate(signal_ids):
        field_vectors[i] = fields[sid].at(t)

    # Compute pairwise coupling
    coupling_matrix = compute_coupling_matrix(
        field_vectors,
        s_values,
        fields,
        signal_ids,
    )

    # Compute divergence
    divergence = compute_divergence(field_vectors, s_values)

    # Discover modes
    mode_labels, mode_coherence = discover_modes(field_vectors, s_values)

    return GeometrySnapshot(
        timestamp=t,
        coupling_matrix=coupling_matrix,
        divergence=divergence,
        mode_labels=mode_labels,
        mode_coherence=mode_coherence,
        signal_ids=signal_ids,
    )


def compute_geometry_trajectory(
    fields: Dict[str, LaplaceField],
    timestamps: np.ndarray,
    s_values: np.ndarray = None,
) -> List[GeometrySnapshot]:
    """
    Compute geometry at each timestamp.

    Args:
        fields: All Laplace fields
        timestamps: Times to compute geometry
        s_values: Laplace s-values

    Returns:
        List of GeometrySnapshot, one per timestamp
    """
    snapshots = []

    for t in timestamps:
        snapshot = compute_geometry_at_t(fields, float(t), s_values)
        snapshots.append(snapshot)

    return snapshots


def snapshot_to_vector(snapshot: GeometrySnapshot) -> np.ndarray:
    """
    Convert GeometrySnapshot to a flat vector for state trajectory.

    Uses: [flattened coupling upper triangle, divergence, mode coherence]
    """
    if snapshot.n_signals == 0:
        return np.array([snapshot.divergence])

    # Upper triangle of coupling matrix (excluding diagonal)
    n = snapshot.coupling_matrix.shape[0]
    upper_tri = snapshot.coupling_matrix[np.triu_indices(n, k=1)]

    # Build position vector
    pos = np.concatenate([
        upper_tri,
        [snapshot.divergence],
        snapshot.mode_coherence,
    ])

    return pos


def get_unified_timestamps(fields: Dict[str, LaplaceField]) -> np.ndarray:
    """
    Get unified timestamp grid from all fields.

    Uses union of all timestamps.
    """
    all_timestamps = set()
    for field in fields.values():
        all_timestamps.update(field.timestamps.astype(float))
    return np.array(sorted(all_timestamps))

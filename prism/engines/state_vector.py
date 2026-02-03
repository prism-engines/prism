"""
PRISM State Vector Engine

Computes the CENTROID (mean position) of signals in feature space.
This is the "where" - the average position of all signals at each I.

Eigenvalues (the "shape") are computed in state_geometry.py.

Pipeline:
    signal_vector.parquet → state_vector.parquet → state_geometry.parquet

Credit: Avery Rudder - insight that Laplace transform IS the state engine.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any


# Feature groups for per-engine centroids
DEFAULT_FEATURE_GROUPS = {
    'shape': ['kurtosis', 'skewness', 'crest_factor'],
    'complexity': ['entropy', 'hurst', 'autocorr'],
    'spectral': ['spectral_entropy', 'spectral_centroid', 'band_ratio_low', 'band_ratio_mid', 'band_ratio_high'],
}

FALLBACK_FEATURES = ['kurtosis', 'skewness', 'crest_factor']


def compute_centroid(
    signal_matrix: np.ndarray,
    feature_names: List[str],
    min_signals: int = 2
) -> Dict[str, Any]:
    """
    Compute centroid (mean position) of signals in feature space.

    This is WHERE the system is - the average of all signals.
    The SHAPE (eigenvalues) is computed separately in state_geometry.

    Args:
        signal_matrix: N_signals × D_features matrix
        feature_names: Names of features (columns)
        min_signals: Minimum signals required

    Returns:
        Dict with centroid and basic statistics
    """
    N, D = signal_matrix.shape

    if N < min_signals:
        return {
            'centroid': np.full(D, np.nan),
            'n_signals': 0,
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'std_distance': np.nan,
        }

    # Remove NaN/Inf rows
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    if valid_mask.sum() < min_signals:
        return {
            'centroid': np.full(D, np.nan),
            'n_signals': 0,
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'std_distance': np.nan,
        }

    signal_matrix = signal_matrix[valid_mask]
    N = len(signal_matrix)

    # Centroid = mean position
    centroid = np.mean(signal_matrix, axis=0)

    # Distance from each signal to centroid
    centered = signal_matrix - centroid
    distances = np.linalg.norm(centered, axis=1)

    return {
        'centroid': centroid,
        'n_signals': N,
        'mean_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'std_distance': float(np.std(distances)),
    }


def compute_state_vector(
    signal_vector_path: str,
    typology_path: str,
    output_path: str = "state_vector.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    compute_per_engine: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state vector (centroids) from signal vector.

    The state vector captures WHERE the system is in feature space.
    The SHAPE (eigenvalues, effective_dim) is computed in state_geometry.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        typology_path: Path to typology.parquet
        output_path: Output path
        feature_groups: Dict mapping engine names to feature lists
        compute_per_engine: Compute per-engine centroids
        verbose: Print progress

    Returns:
        State vector DataFrame with centroids per I
    """
    if verbose:
        print("=" * 70)
        print("STATE VECTOR ENGINE")
        print("Computing centroids (position in feature space)")
        print("Eigenvalues computed separately in state_geometry")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    typology = pl.read_parquet(typology_path)

    # Get active signals (non-constant)
    # Check which column identifies constant signals
    if 'is_constant' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('is_constant') == False)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    elif 'signal_std' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('signal_std') > 1e-10)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    else:
        # No filter available, include all
        active_signals = typology['signal_id'].unique().to_list()

    # Filter to active signals
    signal_col = 'signal_id' if 'signal_id' in signal_vector.columns else 'signal_name'
    signal_vector = signal_vector.filter(pl.col(signal_col).is_in(active_signals))

    # Identify features
    meta_cols = ['unit_id', 'I', 'signal_id', 'signal_name', 'n_samples', 'window_size']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    if verbose:
        print(f"Active signals: {len(active_signals)}")
        print(f"Available features: {len(all_features)}")

    # Determine feature groups
    if feature_groups is None:
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups:
            fallback = [f for f in FALLBACK_FEATURES if f in all_features]
            if len(fallback) >= 2:
                feature_groups['full'] = fallback
            else:
                feature_groups['full'] = all_features[:3] if len(all_features) >= 2 else all_features

    # Composite features (union of all groups)
    composite_features = list(set(f for features in feature_groups.values() for f in features))
    composite_features = [f for f in composite_features if f in all_features]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print(f"Composite features: {len(composite_features)}")
        print()

    # Require I column
    if 'I' not in signal_vector.columns:
        raise ValueError("Missing required column 'I'. Use temporal signal_vector.")

    # Process each I
    results = []
    groups = signal_vector.group_by(['I'], maintain_order=True)
    n_groups = signal_vector.select(['I']).unique().height

    if verbose:
        print(f"Processing {n_groups} time points...")

    for i, (group_key, group) in enumerate(groups):
        I = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Build composite matrix
        available_composite = [f for f in composite_features if f in group.columns]
        if len(available_composite) < 2:
            continue

        composite_matrix = group.select(available_composite).to_numpy()
        state = compute_centroid(composite_matrix, available_composite)

        # Build result row
        row = {
            'unit_id': unit_id,
            'I': I,
            'n_signals': state['n_signals'],
        }

        # Centroid (per feature)
        for j, feat in enumerate(available_composite):
            row[f'centroid_{feat}'] = state['centroid'][j]

        # Dispersion (how spread out are signals around centroid)
        row['mean_distance'] = state['mean_distance']
        row['max_distance'] = state['max_distance']
        row['std_distance'] = state['std_distance']

        # Per-engine centroids
        if compute_per_engine:
            for engine_name, features in feature_groups.items():
                available = [f for f in features if f in group.columns]
                if len(available) >= 2:
                    matrix = group.select(available).to_numpy()
                    engine_state = compute_centroid(matrix, available)
                    for j, feat in enumerate(available):
                        row[f'centroid_{engine_name}_{feat}'] = engine_state['centroid'][j]

        results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")
        print()
        print("Columns: centroid_*, mean_distance, max_distance, std_distance")
        print("NOTE: Eigenvalues computed in state_geometry.py")

    return result


def main():
    import sys

    usage = """
State Vector Engine - Centroids (position in feature space)

Usage:
    python state_vector.py <signal_vector.parquet> <typology.parquet> [output.parquet]

Computes WHERE the system is (centroid = mean of signals).
The SHAPE (eigenvalues, effective_dim) is computed in state_geometry.py.

Pipeline:
    signal_vector → state_vector (centroid) → state_geometry (eigenvalues)
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    signal_path = sys.argv[1]
    typology_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "state_vector.parquet"

    compute_state_vector(signal_path, typology_path, output_path)


if __name__ == "__main__":
    main()

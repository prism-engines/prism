"""
02: State Vector Entry Point
============================

Pure orchestration - calls centroid engine from engines/state/centroid.py.
Computes WHERE the system is (centroid = mean position in feature space).

Four views per system window:
  1. Centroid  — mean feature values across signals (existing)
  2. Fourier   — spectral analysis of per-signal feature trajectories
  3. Hilbert   — envelope (amplitude modulation) of feature trajectories
  4. Laplacian — graph coupling structure across signals

Stages: signal_vector.parquet → state_vector.parquet

The SHAPE (eigenvalues, effective_dim) is computed in 03_state_geometry.py.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the actual computation from engine
from manifold.core.state.centroid import compute as compute_centroid_engine
from manifold.io.writer import write_output

# Primitives for feature trajectory analysis
from manifold.primitives.individual.spectral import spectral_profile
from manifold.primitives.individual.hilbert import envelope
from manifold.primitives.matrix.graph import laplacian_matrix, laplacian_eigenvalues


# Feature groups for per-engine centroids
try:
    from manifold.core.geometry.config import DEFAULT_FEATURE_GROUPS, FALLBACK_FEATURES
except ImportError:
    DEFAULT_FEATURE_GROUPS = {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    }
    FALLBACK_FEATURES = ['kurtosis', 'skewness', 'crest_factor']


def compute_centroid(
    signal_matrix: np.ndarray,
    feature_names: List[str],
    min_signals: int = 1
) -> Dict[str, Any]:
    """
    Wrapper - delegates entirely to engine.

    Args:
        signal_matrix: N_signals × D_features matrix
        feature_names: Names of features (columns)
        min_signals: Minimum signals required

    Returns:
        Dict with centroid and distance statistics
    """
    return compute_centroid_engine(signal_matrix, min_signals=min_signals)


# ═══════════════════════════════════════════════════════════════
# Feature trajectory helpers (fourier, hilbert, laplacian views)
# ═══════════════════════════════════════════════════════════════

def _extract_trajectories(
    window_rows: pl.DataFrame,
    feature_cols: List[str],
    signal_col: str = 'signal_id',
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract per-signal feature trajectories from signal_vector rows
    within a system window.

    Groups by signal_id, sorts by signal_0_center, extracts each feature
    as a 1D array.

    Returns:
        {signal_id: {feature_name: trajectory_array}}
    """
    trajectories = {}

    for (sig_id,), sig_df in window_rows.group_by([signal_col]):
        sig_sorted = sig_df.sort('signal_0_center')
        traj = {}
        for feat in feature_cols:
            if feat in sig_sorted.columns:
                arr = sig_sorted[feat].to_numpy().astype(np.float64)
                # Drop NaNs from edges
                valid = np.isfinite(arr)
                if valid.sum() > 0:
                    traj[feat] = arr[valid]
        if traj:
            trajectories[sig_id] = traj

    return trajectories


def _compute_fourier_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_length: int = 8,
) -> Dict[str, float]:
    """
    Fourier view: run spectral_profile on each signal's feature trajectory.
    Aggregate across signals via nanmedian.

    Output keys: fourier_{feature}_dominant_freq, fourier_{feature}_spectral_flatness
    """
    result = {}

    for feat in feature_cols:
        dom_freqs = []
        flatnesses = []

        for sig_trajs in trajectories.values():
            if feat not in sig_trajs:
                continue
            arr = sig_trajs[feat]
            if len(arr) < min_length:
                continue

            sp = spectral_profile(arr)
            dom_freqs.append(sp.get('dominant_frequency', np.nan))
            flatnesses.append(sp.get('spectral_flatness', np.nan))

        result[f'fourier_{feat}_dominant_freq'] = float(np.nanmedian(dom_freqs)) if dom_freqs else np.nan
        result[f'fourier_{feat}_spectral_flatness'] = float(np.nanmedian(flatnesses)) if flatnesses else np.nan

    return result


def _compute_hilbert_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_length: int = 4,
) -> Dict[str, float]:
    """
    Hilbert view: run envelope on each signal's feature trajectory.
    Compute mean, trend, cv. Aggregate across signals via nanmedian.

    Output keys: envelope_{feature}_mean, envelope_{feature}_trend, envelope_{feature}_cv
    """
    result = {}

    for feat in feature_cols:
        means = []
        trends = []
        cvs = []

        for sig_trajs in trajectories.values():
            if feat not in sig_trajs:
                continue
            arr = sig_trajs[feat]
            if len(arr) < min_length:
                continue

            env = envelope(arr)
            m = float(np.mean(env))
            means.append(m)

            if len(env) >= 2:
                trend = np.polyfit(np.arange(len(env)), env, 1)[0]
                trends.append(float(trend))
            else:
                trends.append(np.nan)

            if m > 0:
                cvs.append(float(np.std(env) / m))
            else:
                cvs.append(np.nan)

        result[f'envelope_{feat}_mean'] = float(np.nanmedian(means)) if means else np.nan
        result[f'envelope_{feat}_trend'] = float(np.nanmedian(trends)) if trends else np.nan
        result[f'envelope_{feat}_cv'] = float(np.nanmedian(cvs)) if cvs else np.nan

    return result


def _compute_laplacian_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_signals: int = 2,
) -> Dict[str, float]:
    """
    Laplacian view: build correlation-based adjacency from concatenated
    feature trajectories, compute graph Laplacian spectrum.

    Output keys: laplacian_algebraic_connectivity, laplacian_spectral_gap,
                 laplacian_n_components, laplacian_max_eigenvalue
    """
    result = {
        'laplacian_algebraic_connectivity': np.nan,
        'laplacian_spectral_gap': np.nan,
        'laplacian_n_components': np.nan,
        'laplacian_max_eigenvalue': np.nan,
    }

    if len(trajectories) < min_signals:
        return result

    # For each signal, concatenate all feature trajectories into one vector
    vectors = []
    for sig_id in sorted(trajectories.keys()):
        parts = []
        for feat in feature_cols:
            if feat in trajectories[sig_id]:
                arr = trajectories[sig_id][feat]
                arr = np.where(np.isfinite(arr), arr, 0.0)
                parts.append(arr)
        if parts:
            vectors.append(np.concatenate(parts))

    if len(vectors) < min_signals:
        return result

    # Pad to same length (signals may have different trajectory lengths)
    max_len = max(len(v) for v in vectors)
    padded = np.zeros((len(vectors), max_len))
    for i, v in enumerate(vectors):
        padded[i, :len(v)] = v

    try:
        # Build adjacency: |corrcoef|
        corr = np.corrcoef(padded)
        if corr.ndim < 2:
            return result
        adj = np.abs(corr)
        np.fill_diagonal(adj, 0.0)
        adj = np.where(np.isfinite(adj), adj, 0.0)

        L = laplacian_matrix(adj, normalized=True)
        eigs = laplacian_eigenvalues(L)

        if len(eigs) >= 2 and np.any(np.isfinite(eigs)):
            result['laplacian_algebraic_connectivity'] = float(eigs[1])
            if eigs[-1] > 0:
                result['laplacian_spectral_gap'] = float(eigs[1] / eigs[-1])
            result['laplacian_n_components'] = float(np.sum(eigs < 1e-10))
            result['laplacian_max_eigenvalue'] = float(eigs[-1])
    except Exception:
        pass

    return result


def compute_state_vector(
    signal_vector_path: str,
    typology_path: str,
    data_path: str = ".",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    compute_per_engine: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state vector (centroids + trajectory views) from signal vector.

    The state vector captures WHERE the system is in feature space (centroid)
    plus HOW features are evolving (fourier, hilbert, laplacian views).
    The SHAPE (eigenvalues, effective_dim) is computed in 03_state_geometry.py.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        typology_path: Path to typology.parquet
        data_path: Root data directory
        feature_groups: Dict mapping engine names to feature lists
        compute_per_engine: Compute per-engine centroids
        verbose: Print progress

    Returns:
        State vector DataFrame with centroids and trajectory views per window
    """
    if verbose:
        print("=" * 70)
        print("02: STATE VECTOR - Centroids + trajectory views")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    typology = pl.read_parquet(typology_path)

    # Get active signals (non-constant)
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
        active_signals = typology['signal_id'].unique().to_list()

    # Filter to active signals
    signal_col = 'signal_id' if 'signal_id' in signal_vector.columns else 'signal_name'
    signal_vector = signal_vector.filter(pl.col(signal_col).is_in(active_signals))

    # Identify features
    meta_cols = ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id', 'signal_name', 'n_samples', 'window_size', 'cohort']
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

    # Require signal_0_end column
    if 'signal_0_end' not in signal_vector.columns:
        raise ValueError("Missing required column 'signal_0_end'. Use temporal signal_vector.")

    # Determine grouping columns
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'signal_0_end'] if has_cohort else ['signal_0_end']

    # Process each group
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    if verbose:
        print(f"Processing {n_groups} groups...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, s0_end = group_key if isinstance(group_key, tuple) else (group_key, None)
        else:
            cohort = None
            s0_end = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Build composite matrix
        available_composite = [f for f in composite_features if f in group.columns]
        if len(available_composite) < 2:
            continue

        composite_matrix = group.select(available_composite).to_numpy()

        # Drop rows where ALL features are NaN (signals with no applicable engines)
        valid_rows = np.isfinite(composite_matrix).any(axis=1)
        composite_matrix = composite_matrix[valid_rows]

        # Need at least 1 signal with data
        if composite_matrix.shape[0] < 1:
            continue

        state = compute_centroid(composite_matrix, available_composite, min_signals=1)

        # Skip if centroid engine returned zero valid signals
        if state['n_signals'] < 1:
            continue

        # Pass through signal_0 columns from the group
        s0_start = group['signal_0_start'].to_list()[0] if 'signal_0_start' in group.columns else None
        s0_center = group['signal_0_center'].to_list()[0] if 'signal_0_center' in group.columns else None

        # Build result row
        row = {
            'signal_0_end': s0_end,
            'signal_0_start': s0_start,
            'signal_0_center': s0_center,
            'n_signals': state['n_signals'],
        }
        if cohort:
            row['cohort'] = cohort
        if unit_id:
            row['unit_id'] = unit_id

        # Dispersion metrics
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
                        row[f'state_{engine_name}_{feat}'] = engine_state['centroid'][j]

        # ── Trajectory views (per-window) ──────────────────────────
        # Gather ALL signal_vector rows whose signal_0_center falls
        # within this system window [s0_start, s0_end].
        # This gives ~N rows per signal (N = window / stride).
        if s0_start is not None and s0_end is not None:
            if has_cohort and cohort is not None:
                window_rows = signal_vector.filter(
                    (pl.col('cohort') == cohort) &
                    (pl.col('signal_0_center') >= s0_start) &
                    (pl.col('signal_0_center') <= s0_end)
                )
            else:
                window_rows = signal_vector.filter(
                    (pl.col('signal_0_center') >= s0_start) &
                    (pl.col('signal_0_center') <= s0_end)
                )

            trajectories = _extract_trajectories(window_rows, composite_features, signal_col)
            if trajectories:
                row.update(_compute_fourier_view(trajectories, composite_features))
                row.update(_compute_hilbert_view(trajectories, composite_features))
                row.update(_compute_laplacian_view(trajectories, composite_features))

        results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    write_output(result, data_path, 'state_vector', verbose=verbose)

    return result


# Alias for run_pipeline.py compatibility
def run(
    signal_vector_path: str,
    data_path: str = ".",
    typology_path: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run state vector computation (wrapper for compute_state_vector)."""
    return compute_state_vector(
        signal_vector_path,
        typology_path,
        data_path,
        verbose=verbose,
    )


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python 02_state_vector.py <signal_vector.parquet> <typology.parquet> [output.parquet]")
        sys.exit(1)

    signal_path = sys.argv[1]
    typology_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "state_vector.parquet"

    compute_state_vector(signal_path, typology_path, output_path)


if __name__ == "__main__":
    main()

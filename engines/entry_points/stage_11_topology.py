"""
Stage 11: Topology Entry Point
==============================

Pure orchestration - calls topology computation engines.

Inputs:
    - signal_vector.parquet

Output:
    - topology.parquet

Computes topological features of signal manifold:
    - Persistent homology (Betti numbers)
    - Topological complexity
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def compute_basic_topology(signal_matrix: np.ndarray, threshold: float = None) -> Dict[str, Any]:
    """
    Compute basic topological features.

    For full persistent homology, use external libraries (ripser, gudhi).
    This provides a simplified approximation.

    Args:
        signal_matrix: N x D matrix of signals
        threshold: Correlation threshold for adjacency. If None, uses
                   90th percentile of abs(correlation) from the data.

    Returns:
        Dict with topological metrics
    """
    if signal_matrix.shape[0] < 3:
        return {'topology_computed': False}

    # Basic connectivity analysis
    # Correlation-based adjacency (fill NaN with 0 before corrcoef)
    signal_matrix_filled = np.where(np.isfinite(signal_matrix), signal_matrix, 0.0)
    corr = np.corrcoef(signal_matrix_filled.T)
    np.fill_diagonal(corr, 0)

    # Adaptive threshold: use 90th percentile of abs(correlation) if not specified
    if threshold is None:
        upper_tri = np.abs(corr[np.triu_indices_from(corr, k=1)])
        finite_vals = upper_tri[np.isfinite(upper_tri)]
        if len(finite_vals) > 0:
            threshold = float(np.percentile(finite_vals, 90))
        else:
            threshold = 0.5

    # Replace NaN correlations with 0 for adjacency computation
    corr_clean = np.where(np.isfinite(corr), corr, 0.0)
    adjacency = np.abs(corr_clean) > threshold

    # Connected components (simplified)
    n_signals = signal_matrix.shape[1]
    n_edges = np.sum(adjacency) // 2
    density = n_edges / (n_signals * (n_signals - 1) / 2) if n_signals > 1 else 0

    # Degree distribution
    degrees = np.sum(adjacency, axis=0)
    mean_degree = np.mean(degrees)
    max_degree = np.max(degrees)

    return {
        'topology_computed': True,
        'n_signals': n_signals,
        'n_edges': int(n_edges),
        'density': float(density),
        'mean_degree': float(mean_degree),
        'max_degree': int(max_degree),
        'threshold': threshold,
    }


def run(
    signal_vector_path: str,
    output_path: str = "topology.parquet",
    signal_pairwise_path: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run topology computation.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        output_path: Output path for topology.parquet
        signal_pairwise_path: Optional path to signal_pairwise.parquet for adaptive threshold
        verbose: Print progress

    Returns:
        Topology DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 11: TOPOLOGY")
        print("Topological features of signal manifold")
        print("=" * 70)

    # Load signal vector
    sv = pl.read_parquet(signal_vector_path)

    # Adaptive threshold: None means compute_basic_topology will use 90th
    # percentile of its own internal abs(correlation) matrix per group.
    # This adapts to each group's correlation structure automatically.
    adaptive_threshold = None

    if verbose:
        print(f"Loaded signal vector: {len(sv)} rows")

    # Determine grouping
    has_I = 'I' in sv.columns
    group_cols = ['I'] if has_I else []

    if 'cohort' in sv.columns:
        group_cols = ['cohort'] + group_cols

    # Get feature columns
    meta_cols = ['unit_id', 'I', 'signal_id', 'cohort']
    feature_cols = [c for c in sv.columns if c not in meta_cols
                    and sv[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not feature_cols:
        if verbose:
            print("No numeric feature columns found")
        return pl.DataFrame()

    results = []

    if group_cols:
        groups = sv.group_by(group_cols, maintain_order=True)
        for group_key, group in groups:
            matrix = group.select(feature_cols).to_numpy()

            result = compute_basic_topology(matrix.T, threshold=adaptive_threshold)  # Transpose for signals x features

            if isinstance(group_key, tuple):
                for i, col in enumerate(group_cols):
                    result[col] = group_key[i]
            else:
                result[group_cols[0]] = group_key

            results.append(result)
    else:
        matrix = sv.select(feature_cols).to_numpy()
        result = compute_basic_topology(matrix.T, threshold=adaptive_threshold)
        results.append(result)

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {df.shape}")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 11: Topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes topological features of signal manifold:
  - Network density
  - Degree distribution

Example:
  python -m engines.entry_points.stage_11_topology \\
      signal_vector.parquet -o topology.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='topology.parquet',
                        help='Output path (default: topology.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

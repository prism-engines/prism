"""
Stage 09: Dynamics Entry Point
==============================

TODO: This stage is currently redundant with stage_08_ftle.
It should be refactored to compute attractor metrics, RQA,
correlation dimension, etc. - "everything else" in dynamics.

For now, it computes FTLE-based stability classification.

Inputs:
    - observations.parquet

Output:
    - dynamics.parquet
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional

from engines.manifold.dynamics.ftle import compute as compute_ftle
from engines.manifold.dynamics.formal_definitions import classify_stability


def run(
    observations_path: str,
    output_path: str = "dynamics.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    min_samples: int = 200,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run dynamics computation for all signals, per-cohort.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for dynamics.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        min_samples: Minimum samples for computation
        verbose: Print progress

    Returns:
        Dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 09: DYNAMICS")
        print("Per-signal per-cohort stability classification")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    signals = obs[signal_column].unique().to_list()

    # Determine cohorts
    has_cohort = 'cohort' in obs.columns
    if has_cohort:
        cohorts = obs['cohort'].unique().sort().to_list()
    else:
        cohorts = [None]

    if verbose:
        print(f"Loaded: {len(obs):,} observations, {len(signals)} signals, {len(cohorts)} cohorts")
        print(f"Work: {len(signals)} signals × {len(cohorts)} cohorts = {len(signals) * len(cohorts)} items")

    results = []
    total_items = 0

    for ci, cohort in enumerate(cohorts):
        # Filter observations to this cohort
        if cohort is not None:
            cohort_obs = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_obs = obs

        for signal in signals:
            signal_data = cohort_obs.filter(pl.col(signal_column) == signal).sort(index_column)
            values = signal_data[value_column].to_numpy()
            values = values[~np.isnan(values)]

            if len(values) < min_samples:
                continue

            # Compute FTLE
            result = compute_ftle(values, min_samples=min_samples)

            # Add stability classification based on FTLE
            ftle_val = result.get('ftle')
            if ftle_val is not None:
                stability = classify_stability(ftle_val)
                result['stability'] = stability.value
                result['ftle'] = ftle_val
            else:
                result['stability'] = 'unknown'
                result['ftle'] = np.nan

            result['signal_id'] = signal
            result['n_samples'] = len(values)
            if cohort is not None:
                result['cohort'] = cohort
            results.append(result)
            total_items += 1

        if verbose and (ci + 1) % 10 == 0:
            print(f"  Processed {ci + 1}/{len(cohorts)} cohorts ({total_items} items so far)...")

    # Build DataFrame
    df = pl.DataFrame(results, infer_schema_length=len(results)) if results else pl.DataFrame()
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {df.shape}")

        # Summary
        stable_count = len([r for r in results
                           if r.get('stability') in ['stable', 'asymptotically_stable']])
        unstable_count = len([r for r in results
                             if r.get('stability') in ['weakly_unstable', 'chaotic', 'unstable']])
        print(f"Stable: {stable_count}, Unstable: {unstable_count}")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 09: Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal dynamics metrics:
  - ftle: Finite-Time Lyapunov Exponent
  - stability: stable/marginal/unstable/chaotic
  - embedding_dim, embedding_tau: Phase space parameters

Example:
  python -m engines.entry_points.stage_09_dynamics \\
      observations.parquet -o dynamics.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='dynamics.parquet',
                        help='Output path (default: dynamics.parquet)')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples for computation (default: 200)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

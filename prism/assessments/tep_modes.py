"""
TEP Modes Computation
=====================

Compute behavioral modes for TEP signals using Laplace fingerprints.
Lightweight version that samples dates to avoid OOM.

Usage:
    python -m prism.assessments.tep_modes --domain cheme
"""

import argparse
import polars as pl
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import os
import warnings
import gc

warnings.filterwarnings('ignore')


def compute_tep_modes(domain: str, sample_dates: int = 50):
    """
    Compute modes for TEP signals.

    Uses lazy evaluation and date sampling to avoid OOM.
    """
    from prism.db.parquet_store import get_parquet_path
    from prism.db.polars_io import write_parquet_atomic

    print("=" * 100)
    print("TEP MODES COMPUTATION")
    print("=" * 100)
    print()

    field_path = get_parquet_path('vector', 'signal_field', domain)

    if not field_path.exists():
        print(f"ERROR: Field data not found at {field_path}")
        print("Run geometry first to generate field data")
        return None

    # Get available dates (lazy)
    print("Scanning field data...")
    lf = pl.scan_parquet(field_path)

    # Filter to TEP only
    tep_lf = lf.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    # Get unique dates
    dates = tep_lf.select('window_end').unique().sort('window_end').collect()
    all_dates = dates['window_end'].to_list()
    print(f"  Total dates: {len(all_dates)}")

    # Sample dates evenly
    if len(all_dates) > sample_dates:
        step = len(all_dates) // sample_dates
        sampled_dates = all_dates[::step][:sample_dates]
    else:
        sampled_dates = all_dates

    print(f"  Sampled dates: {len(sampled_dates)}")

    # Fingerprint features for clustering
    fingerprint_cols = [
        'gradient_mean', 'gradient_std', 'gradient_magnitude',
        'laplacian_mean', 'laplacian_std', 'divergence'
    ]

    all_modes = []

    for i, date in enumerate(sampled_dates):
        if i % 10 == 0:
            print(f"  Processing date {i+1}/{len(sampled_dates)}: {date}")

        # Load just this date (lazy filter)
        date_data = (
            pl.scan_parquet(field_path)
            .filter(
                (pl.col('window_end') == date) &
                pl.col('signal_id').str.starts_with('TEP_') &
                ~pl.col('signal_id').str.contains('FAULT')
            )
            .collect()
        )

        if len(date_data) < 10:
            continue

        # Aggregate fingerprint per signal
        agg_exprs = [pl.col(c).mean().alias(c) for c in fingerprint_cols if c in date_data.columns]
        if not agg_exprs:
            continue

        fingerprints = date_data.group_by('signal_id').agg(agg_exprs)

        if len(fingerprints) < 5:
            continue

        # Build feature matrix
        available_cols = [c for c in fingerprint_cols if c in fingerprints.columns]
        X = fingerprints.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0)

        if X.shape[0] < 5 or X.shape[1] < 2:
            continue

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # GMM clustering
        n_modes = min(5, X.shape[0] // 3)
        if n_modes < 2:
            n_modes = 2

        try:
            gmm = GaussianMixture(n_components=n_modes, random_state=42, max_iter=100)
            gmm.fit(X_scaled)

            # Get assignments
            mode_ids = gmm.predict(X_scaled)
            probs = gmm.predict_proba(X_scaled)

            # Compute affinity and entropy
            affinities = np.max(probs, axis=1)
            entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)

            # Build result
            signals = fingerprints['signal_id'].to_list()
            for j, ind in enumerate(signals):
                all_modes.append({
                    'signal_id': ind,
                    'obs_date': date,
                    'mode_id': int(mode_ids[j]),
                    'mode_affinity': float(affinities[j]),
                    'mode_entropy': float(entropies[j]),
                })

        except Exception as e:
            print(f"    GMM failed for {date}: {e}")
            continue

        # Memory cleanup
        del date_data, fingerprints, X, X_scaled
        gc.collect()

    if not all_modes:
        print("ERROR: No modes computed")
        return None

    # Create DataFrame
    modes_df = pl.DataFrame(all_modes)
    print(f"\nModes computed: {len(modes_df):,} rows")

    # Save
    output_path = get_parquet_path('vector', 'signal_modes', domain)
    write_parquet_atomic(modes_df, output_path)
    print(f"Saved to: {output_path}")

    # Summary stats
    print()
    print("=" * 100)
    print("MODE SUMMARY")
    print("=" * 100)
    print(f"  Unique modes: {modes_df['mode_id'].n_unique()}")
    print(f"  Mean affinity: {modes_df['mode_affinity'].mean():.3f}")
    print(f"  Min affinity: {modes_df['mode_affinity'].min():.3f}")
    print(f"  Mean entropy: {modes_df['mode_entropy'].mean():.3f}")
    print(f"  Max entropy: {modes_df['mode_entropy'].max():.3f}")

    # Mode distribution
    print("\nMode distribution:")
    mode_counts = modes_df.group_by('mode_id').len().sort('mode_id')
    for row in mode_counts.iter_rows(named=True):
        print(f"  Mode {row['mode_id']}: {row['len']:,} assignments")

    print()
    print("=" * 100)

    return modes_df


def main():
    parser = argparse.ArgumentParser(description='TEP Modes Computation')
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--sample', type=int, default=50, help='Number of dates to sample')
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain")
    os.environ["PRISM_DOMAIN"] = domain

    compute_tep_modes(domain, sample_dates=args.sample)


if __name__ == '__main__':
    main()

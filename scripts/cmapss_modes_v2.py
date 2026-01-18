#!/usr/bin/env python3
"""
C-MAPSS: Discover behavioral modes using vector metrics (not field).

This version works for ALL engines including short-lifecycle ones (u039, u091).
"""

import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DOMAIN = "cmapss_fd001"
DATA_DIR = Path("/Users/jasonrudder/prism-mac/data") / DOMAIN


def extract_vector_fingerprint(
    vectors: pl.DataFrame,
    signal_id: str,
) -> Optional[Dict]:
    """Extract fingerprint from vector metrics (not field)."""
    subset = vectors.filter(pl.col('signal_id') == signal_id)

    if len(subset) < 5:
        return None

    # Key metrics for fingerprint
    metrics = ['hurst_exponent', 'sample_entropy', 'realized_vol', 'signal_to_noise']

    fingerprint = {'signal_id': signal_id}

    for metric in metrics:
        metric_data = subset.filter(pl.col('metric_name') == metric)['metric_value'].drop_nulls()

        if len(metric_data) > 0:
            vals = metric_data.to_numpy()
            fingerprint[f'{metric}_mean'] = float(np.mean(vals))
            fingerprint[f'{metric}_std'] = float(np.std(vals))
            fingerprint[f'{metric}_trend'] = float(np.polyfit(range(len(vals)), vals, 1)[0]) if len(vals) > 1 else 0.0
        else:
            fingerprint[f'{metric}_mean'] = 0.0
            fingerprint[f'{metric}_std'] = 0.0
            fingerprint[f'{metric}_trend'] = 0.0

    return fingerprint


def find_optimal_modes(X: np.ndarray, max_modes: int = 10) -> int:
    """Find optimal modes using BIC."""
    if len(X) < 3:
        return 2

    best_bic = np.inf
    best_n = 2

    for n in range(2, min(max_modes + 1, len(X))):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except:
            continue

    return best_n


def discover_cohort_modes(
    vectors: pl.DataFrame,
    cohort_id: str,
    signals: List[str],
    max_modes: int = 10
) -> Optional[pd.DataFrame]:
    """Discover modes for a cohort from vector fingerprints."""

    # Extract fingerprints
    fingerprints = []
    for ind in signals:
        fp = extract_vector_fingerprint(vectors, ind)
        if fp:
            fp['cohort_id'] = cohort_id
            fingerprints.append(fp)

    if len(fingerprints) < 3:
        logger.warning(f"Cohort {cohort_id}: insufficient fingerprints ({len(fingerprints)})")
        return None

    fp_df = pd.DataFrame(fingerprints)

    # Feature columns
    feature_cols = [c for c in fp_df.columns if c not in ['signal_id', 'cohort_id']]

    X = fp_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal modes
    best_n = find_optimal_modes(X_scaled, max_modes)

    # Fit GMM
    gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)

    probs = gmm.predict_proba(X_scaled)

    # Compute scores
    mode_id = np.argmax(probs, axis=1)
    mode_affinity = np.max(probs, axis=1)
    mode_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    result = fp_df[['signal_id', 'cohort_id']].copy()
    result['domain_id'] = DOMAIN
    result['mode_id'] = mode_id
    result['mode_affinity'] = mode_affinity
    result['mode_entropy'] = mode_entropy
    result['n_modes'] = best_n

    # Include fingerprint
    for col in feature_cols:
        result[f'fingerprint_{col}'] = fp_df[col].values

    logger.info(f"Cohort {cohort_id}: {best_n} modes from {len(fingerprints)} signals")

    return result


def main():
    print("=" * 70)
    print("C-MAPSS FD001 - MODE DISCOVERY (v2 - from vectors)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    vectors = pl.read_parquet(DATA_DIR / 'vector' / 'signal.parquet')
    cohort_members = pl.read_parquet(DATA_DIR / 'config' / 'cohort_members.parquet')

    vectors = vectors.with_columns([
        pl.col('signal_id').str.extract(r'u(\d+)_').cast(pl.Int64).alias('unit'),
    ])

    cohorts = cohort_members['cohort_id'].unique().sort().to_list()
    print(f"  Cohorts: {len(cohorts)}")
    print(f"  Vector rows: {len(vectors):,}")

    # Process all cohorts
    print("\n[2] Discovering modes...")
    all_modes = []

    for cohort_id in cohorts:
        signals = cohort_members.filter(
            pl.col('cohort_id') == cohort_id
        )['signal_id'].to_list()

        modes_df = discover_cohort_modes(vectors, cohort_id, signals)
        if modes_df is not None:
            all_modes.append(modes_df)

    # Combine and save
    print("\n[3] Saving results...")
    result = pd.concat(all_modes, ignore_index=True)

    output_path = DATA_DIR / 'geometry' / 'cohort_modes.parquet'
    result.to_parquet(output_path, index=False)

    print(f"  Total assignments: {len(result):,}")
    print(f"  Cohorts processed: {result['cohort_id'].nunique()}")
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("MODE DISCOVERY RESULTS")
    print("=" * 70)

    modes_per_cohort = result.groupby('cohort_id')['mode_id'].nunique()
    print(f"\nModes per cohort: mean={modes_per_cohort.mean():.1f}, min={modes_per_cohort.min()}, max={modes_per_cohort.max()}")
    print(f"Mode affinity: mean={result['mode_affinity'].mean():.3f}")
    print(f"Mode entropy: mean={result['mode_entropy'].mean():.3f}")

    # Verify all 100 engines
    missing = set(cohorts) - set(result['cohort_id'].unique())
    if missing:
        print(f"\nMissing cohorts: {missing}")
    else:
        print(f"\n✓ All {len(cohorts)} engines processed successfully")


if __name__ == '__main__':
    main()

"""
TEP Break Detection Agent
=========================

Focused assessment using discontinuity detection engines:
- Break Detector: Structural breaks via z-score, CUSUM
- Dirac: Impulse detection (sudden shocks)
- Heaviside: Step function detection (level shifts)

TEP faults are SUDDEN EVENTS - these engines are designed for exactly this.

Usage:
    python -m prism.assessments.tep_break_detection --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from datetime import timedelta, date
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DISCONTINUITY METRICS (from break_detector, dirac, heaviside)
# =============================================================================

BREAK_METRICS = [
    # Break detector
    'break_n', 'break_rate', 'break_is_accelerating',
    'break_magnitude_mean', 'break_magnitude_max',
    'break_cusum_max', 'break_volatility_ratio',
    # Dirac (impulses)
    'dirac_n_impulses', 'dirac_mean_magnitude', 'dirac_max_magnitude',
    'dirac_total_energy', 'dirac_mean_decay_rate', 'dirac_up_ratio',
    # Heaviside (steps)
    'heaviside_n_steps', 'heaviside_mean_magnitude', 'heaviside_max_magnitude',
    'heaviside_up_ratio', 'heaviside_persistence',
]


def load_data(domain: str):
    """Load TEP data."""
    from prism.db.parquet_store import get_parquet_path

    vec_path = get_parquet_path('vector', 'signal', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    vec_df = pl.read_parquet(vec_path)
    obs_df = pl.read_parquet(obs_path)

    return vec_df, obs_df


def get_break_features(vec_df: pl.DataFrame) -> pl.DataFrame:
    """Extract break detection features per date."""

    # Filter to TEP process signals
    process_df = vec_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    # Filter to break detection metrics
    break_df = process_df.filter(pl.col('metric_name').is_in(BREAK_METRICS))

    if len(break_df) == 0:
        print("  WARNING: No break detection metrics found in vector data")
        # Try partial match
        all_metrics = process_df['metric_name'].unique().to_list()
        matching = [m for m in all_metrics if any(x in m for x in ['break', 'dirac', 'heaviside'])]
        print(f"  Available matching metrics: {matching[:20]}")
        if matching:
            break_df = process_df.filter(pl.col('metric_name').is_in(matching))

    # Aggregate by date
    agg_df = break_df.group_by(['obs_date', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('mean'),
        pl.col('metric_value').std().alias('std'),
        pl.col('metric_value').max().alias('max'),
        pl.col('metric_value').sum().alias('sum'),
    ])

    # Pivot to wide format
    features = None
    for stat in ['mean', 'std', 'max', 'sum']:
        pivot = agg_df.select([
            'obs_date',
            (pl.col('metric_name') + f'_{stat}').alias('feature'),
            pl.col(stat).alias('value')
        ]).pivot(on='feature', index='obs_date', values='value')

        if features is None:
            features = pivot
        else:
            features = features.join(pivot, on='obs_date', how='outer', coalesce=True)

    return features.sort('obs_date')


def get_fault_onsets(obs_df: pl.DataFrame) -> pl.DataFrame:
    """Find fault onset timestamps (0 -> fault transition)."""
    fault_df = obs_df.filter(pl.col('signal_id') == 'TEP_FAULT').select([
        'obs_date',
        pl.col('value').alias('fault_code')
    ]).group_by('obs_date').agg(
        pl.col('fault_code').mode().first()
    ).sort('obs_date')

    # Find transitions from 0 to non-zero
    onsets = fault_df.with_columns([
        pl.col('fault_code').shift(1).alias('prev_fault'),
    ]).filter(
        (pl.col('prev_fault') == 0) & (pl.col('fault_code') > 0)
    ).select([
        'obs_date',
        'fault_code',
    ]).rename({'obs_date': 'onset_date', 'fault_code': 'fault_id'})

    return onsets


def compute_break_signal(features: pl.DataFrame) -> pl.DataFrame:
    """
    Compute composite break signal from individual metrics.

    Combines break_n, dirac_n_impulses, heaviside_n_steps into a
    unified "discontinuity score" that spikes at regime changes.
    """

    # Get available columns
    cols = features.columns

    # Find break count columns
    break_cols = [c for c in cols if 'break_n' in c.lower() or 'n_impulses' in c.lower() or 'n_steps' in c.lower()]
    mag_cols = [c for c in cols if 'magnitude' in c.lower() or 'energy' in c.lower()]

    if not break_cols:
        print(f"  WARNING: No break count columns found. Available: {cols[:10]}")
        return features

    # Compute composite signal
    signal_expr = []
    for col in break_cols[:5]:  # Top 5 count columns
        signal_expr.append(pl.col(col).fill_null(0))

    # Composite = sum of break counts (normalized later)
    features = features.with_columns([
        sum(signal_expr).alias('break_signal_raw')
    ])

    # Z-score normalization over rolling window
    features = features.with_columns([
        pl.col('break_signal_raw').rolling_mean(window_size=7).alias('break_signal_ma'),
        pl.col('break_signal_raw').rolling_std(window_size=7).alias('break_signal_std'),
    ])

    features = features.with_columns([
        ((pl.col('break_signal_raw') - pl.col('break_signal_ma')) /
         (pl.col('break_signal_std') + 1e-6)).alias('break_zscore')
    ])

    return features


def evaluate_detection(
    features: pl.DataFrame,
    onsets: pl.DataFrame,
    signal_col: str = 'break_zscore',
    threshold: float = 2.0,
    lead_days: int = 3
) -> Dict:
    """
    Evaluate break detection against known fault onsets.

    Args:
        features: DataFrame with break signal per date
        onsets: DataFrame with fault onset dates
        signal_col: Column containing break signal
        threshold: Z-score threshold for detection
        lead_days: How many days before onset counts as "early detection"

    Returns:
        Dict with detection metrics
    """

    if signal_col not in features.columns:
        print(f"  Signal column {signal_col} not found")
        return {'detected': 0, 'total': 0, 'rate': 0.0}

    results = []
    detected = 0
    early_detected = 0

    for row in onsets.iter_rows(named=True):
        onset_date = row['onset_date']
        fault_id = row['fault_id']

        # Look at window around onset: lead_days before to onset
        window_start = onset_date - timedelta(days=lead_days)
        window_end = onset_date

        window_data = features.filter(
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )

        if len(window_data) == 0:
            results.append({
                'onset_date': onset_date,
                'fault_id': f'IDV{int(fault_id):02d}',
                'detected': False,
                'max_zscore': None,
                'lead_time': None,
            })
            continue

        # Check if any signal exceeds threshold
        max_zscore = window_data[signal_col].max()
        if max_zscore is None:
            max_zscore = 0

        is_detected = abs(max_zscore) >= threshold

        # Find first detection
        lead_time = None
        if is_detected:
            detected += 1
            detection_rows = window_data.filter(
                pl.col(signal_col).abs() >= threshold
            ).sort('obs_date')
            if len(detection_rows) > 0:
                first_detection = detection_rows['obs_date'][0]
                lead_time = (onset_date - first_detection).days
                if lead_time > 0:
                    early_detected += 1

        results.append({
            'onset_date': onset_date,
            'fault_id': f'IDV{int(fault_id):02d}',
            'detected': is_detected,
            'max_zscore': float(max_zscore) if max_zscore else 0,
            'lead_time': lead_time,
        })

    total = len(onsets)
    detection_rate = detected / total if total > 0 else 0
    early_rate = early_detected / total if total > 0 else 0

    return {
        'detected': detected,
        'early_detected': early_detected,
        'total': total,
        'detection_rate': detection_rate,
        'early_rate': early_rate,
        'results': results,
    }


def analyze_per_fault(results: List[Dict]) -> Dict:
    """Analyze detection rate per fault type."""
    from collections import defaultdict

    per_fault = defaultdict(lambda: {'detected': 0, 'total': 0})

    for r in results:
        fid = r['fault_id']
        per_fault[fid]['total'] += 1
        if r['detected']:
            per_fault[fid]['detected'] += 1

    return {k: {'detected': v['detected'], 'total': v['total'],
                'rate': v['detected'] / v['total'] if v['total'] > 0 else 0}
            for k, v in per_fault.items()}


def run_break_detection(domain: str):
    """Run break detection evaluation on TEP."""

    print("=" * 100)
    print("TEP BREAK DETECTION AGENT")
    print("=" * 100)
    print()
    print("Using discontinuity engines:")
    print("  - Break Detector: Structural breaks via z-score, CUSUM")
    print("  - Dirac: Impulse detection (sudden shocks)")
    print("  - Heaviside: Step function detection (level shifts)")
    print()

    # Load data
    print("Loading data...")
    vec_df, obs_df = load_data(domain)

    # Get break features
    print("Extracting break detection features...")
    features = get_break_features(vec_df)
    print(f"  Features: {len(features.columns) - 1}")
    print(f"  Dates: {features['obs_date'].min()} to {features['obs_date'].max()}")

    # Get fault onsets
    print("\nFinding fault onsets...")
    onsets = get_fault_onsets(obs_df)
    print(f"  Found {len(onsets)} fault onset events")

    # Compute composite break signal
    print("\nComputing composite break signal...")
    features = compute_break_signal(features)

    # Test multiple thresholds
    print()
    print("=" * 100)
    print("DETECTION EVALUATION")
    print("=" * 100)

    best_result = None
    best_rate = 0

    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        result = evaluate_detection(features, onsets, 'break_zscore', threshold, lead_days=3)

        print(f"\n  Threshold z={threshold:.1f}:")
        print(f"    Detection rate: {result['detection_rate']:.1%} ({result['detected']}/{result['total']})")
        print(f"    Early detection: {result['early_rate']:.1%} ({result['early_detected']}/{result['total']})")

        if result['detection_rate'] > best_rate:
            best_rate = result['detection_rate']
            best_result = result

    # Analyze best result per fault
    if best_result:
        print()
        print("=" * 100)
        print("PER-FAULT DETECTION (Best Threshold)")
        print("=" * 100)

        per_fault = analyze_per_fault(best_result['results'])

        for fid in sorted(per_fault.keys()):
            stats = per_fault[fid]
            status = "+" if stats['rate'] >= 0.7 else "~" if stats['rate'] >= 0.3 else "-"
            print(f"  {status} {fid}: {stats['rate']:.0%} ({stats['detected']}/{stats['total']})")

        # Show sample detections
        print()
        print("=" * 100)
        print("SAMPLE DETECTIONS")
        print("=" * 100)

        detected_events = [r for r in best_result['results'] if r['detected']][:10]
        for r in detected_events:
            lead = f"+{r['lead_time']}d early" if r['lead_time'] and r['lead_time'] > 0 else "on onset"
            print(f"  {r['fault_id']} @ {r['onset_date']}: z={r['max_zscore']:.2f} ({lead})")

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Best detection rate: {best_rate:.1%}")
    print(f"Events detected: {best_result['detected']}/{best_result['total']}" if best_result else "N/A")
    print()
    print("=" * 100)

    return best_result


def main():
    parser = argparse.ArgumentParser(description='TEP Break Detection Agent')
    parser.add_argument('--domain', type=str, default=None)
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for TEP break detection")
    os.environ["PRISM_DOMAIN"] = domain

    run_break_detection(domain)


if __name__ == '__main__':
    main()

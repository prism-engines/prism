"""
PRISM Assessment Runner
========================

Domain-agnostic fault/regime detection using WHAT + WHEN + MODE layers.

Usage:
    python -m prism.assessments.run --domain cheme
    python -m prism.assessments.run --domain turbofan
    python -m prism.assessments.run --domain cheme --show-config
"""

import argparse
import polars as pl
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import os
import warnings

from prism.assessments.config import (
    get_domain_config,
    get_windows,
    get_thresholds,
    get_what_features,
    get_when_features,
    get_mode_features,
    get_signal_patterns,
    get_precursor_mode,
    get_precursor_signals,
    print_config,
)

warnings.filterwarnings('ignore')


def load_data(domain: str):
    """Load all data sources for assessment."""
    from prism.db.parquet_store import get_parquet_path

    vec_path = get_parquet_path('vector', 'signal', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)
    modes_path = get_parquet_path('vector', 'signal_modes', domain)

    vec_df = pl.read_parquet(vec_path) if vec_path.exists() else None
    obs_df = pl.read_parquet(obs_path) if obs_path.exists() else None
    modes_df = pl.read_parquet(modes_path) if modes_path.exists() else None

    return vec_df, obs_df, modes_df


def filter_signals(df: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Filter to domain signals, excluding fault labels."""
    patterns = get_signal_patterns(domain)

    if not patterns['prefix']:
        return df

    filtered = df.filter(pl.col('signal_id').str.starts_with(patterns['prefix']))

    if patterns['exclude_pattern']:
        filtered = filtered.filter(~pl.col('signal_id').str.contains(patterns['exclude_pattern']))

    return filtered


def get_fault_events(obs_df: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Get fault onset events from observations."""
    patterns = get_signal_patterns(domain)

    if not patterns['fault_signal']:
        print(f"Warning: No fault_signal configured for {domain}")
        return pl.DataFrame()

    fault_df = obs_df.filter(
        pl.col('signal_id') == patterns['fault_signal']
    ).select([
        'obs_date',
        pl.col('value').alias('fault_code')
    ]).group_by('obs_date').agg(
        pl.col('fault_code').mode().first()
    ).sort('obs_date')

    # Compute previous fault to detect onsets
    fault_df = fault_df.with_columns([
        pl.col('fault_code').shift(1).alias('prev_fault'),
    ])

    # Onset = transition from 0 to non-zero
    onsets = fault_df.filter(
        (pl.col('prev_fault') == 0) & (pl.col('fault_code') > 0)
    )

    return onsets


# =============================================================================
# LAYER 1: WHAT (Classification Features)
# =============================================================================

def get_what_signals(vec_df: pl.DataFrame, date, domain: str) -> Dict[str, float]:
    """Get WHAT layer features for a date."""
    features = get_what_features(domain)

    day_data = filter_signals(vec_df, domain).filter(
        (pl.col('obs_date') == date) &
        pl.col('metric_name').is_in(features)
    )

    if len(day_data) == 0:
        return {}

    agg = day_data.group_by('metric_name').agg([
        pl.col('metric_value').mean().alias('mean'),
        pl.col('metric_value').std().alias('std'),
    ])

    result = {}
    for row in agg.iter_rows(named=True):
        result[f"what_{row['metric_name']}_mean"] = row['mean']
        result[f"what_{row['metric_name']}_std"] = row['std']

    return result


# =============================================================================
# LAYER 2: WHEN (Break Detection Features)
# =============================================================================

def get_when_signals(vec_df: pl.DataFrame, date, domain: str) -> Dict[str, float]:
    """Get WHEN layer features for a date."""
    features = get_when_features(domain)

    day_data = filter_signals(vec_df, domain).filter(
        (pl.col('obs_date') == date) &
        pl.col('metric_name').is_in(features)
    )

    if len(day_data) == 0:
        return {}

    agg = day_data.group_by('metric_name').agg([
        pl.col('metric_value').sum().alias('sum'),
        pl.col('metric_value').max().alias('max'),
    ])

    result = {}
    for row in agg.iter_rows(named=True):
        result[f"when_{row['metric_name']}_sum"] = row['sum']
        result[f"when_{row['metric_name']}_max"] = row['max']

    return result


# =============================================================================
# LAYER 3: MODE (Behavioral Trajectory)
# =============================================================================

def get_mode_signals(modes_df: pl.DataFrame, date, domain: str) -> Dict[str, float]:
    """Get MODE layer features for a date."""
    if modes_df is None:
        return {}

    day_data = filter_signals(modes_df, domain).filter(pl.col('obs_date') == date)

    if len(day_data) == 0:
        return {}

    result = {}

    if 'mode_affinity' in day_data.columns:
        affinities = day_data['mode_affinity'].drop_nulls()
        if len(affinities) > 0:
            result['mode_affinity_mean'] = float(affinities.mean())
            result['mode_affinity_min'] = float(affinities.min())

    if 'mode_entropy' in day_data.columns:
        entropy = day_data['mode_entropy'].drop_nulls()
        if len(entropy) > 0:
            result['mode_entropy_mean'] = float(entropy.mean())
            result['mode_entropy_max'] = float(entropy.max())

    return result


def get_precursor_mode_signal(modes_df: pl.DataFrame, date, domain: str) -> Dict:
    """Detect precursor mode presence for a date."""
    if modes_df is None:
        return {'precursor_count': 0, 'precursor_present': False}

    precursor_mode = get_precursor_mode(domain)
    known_precursors = get_precursor_signals(domain)

    day_data = filter_signals(modes_df, domain).filter(pl.col('obs_date') == date)

    if len(day_data) == 0:
        return {'precursor_count': 0, 'precursor_present': False}

    # Count precursor mode assignments
    precursor_data = day_data.filter(pl.col('mode_id') == precursor_mode)
    precursor_count = len(precursor_data)

    # Get signals in precursor mode
    precursor_signals = precursor_data['signal_id'].to_list() if precursor_count > 0 else []

    # Check if known precursor signals are in the mode
    known_in_mode = [ind for ind in precursor_signals if ind in known_precursors]

    return {
        'precursor_count': precursor_count,
        'precursor_present': precursor_count > 0,
        'precursor_signals': precursor_signals[:5],
        'known_precursors_active': len(known_in_mode) > 0,
    }


def get_precursor_warning(modes_df: pl.DataFrame, onset_date, domain: str) -> Dict:
    """Check for precursor mode in window around onset."""
    if modes_df is None:
        return {'precursor_warning': False}

    windows = get_windows(domain)
    lookback = windows.get('mode1_lookback', 7)

    # Check window before onset
    before_dates = [onset_date - timedelta(days=d) for d in range(1, lookback + 1)]
    precursor_before = 0

    for d in before_dates:
        signal = get_precursor_mode_signal(modes_df, d, domain)
        if signal['precursor_present']:
            precursor_before += signal['precursor_count']

    # Check at onset
    at_signal = get_precursor_mode_signal(modes_df, onset_date, domain)
    precursor_at = at_signal['precursor_count']

    return {
        'precursor_warning': (precursor_before > 0) or (precursor_at > 0),
        'precursor_before': precursor_before,
        'precursor_at': precursor_at,
        'precursor_total': precursor_before + precursor_at,
    }


# =============================================================================
# INTEGRATED ANALYSIS
# =============================================================================

def analyze_onset(vec_df, modes_df, onset_date, fault_id, domain: str) -> Dict:
    """Analyze all layers around a fault onset."""
    windows = get_windows(domain)
    thresholds = get_thresholds(domain)

    pre_window = windows.get('pre_onset', 7)
    post_window = windows.get('post_onset', 5)

    # Collect signals from pre-onset window
    pre_dates = [onset_date - timedelta(days=d) for d in range(1, pre_window + 1)]

    break_sum = 0
    dirac_sum = 0
    heaviside_sum = 0
    affinities = []

    for d in pre_dates:
        when = get_when_signals(vec_df, d, domain)
        break_sum += when.get('when_break_n_sum', 0) or 0
        dirac_sum += when.get('when_dirac_n_impulses_sum', 0) or 0
        heaviside_sum += when.get('when_heaviside_n_steps_sum', 0) or 0

        mode = get_mode_signals(modes_df, d, domain)
        if 'mode_affinity_mean' in mode:
            affinities.append(mode['mode_affinity_mean'])

    # Affinity trend
    affinity_trend = 0
    if len(affinities) >= 2:
        affinity_trend = affinities[-1] - affinities[0]

    # Precursor mode warning
    precursor = get_precursor_warning(modes_df, onset_date, domain)

    # Determine if warning triggered
    has_warning = (
        (break_sum + dirac_sum + heaviside_sum) > 0 or
        affinity_trend < -thresholds.get('affinity_drop', 0.1) or
        precursor.get('precursor_warning', False)
    )

    return {
        'onset_date': onset_date,
        'fault_id': fault_id,
        'break_count': break_sum,
        'dirac_count': dirac_sum,
        'heaviside_count': heaviside_sum,
        'affinity_trend': affinity_trend,
        'precursor_warning': precursor.get('precursor_warning', False),
        'precursor_before': precursor.get('precursor_before', 0),
        'precursor_at': precursor.get('precursor_at', 0),
        'has_warning': has_warning,
    }


def run_assessment(domain: str, max_onsets: int = 30):
    """Run integrated assessment for a domain."""

    config = get_domain_config(domain)
    patterns = get_signal_patterns(domain)
    precursor_mode = get_precursor_mode(domain)

    print("=" * 100)
    print(f"PRISM ASSESSMENT: {config.get('description', domain)}")
    print("=" * 100)
    print()
    print("Three detection layers:")
    print("  WHAT  - Classification features (volatility, entropy)")
    print("  WHEN  - Break detection (structural changes, impulses)")
    print(f"  MODE  - Behavioral trajectory (Mode {precursor_mode} = precursor)")
    print()

    # Load data
    print("Loading data...")
    vec_df, obs_df, modes_df = load_data(domain)

    if vec_df is None:
        print("ERROR: Vector data not found. Run signal_vector first.")
        return

    if obs_df is None:
        print("ERROR: Observations not found.")
        return

    print(f"  Vector data: {len(vec_df):,} rows")
    print(f"  Modes data: {'available' if modes_df is not None else 'NOT FOUND (run laplace first)'}")

    # Get fault events
    print("\nIdentifying fault events...")
    onsets = get_fault_events(obs_df, domain)
    print(f"  Found {len(onsets)} fault onsets")

    if len(onsets) == 0:
        print("No fault onsets found. Check fault_signal config.")
        return

    # Analyze each onset
    print()
    print("=" * 100)
    print("INTEGRATED ANALYSIS PER FAULT ONSET")
    print("=" * 100)

    results = []
    warning_count = 0
    precursor_warnings = 0

    for row in onsets.head(max_onsets).iter_rows(named=True):
        onset_date = row['obs_date']
        fault_id = int(row['fault_code'])

        analysis = analyze_onset(vec_df, modes_df, onset_date, fault_id, domain)
        results.append(analysis)

        if analysis['has_warning']:
            warning_count += 1
        if analysis['precursor_warning']:
            precursor_warnings += 1

        # Print summary
        p_flag = f"M{precursor_mode}!" if analysis['precursor_warning'] else ""
        warn = "EARLY WARNING" if analysis['has_warning'] else ""

        fault_label = f"IDV{fault_id:02d}" if domain == 'cheme' else f"F{fault_id}"

        print(f"  {fault_label} @ {onset_date}: "
              f"breaks={analysis['break_count']}, "
              f"dirac={analysis['dirac_count']}, "
              f"heaviside={analysis['heaviside_count']}, "
              f"aff_delta={analysis['affinity_trend']:+.3f} {p_flag} {warn}")

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Fault onsets analyzed: {len(results)}")
    print(f"Early warnings detected: {warning_count}")
    print(f"Overall warning rate: {warning_count/len(results):.1%}" if results else "N/A")
    print()

    # Signal decomposition
    print("Signal decomposition (pre-onset):")
    total_breaks = sum(r['break_count'] for r in results)
    total_dirac = sum(r['dirac_count'] for r in results)
    total_heaviside = sum(r['heaviside_count'] for r in results)

    print(f"  Break detector:  {total_breaks:,}")
    print(f"  Dirac impulses:  {total_dirac:,}")
    print(f"  Heaviside steps: {total_heaviside:,}")
    print()

    if modes_df is not None:
        print(f"Precursor mode (M{precursor_mode}) warnings: {precursor_warnings}/{len(results)} "
              f"({precursor_warnings/len(results):.1%})")

        total_before = sum(r['precursor_before'] for r in results)
        total_at = sum(r['precursor_at'] for r in results)
        print(f"  Before onset: {total_before} signal-days")
        print(f"  At onset:     {total_at} signal-days")

    print()
    print("=" * 100)

    return results


def main():
    parser = argparse.ArgumentParser(description='PRISM Assessment Runner')
    parser.add_argument('--domain', type=str, default=None, help='Domain to assess')
    parser.add_argument('--show-config', action='store_true', help='Show config and exit')
    parser.add_argument('--max-onsets', type=int, default=30, help='Max onsets to analyze')
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for assessment")
    os.environ["PRISM_DOMAIN"] = domain

    if args.show_config:
        print_config(domain)
        return

    run_assessment(domain, max_onsets=args.max_onsets)


if __name__ == '__main__':
    main()

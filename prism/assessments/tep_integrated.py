"""
TEP Integrated Assessment - WHAT + WHEN + MODE
===============================================

Three layers of regime understanding:

1. WHAT  - Which fault type (classification)
2. WHEN  - Timing of change (break detection)
3. MODE  - Behavioral trajectory (sequence + affinity)

The MODE layer reveals:
- Sequence of behavioral states
- Mode affinity drops = regime transition signal
- Mode entropy spikes = system becoming unstable
- MODE 1 = Precursor/transition state (rare, appears before changes)

Mode 1 Signal:
- Only ~4% of assignments are Mode 1
- Dominated by XMEAS03 (D Feed - upstream reactor input)
- Lower affinity (0.937 vs 0.965) = behavioral uncertainty
- Appears at fault onset: 7.8%, before onset: 20.4%
- Combined: 28.2% early warning rate from Mode 1 alone

Usage:
    python -m prism.assessments.tep_integrated --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import warnings

warnings.filterwarnings('ignore')


def load_all_data(domain: str):
    """Load all TEP data sources."""
    from prism.db.parquet_store import get_parquet_path

    vec_df = pl.read_parquet(get_parquet_path('vector', 'signal', domain))
    obs_df = pl.read_parquet(get_parquet_path('raw', 'observations', domain))

    # Try to load field and modes if available
    field_path = get_parquet_path('vector', 'signal_field', domain)
    modes_path = get_parquet_path('vector', 'signal_modes', domain)

    field_df = pl.read_parquet(field_path) if field_path.exists() else None
    modes_df = pl.read_parquet(modes_path) if modes_path.exists() else None

    return vec_df, obs_df, field_df, modes_df


def get_fault_events(obs_df: pl.DataFrame) -> pl.DataFrame:
    """Get fault onset and recovery events."""
    fault_df = obs_df.filter(pl.col('signal_id') == 'TEP_FAULT').select([
        'obs_date', pl.col('value').alias('fault_code')
    ]).group_by('obs_date').agg(
        pl.col('fault_code').mode().first()
    ).sort('obs_date')

    # Add previous and next fault
    fault_df = fault_df.with_columns([
        pl.col('fault_code').shift(1).alias('prev_fault'),
        pl.col('fault_code').shift(-1).alias('next_fault'),
    ])

    # Classify events
    events = fault_df.with_columns([
        pl.when((pl.col('prev_fault') == 0) & (pl.col('fault_code') > 0))
          .then(pl.lit('onset'))
          .when((pl.col('prev_fault') > 0) & (pl.col('fault_code') == 0))
          .then(pl.lit('recovery'))
          .when((pl.col('prev_fault') > 0) & (pl.col('fault_code') > 0) & (pl.col('prev_fault') != pl.col('fault_code')))
          .then(pl.lit('transition'))
          .otherwise(pl.lit('stable'))
          .alias('event_type')
    ])

    return events


# =============================================================================
# LAYER 1: WHAT (Classification Features)
# =============================================================================

def get_what_features(vec_df: pl.DataFrame, date: 'date') -> Dict[str, float]:
    """Get classification-relevant features for a date."""

    # Top classification features from eval
    key_metrics = [
        'alpha', 'beta', 'omega', 'unconditional_vol',  # GARCH
        'spectral_slope', 'spectral_entropy',  # Spectral
        'permutation_entropy', 'sample_entropy',  # Entropy
    ]

    day_data = vec_df.filter(
        (pl.col('obs_date') == date) &
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT') &
        pl.col('metric_name').is_in(key_metrics)
    )

    if len(day_data) == 0:
        return {}

    # Aggregate across signals
    agg = day_data.group_by('metric_name').agg([
        pl.col('metric_value').mean().alias('mean'),
        pl.col('metric_value').std().alias('std'),
    ])

    features = {}
    for row in agg.iter_rows(named=True):
        features[f"what_{row['metric_name']}_mean"] = row['mean']
        features[f"what_{row['metric_name']}_std"] = row['std']

    return features


# =============================================================================
# LAYER 2: WHEN (Break Detection Features)
# =============================================================================

def get_when_features(vec_df: pl.DataFrame, date: 'date') -> Dict[str, float]:
    """Get break detection features for a date."""

    break_metrics = [
        'break_n', 'break_rate', 'break_is_accelerating',
        'dirac_n_impulses', 'dirac_mean_magnitude',
        'heaviside_n_steps', 'heaviside_mean_magnitude',
    ]

    day_data = vec_df.filter(
        (pl.col('obs_date') == date) &
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT') &
        pl.col('metric_name').is_in(break_metrics)
    )

    if len(day_data) == 0:
        return {}

    # Sum breaks across signals (total system discontinuity)
    agg = day_data.group_by('metric_name').agg([
        pl.col('metric_value').sum().alias('sum'),
        pl.col('metric_value').max().alias('max'),
    ])

    features = {}
    for row in agg.iter_rows(named=True):
        features[f"when_{row['metric_name']}_sum"] = row['sum']
        features[f"when_{row['metric_name']}_max"] = row['max']

    return features


# =============================================================================
# LAYER 3: MODE (Behavioral Trajectory)
# =============================================================================

def get_mode_features(modes_df: pl.DataFrame, date: 'date') -> Dict[str, float]:
    """Get mode features for a date."""

    if modes_df is None:
        return {}

    day_data = modes_df.filter(
        (pl.col('obs_date') == date) &
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    if len(day_data) == 0:
        return {}

    features = {}

    # Mode affinity stats
    if 'mode_affinity' in day_data.columns:
        affinities = day_data['mode_affinity'].drop_nulls()
        if len(affinities) > 0:
            features['mode_affinity_mean'] = float(affinities.mean())
            features['mode_affinity_min'] = float(affinities.min())
            features['mode_affinity_std'] = float(affinities.std()) if len(affinities) > 1 else 0

    # Mode entropy stats (high entropy = unstable)
    if 'mode_entropy' in day_data.columns:
        entropy = day_data['mode_entropy'].drop_nulls()
        if len(entropy) > 0:
            features['mode_entropy_mean'] = float(entropy.mean())
            features['mode_entropy_max'] = float(entropy.max())

    # Mode distribution (how many signals in each mode)
    if 'mode_id' in day_data.columns:
        mode_counts = day_data.group_by('mode_id').len()
        n_modes = len(mode_counts)
        features['mode_n_active'] = n_modes

        # Mode concentration (high = most signals in one mode)
        if n_modes > 0:
            counts = mode_counts['len'].to_numpy()
            max_frac = np.max(counts) / np.sum(counts)
            features['mode_concentration'] = float(max_frac)

    return features


def compute_mode_trajectory(modes_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute mode trajectory for each signal over time.

    Returns DataFrame with:
    - signal_id
    - obs_date
    - mode_id
    - mode_affinity
    - mode_changed (bool) - did mode change from previous day?
    - affinity_delta - change in affinity
    """

    if modes_df is None:
        return None

    tep_modes = modes_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    ).sort(['signal_id', 'obs_date'])

    if len(tep_modes) == 0:
        return None

    # Compute changes per signal
    trajectory = tep_modes.with_columns([
        pl.col('mode_id').shift(1).over('signal_id').alias('prev_mode'),
        pl.col('mode_affinity').shift(1).over('signal_id').alias('prev_affinity'),
    ])

    trajectory = trajectory.with_columns([
        (pl.col('mode_id') != pl.col('prev_mode')).alias('mode_changed'),
        (pl.col('mode_affinity') - pl.col('prev_affinity')).alias('affinity_delta'),
    ])

    return trajectory


# =============================================================================
# MODE 1 DETECTION (Precursor/Transition Signal)
# =============================================================================

def get_mode1_signal(modes_df: pl.DataFrame, date: 'date') -> Dict:
    """
    Detect Mode 1 presence for a date.

    Mode 1 is a rare (~4%) transitional mode that appears:
    - Before fault onsets (20.4%)
    - At fault onsets (7.8%)
    - Dominated by XMEAS03 (D Feed - upstream signal)

    Returns:
        Dict with mode1 detection info
    """
    if modes_df is None:
        return {'mode1_count': 0, 'mode1_present': False}

    day_data = modes_df.filter(
        (pl.col('obs_date') == date) &
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    if len(day_data) == 0:
        return {'mode1_count': 0, 'mode1_present': False}

    # Count Mode 1 assignments
    mode1_data = day_data.filter(pl.col('mode_id') == 1)
    mode1_count = len(mode1_data)

    # Get signals in Mode 1
    mode1_signals = mode1_data['signal_id'].to_list() if mode1_count > 0 else []

    # Check if XMEAS03 is in Mode 1 (the dominant signal)
    xmeas03_in_mode1 = any('XMEAS03' in ind for ind in mode1_signals)

    # Mode 1 affinity (lower = more uncertainty)
    mode1_affinity = float(mode1_data['mode_affinity'].mean()) if mode1_count > 0 else None

    return {
        'mode1_count': mode1_count,
        'mode1_present': mode1_count > 0,
        'mode1_signals': mode1_signals[:5],  # Top 5 for display
        'mode1_xmeas03': xmeas03_in_mode1,
        'mode1_affinity': mode1_affinity,
    }


def get_mode1_warning(modes_df: pl.DataFrame, onset_date: 'date', window_days: int = 7) -> Dict:
    """
    Check for Mode 1 presence in window around fault onset.

    Mode 1 appearing before or at onset = early warning signal.
    """
    from datetime import timedelta

    if modes_df is None:
        return {'mode1_warning': False, 'mode1_before': 0, 'mode1_at': 0}

    # Check window before onset
    before_dates = [onset_date - timedelta(days=d) for d in range(1, window_days + 1)]
    mode1_before = 0
    mode1_before_signals = []

    for d in before_dates:
        signal = get_mode1_signal(modes_df, d)
        if signal['mode1_present']:
            mode1_before += signal['mode1_count']
            mode1_before_signals.extend(signal.get('mode1_signals', []))

    # Check at onset
    at_signal = get_mode1_signal(modes_df, onset_date)
    mode1_at = at_signal['mode1_count']

    # Mode 1 warning: present before OR at onset
    mode1_warning = (mode1_before > 0) or (mode1_at > 0)

    # Unique signals that entered Mode 1
    unique_mode1_signals = list(set(mode1_before_signals + at_signal.get('mode1_signals', [])))

    return {
        'mode1_warning': mode1_warning,
        'mode1_before': mode1_before,
        'mode1_at': mode1_at,
        'mode1_total': mode1_before + mode1_at,
        'mode1_signals': unique_mode1_signals[:10],  # Top 10
    }


# =============================================================================
# INTEGRATED ANALYSIS
# =============================================================================

def analyze_fault_window(
    vec_df: pl.DataFrame,
    modes_df: pl.DataFrame,
    onset_date: 'date',
    fault_id: int,
    window_before: int = 5,
    window_after: int = 5
) -> Dict:
    """
    Analyze WHAT + WHEN + MODE around a fault onset.

    Returns:
        Dict with integrated analysis
    """
    from datetime import timedelta

    result = {
        'onset_date': onset_date,
        'fault_id': f'IDV{fault_id:02d}',
        'timeline': [],
    }

    dates = [onset_date + timedelta(days=d) for d in range(-window_before, window_after + 1)]

    for d in dates:
        day_analysis = {
            'date': d,
            'days_from_onset': (d - onset_date).days,
        }

        # Layer 1: WHAT
        what = get_what_features(vec_df, d)
        day_analysis['what'] = what

        # Layer 2: WHEN
        when = get_when_features(vec_df, d)
        day_analysis['when'] = when

        # Layer 3: MODE
        mode = get_mode_features(modes_df, d)
        day_analysis['mode'] = mode

        result['timeline'].append(day_analysis)

    return result


def summarize_pre_onset_signals(analysis: Dict, mode1_info: Dict = None) -> Dict:
    """Extract key signals from pre-onset window."""

    pre_onset = [t for t in analysis['timeline'] if t['days_from_onset'] < 0]

    if not pre_onset:
        return {}

    # Aggregate WHEN signals (breaks)
    break_sum = sum(t['when'].get('when_break_n_sum', 0) or 0 for t in pre_onset)
    dirac_sum = sum(t['when'].get('when_dirac_n_impulses_sum', 0) or 0 for t in pre_onset)
    heaviside_sum = sum(t['when'].get('when_heaviside_n_steps_sum', 0) or 0 for t in pre_onset)

    # Mode affinity trend (is it dropping?)
    affinities = [t['mode'].get('mode_affinity_mean', None) for t in pre_onset]
    affinities = [a for a in affinities if a is not None]

    affinity_trend = 0
    if len(affinities) >= 2:
        affinity_trend = affinities[-1] - affinities[0]  # Negative = dropping

    # Mode 1 signal (precursor/transition mode)
    mode1_warning = False
    mode1_before = 0
    mode1_at = 0
    if mode1_info:
        mode1_warning = mode1_info.get('mode1_warning', False)
        mode1_before = mode1_info.get('mode1_before', 0)
        mode1_at = mode1_info.get('mode1_at', 0)

    # Combined warning: WHEN signals OR affinity drop OR Mode 1 presence
    has_warning = (
        (break_sum + dirac_sum + heaviside_sum) > 0 or
        affinity_trend < -0.1 or
        mode1_warning
    )

    return {
        'break_count': break_sum,
        'dirac_count': dirac_sum,
        'heaviside_count': heaviside_sum,
        'total_discontinuities': break_sum + dirac_sum + heaviside_sum,
        'affinity_trend': affinity_trend,
        'mode1_warning': mode1_warning,
        'mode1_before': mode1_before,
        'mode1_at': mode1_at,
        'has_warning': has_warning,
    }


def run_integrated_assessment(domain: str):
    """Run integrated WHAT + WHEN + MODE assessment."""

    print("=" * 100)
    print("TEP INTEGRATED ASSESSMENT: WHAT + WHEN + MODE")
    print("=" * 100)
    print()
    print("Three layers of regime understanding:")
    print("  WHAT  - Which fault type (classification features)")
    print("  WHEN  - Timing of change (break/dirac/heaviside)")
    print("  MODE  - Behavioral trajectory (affinity + sequence)")
    print()

    # Load data
    print("Loading data...")
    vec_df, obs_df, field_df, modes_df = load_all_data(domain)
    print(f"  Vector data: {len(vec_df):,} rows")
    print(f"  Modes data: {'available' if modes_df is not None else 'NOT FOUND'}")

    # Get fault events
    print("\nIdentifying fault events...")
    events = get_fault_events(obs_df)
    onsets = events.filter(pl.col('event_type') == 'onset')
    print(f"  Found {len(onsets)} fault onsets")

    # Analyze each onset
    print()
    print("=" * 100)
    print("INTEGRATED ANALYSIS PER FAULT ONSET")
    print("=" * 100)

    results = []
    warning_count = 0

    mode1_warnings = 0

    for row in onsets.head(30).iter_rows(named=True):  # Sample first 30
        onset_date = row['obs_date']
        fault_id = int(row['fault_code'])

        analysis = analyze_fault_window(vec_df, modes_df, onset_date, fault_id)

        # Get Mode 1 specific warning
        mode1_info = get_mode1_warning(modes_df, onset_date, window_days=7)
        signals = summarize_pre_onset_signals(analysis, mode1_info)

        results.append({
            'fault_id': f'IDV{fault_id:02d}',
            'onset_date': onset_date,
            'signals': signals,
            'mode1_info': mode1_info,
        })

        if signals.get('has_warning', False):
            warning_count += 1
        if signals.get('mode1_warning', False):
            mode1_warnings += 1

        # Print summary with Mode 1 signal
        disc = signals.get('total_discontinuities', 0)
        aff = signals.get('affinity_trend', 0)
        m1 = "M1!" if signals.get('mode1_warning') else ""
        warn = "EARLY WARNING" if signals.get('has_warning') else ""

        print(f"  IDV{fault_id:02d} @ {onset_date}: "
              f"breaks={signals.get('break_count', 0)}, "
              f"dirac={signals.get('dirac_count', 0)}, "
              f"heaviside={signals.get('heaviside_count', 0)}, "
              f"affinity_delta={aff:+.3f} {m1} {warn}")

    # Summary by fault type
    print()
    print("=" * 100)
    print("EARLY WARNING RATE BY FAULT TYPE")
    print("=" * 100)

    per_fault = defaultdict(lambda: {'total': 0, 'warned': 0})
    for r in results:
        fid = r['fault_id']
        per_fault[fid]['total'] += 1
        if r['signals'].get('has_warning'):
            per_fault[fid]['warned'] += 1

    for fid in sorted(per_fault.keys()):
        stats = per_fault[fid]
        rate = stats['warned'] / stats['total'] if stats['total'] > 0 else 0
        status = "+" if rate >= 0.7 else "~" if rate >= 0.3 else "-"
        print(f"  {status} {fid}: {rate:.0%} ({stats['warned']}/{stats['total']})")

    # Overall summary
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
    total_breaks = sum(r['signals'].get('break_count', 0) for r in results)
    total_dirac = sum(r['signals'].get('dirac_count', 0) for r in results)
    total_heaviside = sum(r['signals'].get('heaviside_count', 0) for r in results)

    print(f"  Break detector:  {total_breaks:,} breaks detected")
    print(f"  Dirac impulses:  {total_dirac:,} impulses detected")
    print(f"  Heaviside steps: {total_heaviside:,} steps detected")
    print()

    if modes_df is not None:
        affinity_drops = sum(1 for r in results if r['signals'].get('affinity_trend', 0) < -0.1)
        print(f"  Mode affinity drops: {affinity_drops} (pre-onset destabilization)")

        # Mode 1 summary
        print()
        print("=" * 100)
        print("MODE 1 DETECTION (Precursor/Transition Signal)")
        print("=" * 100)
        print()
        print(f"Mode 1 warnings: {mode1_warnings}/{len(results)} ({mode1_warnings/len(results):.1%})")
        print()

        # Mode 1 breakdown
        total_m1_before = sum(r['signals'].get('mode1_before', 0) for r in results)
        total_m1_at = sum(r['signals'].get('mode1_at', 0) for r in results)
        print(f"  Mode 1 before onset: {total_m1_before} signal-days")
        print(f"  Mode 1 at onset:     {total_m1_at} signal-days")
        print()

        # Which signals triggered Mode 1 warnings
        all_mode1_signals = []
        for r in results:
            if 'mode1_info' in r and r['mode1_info']:
                all_mode1_signals.extend(r['mode1_info'].get('mode1_signals', []))

        if all_mode1_signals:
            from collections import Counter
            mode1_counter = Counter(all_mode1_signals)
            print("Top Mode 1 signals (precursors):")
            for ind, count in mode1_counter.most_common(5):
                print(f"    {ind}: {count} occurrences")
        print()

        print("Mode 1 interpretation:")
        print("  - Mode 1 is a rare (~4%) transitional mode")
        print("  - Dominated by XMEAS03 (D Feed - upstream reactor input)")
        print("  - Lower affinity (0.937) = system in uncertain state")
        print("  - Appears BEFORE structural breaks become visible")
        print("  - M1! flag = Mode 1 detected in pre-onset window")

    print()
    print("=" * 100)

    return results


def main():
    parser = argparse.ArgumentParser(description='TEP Integrated Assessment')
    parser.add_argument('--domain', type=str, default=None)
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for TEP integrated assessment")
    os.environ["PRISM_DOMAIN"] = domain

    run_integrated_assessment(domain)


if __name__ == '__main__':
    main()

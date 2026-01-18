#!/usr/bin/env python3
"""
PRISM Batch Characterization Runner

Characterizes each signal ONCE using ALL available data.
Characterization describes WHAT KIND of process this is - structural, not temporal.
One row per signal.

Pattern: Read Parquet → Compute → Write Parquet (upsert)

Usage:
    python -m prism.entry_points.characterize
    python -m prism.entry_points.characterize --cohort climate
    python -m prism.entry_points.characterize --signals sensor_1,sensor_2
    python -m prism.entry_points.characterize --force  # Recompute all
"""

import argparse
import gc
import json
import sys
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import numpy as np
import polars as pl

from prism.engines.characterize import Characterizer, CharacterizationResult
from prism.cohorts import get_cohort
from prism.db.parquet_store import get_parquet_path, ensure_directories, table_exists
from prism.db.polars_io import read_parquet, upsert_parquet
from prism.utils.memory import force_gc, get_memory_usage_mb


# =============================================================================
# COHORT DEFINITIONS
# =============================================================================

COHORTS = {
    'climate': [
        'CO2_MONTHLY', 'CO2_ANNUAL', 'CO2_GROWTH_RATE',
        'CH4_MONTHLY', 'N2O_MONTHLY', 'SF6_MONTHLY',
        'GISS_TEMP_GLOBAL', 'GISS_TEMP_NH', 'GISS_TEMP_SH',
        'NOAA_TEMP_GLOBAL', 'NOAA_TEMP_LAND', 'NOAA_TEMP_OCEAN',
        'NAO_INDEX', 'AO_INDEX', 'PNA_INDEX', 'AAO_INDEX',
        'SOI_INDEX', 'SST_NINO34',
        'ARCTIC_SEA_ICE_EXTENT', 'ARCTIC_SEA_ICE_AREA',
        'ANTARCTIC_SEA_ICE_EXTENT', 'ANTARCTIC_SEA_ICE_AREA',
        'SEA_LEVEL_GLOBAL',
        'SUNSPOT_NUMBER',
    ],
}

# Default minimum observations for characterization
DEFAULT_MIN_OBSERVATIONS = 100


# =============================================================================
# DATA OPERATIONS (Polars/Parquet)
# =============================================================================

def get_min_observations() -> int:
    """Get min_observations from domain_config parquet, default 100 for daily data."""
    try:
        config_path = get_parquet_path('raw', 'domain_config')
        if config_path.exists():
            df = read_parquet(config_path)
            row = df.filter(pl.col('key') == 'min_observations')
            if len(row) > 0:
                return int(row['value'][0])
    except Exception:
        pass
    return DEFAULT_MIN_OBSERVATIONS


def get_available_signals() -> List[str]:
    """Get all signals with data in raw.observations parquet."""
    obs_path = get_parquet_path('raw', 'observations')
    if not obs_path.exists():
        return []

    df = pl.scan_parquet(obs_path).select('signal_id').unique().collect()
    return sorted(df['signal_id'].to_list())


def get_existing_characterizations() -> set:
    """Get set of signal_ids that have already been characterized."""
    char_path = get_parquet_path('raw', 'characterization')
    if not char_path.exists():
        return set()

    df = read_parquet(char_path, columns=['signal_id'])
    return set(df['signal_id'].to_list())


def get_all_signal_data(
    signal_id: str,
    observations_df: pl.DataFrame,
    min_observations: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Fetch ALL data for an signal from the observations DataFrame.

    Returns dict with values, dates, start_date, end_date, n_obs
    or None if insufficient data.
    """
    # Filter for this signal
    signal_df = observations_df.filter(pl.col('signal_id') == signal_id)

    if len(signal_df) < min_observations:
        return None

    # Sort by date and extract data
    signal_df = signal_df.sort('obs_date')

    dates = signal_df['obs_date'].to_numpy()
    values = signal_df['value'].to_numpy().astype(float)

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    dates = dates[valid_mask]

    if len(values) < min_observations:
        return None

    return {
        'values': values,
        'dates': dates,
        'start_date': dates[0],
        'end_date': dates[-1],
        'n_observations': len(values),
    }


# =============================================================================
# MAIN RUNNER (Signal Characterization)
# =============================================================================

def run_characterization(
    signals: Optional[List[str]] = None,
    cohort: Optional[str] = None,
    domain: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Characterize signals using ALL available data.

    Pattern: Read Parquet → Compute → Write Parquet (upsert)

    Args:
        signals: List of signal IDs (None = all available)
        cohort: Named cohort from COHORTS dict
        domain: Domain for cohort classification (e.g., 'climate', 'industrial')
        force: Recompute even if already characterized
        verbose: Print progress

    Returns:
        Summary dict with counts and any errors
    """
    # Ensure data directories exist
    ensure_directories()

    if verbose:
        print("Storage: Parquet files")
        print("Pattern: Read → Compute → Write (upsert)")
        print("=" * 70)

    results = {
        'processed': 0,
        'skipped': 0,
        'errors': [],
        'by_class': {},
        'signals': [],
    }

    # Validate observations parquet exists
    obs_path = get_parquet_path('raw', 'observations')
    if not obs_path.exists():
        raise RuntimeError(
            f"Observations file not found: {obs_path}\n"
            "Run data fetch first: python -m prism.entry_points.fetch --cmapss"
        )

    # Memory tracking
    start_memory = get_memory_usage_mb()
    if verbose:
        print(f"Starting memory: {start_memory:.1f} MB")

    # Load all observations once (more efficient for multiple signals)
    if verbose:
        print("Loading observations...")
    observations_df = read_parquet(obs_path)
    if verbose:
        print(f"  Loaded {len(observations_df):,} observations")

    # Determine which signals to process
    if cohort and cohort in COHORTS:
        target_signals = COHORTS[cohort]
        if verbose:
            print(f"Using cohort '{cohort}': {len(target_signals)} signals")
    elif signals:
        target_signals = signals
        if verbose:
            print(f"Characterizing {len(target_signals)} specified signals")
    else:
        target_signals = get_available_signals()
        if verbose:
            print(f"Processing all available signals: {len(target_signals)}")

    # Check what's already characterized (unless force)
    if not force:
        existing_ids = get_existing_characterizations()
        if existing_ids:
            before_count = len(target_signals)
            target_signals = [i for i in target_signals if i not in existing_ids]

            if verbose and before_count != len(target_signals):
                print(f"Skipping {before_count - len(target_signals)} already characterized")

    if not target_signals:
        if verbose:
            print("\nNo signals to characterize")
        return results

    # Get min_observations from config
    min_obs = get_min_observations()

    if verbose:
        print(f"\nCharacterizing {len(target_signals)} signals (using ALL data)")
        print(f"Min observations: {min_obs}")
        print("=" * 70)

    char = Characterizer()
    effective_domain = domain or 'industrial'

    # Collect results for batch write - COMPUTE → WRITE → RELEASE pattern
    BATCH_SIZE = 50  # Write every 50 signals
    batch_rows = []
    char_path = get_parquet_path('raw', 'characterization')

    for i, signal_id in enumerate(target_signals):
        try:
            # Get ALL data for this signal
            data_info = get_all_signal_data(signal_id, observations_df, min_obs)

            if data_info is None:
                if verbose:
                    print(f"  {signal_id}: SKIP (insufficient data, need {min_obs}+)")
                results['skipped'] += 1
                continue

            # Characterize using ALL data
            char_result = char.compute(
                values=data_info['values'],
                signal_id=signal_id,
                window_end=data_info['end_date'],
                dates=data_info['dates'],
            )

            # Get cohort classification
            sub_cohort = get_cohort(signal_id, effective_domain)

            # Convert dates to proper Python types
            start_date = data_info['start_date']
            end_date = data_info['end_date']

            # Handle numpy datetime64 or pandas Timestamp
            if hasattr(start_date, 'date'):
                start_date = start_date.date() if callable(getattr(start_date, 'date', None)) else start_date
            elif hasattr(start_date, 'astype'):
                import pandas as pd
                start_date = pd.Timestamp(start_date).date()

            if hasattr(end_date, 'date'):
                end_date = end_date.date() if callable(getattr(end_date, 'date', None)) else end_date
            elif hasattr(end_date, 'astype'):
                import pandas as pd
                end_date = pd.Timestamp(end_date).date()

            # Build row dict
            row = {
                'signal_id': signal_id,
                'sub_cohort': sub_cohort,
                'ax_stationarity': float(char_result.ax_stationarity),
                'ax_memory': float(char_result.ax_memory),
                'ax_periodicity': float(char_result.ax_periodicity),
                'ax_complexity': float(char_result.ax_complexity),
                'ax_determinism': float(char_result.ax_determinism),
                'ax_volatility': float(char_result.ax_volatility),
                'dynamical_class': char_result.dynamical_class,
                'valid_engines': json.dumps(char_result.valid_engines),
                'metric_weights': json.dumps(char_result.metric_weights),
                'return_method': char_result.return_method,
                'frequency': char_result.frequency,
                'avg_gap_days': char_result.avg_gap_days,
                'max_gap_days': char_result.max_gap_days,
                'is_step_function': char_result.is_step_function,
                'step_duration_mean': char_result.step_duration_mean,
                'unique_value_ratio': char_result.unique_value_ratio,
                'change_ratio': char_result.change_ratio,
                'quote_convention': char_result.quote_convention,
                'n_observations': data_info['n_observations'],
                'data_start': start_date,
                'data_end': end_date,
                'computed_at': char_result.computed_at,
                'computation_ms': char_result.computation_ms,
                'memory_method': char_result.memory_method,
                # Discontinuity detection
                'n_breaks': char_result.n_breaks,
                'break_rate': float(char_result.break_rate),
                'break_pattern': char_result.break_pattern,
                'has_steps': char_result.has_steps,
                'has_impulses': char_result.has_impulses,
                'heaviside_n_steps': char_result.heaviside_n_steps,
                'heaviside_mean_magnitude': float(char_result.heaviside_mean_magnitude),
                'dirac_n_impulses': char_result.dirac_n_impulses,
                'dirac_mean_magnitude': float(char_result.dirac_mean_magnitude),
            }
            batch_rows.append(row)

            # Track results
            results['processed'] += 1
            dyn_class = char_result.dynamical_class
            results['by_class'][dyn_class] = results['by_class'].get(dyn_class, 0) + 1
            results['signals'].append({
                'signal_id': signal_id,
                'dynamical_class': dyn_class,
                'n_observations': data_info['n_observations'],
            })

            if verbose:
                print(f"  [{i+1}/{len(target_signals)}] {signal_id}: {dyn_class}")

            # Batch write - COMPUTE → WRITE → RELEASE pattern
            if len(batch_rows) >= BATCH_SIZE:
                df = pl.DataFrame(batch_rows)
                upsert_parquet(df, char_path, key_cols=['signal_id'])
                row_count = len(batch_rows)

                # RELEASE
                del df
                del batch_rows
                batch_rows = []
                force_gc()

                current_mem = get_memory_usage_mb()
                if verbose:
                    print(f"    -> Batch written ({row_count} rows) [mem: {current_mem:.0f} MB]")

        except Exception as e:
            results['errors'].append({'signal': signal_id, 'error': str(e)})
            if verbose:
                print(f"  {signal_id}: ERROR - {e}")

    # Final batch write - WRITE → RELEASE
    if batch_rows:
        df = pl.DataFrame(batch_rows)
        row_count = upsert_parquet(df, char_path, key_cols=['signal_id'])

        # RELEASE
        del df
        del batch_rows
        force_gc()

        if verbose:
            print(f"\nWrote final batch to {char_path}")
            print(f"Total rows in characterization: {row_count}")

    # Release observations (no longer needed)
    del observations_df
    force_gc()

    # Summary with memory delta
    end_memory = get_memory_usage_mb()
    delta = end_memory - start_memory

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Processed: {results['processed']}")
        print(f"Skipped (insufficient data): {results['skipped']}")
        print(f"Errors: {len(results['errors'])}")
        print(f"Memory: {start_memory:.0f} → {end_memory:.0f} MB (Δ{delta:+.0f} MB)")
        if results['by_class']:
            print("\nBy dynamical class:")
            for cls, count in sorted(results['by_class'].items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Characterize signals (one-time, using ALL data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All signals in database
  python -m prism.entry_points.characterize

  # By domain (for cohort classification)
  python -m prism.entry_points.characterize --domain climate
  python -m prism.entry_points.characterize --domain cmapss

  # By cohort
  python -m prism.entry_points.characterize --cohort climate

  # Specific signals
  python -m prism.entry_points.characterize --signals sensor_1,sensor_2

  # Force recompute (if engine logic changed)
  python -m prism.entry_points.characterize --force

  # List cohorts
  python -m prism.entry_points.characterize --list-cohorts
        """
    )

    parser.add_argument('--signals', type=str,
                        help='Comma-separated signal IDs')
    parser.add_argument('--cohort', type=str,
                        help=f'Named cohort: {", ".join(COHORTS.keys())}')
    parser.add_argument('--domain', type=str,
                        help='Domain for cohort classification (climate, industrial, etc.)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if already characterized')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')
    parser.add_argument('--list-cohorts', action='store_true',
                        help='List available cohorts and exit')

    # Testing mode - REQUIRED to use any limiting flags
    parser.add_argument('--testing', action='store_true',
                        help='Enable testing mode. REQUIRED to use limiting flags (--signals, --cohort). Without --testing, all limiting flags are ignored and full run executes.')

    args = parser.parse_args()

    # ==========================================================================
    # CRITICAL: --testing guard
    # Without --testing, ALL limiting flags are ignored and full run executes.
    # This prevents accidentally running partial computations for hours/days.
    # ==========================================================================
    if not args.testing and not args.list_cohorts:
        limiting_flags_used = []
        if args.signals:
            limiting_flags_used.append('--signals')
        if args.cohort:
            limiting_flags_used.append('--cohort')

        if limiting_flags_used:
            print("=" * 80)
            print("WARNING: LIMITING FLAGS IGNORED - --testing not specified")
            print(f"Ignored flags: {', '.join(limiting_flags_used)}")
            print("Running FULL computation instead. Use --testing to enable limiting flags.")
            print("=" * 80)

        # Override to full defaults - characterize ALL signals
        args.signals = None
        args.cohort = None

    if args.list_cohorts:
        print("\nAvailable cohorts:")
        print("-" * 50)
        for name, inds in COHORTS.items():
            print(f"\n  {name}: {len(inds)} signals")
            for ind in inds[:8]:
                print(f"    - {ind}")
            if len(inds) > 8:
                print(f"    ... and {len(inds) - 8} more")
        print()
        return

    # Parse arguments
    signals = args.signals.split(',') if args.signals else None

    # Run signal characterization
    results = run_characterization(
        signals=signals,
        cohort=args.cohort,
        domain=args.domain,
        force=args.force,
        verbose=not args.quiet,
    )

    # Exit code
    if results['errors']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()

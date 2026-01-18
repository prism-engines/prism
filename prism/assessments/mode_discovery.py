"""
PRISM Mode Discovery Assessment
===============================

Discovers behavioral modes from Laplace field fingerprints and outputs
results to timestamped parquet files for reproducibility.

Modes are DISCOVERED groupings based on similar Laplace dynamics
(gradient/divergence patterns), unlike cohorts which are PREDEFINED.

Key Insight: Low affinity / high entropy = signal changing modes = REGIME TRANSITION SIGNAL

Output Files (saved to data/{domain}/assessments/):
    - {domain}_modes_{timestamp}.parquet      - Mode assignments per signal
    - {domain}_mode_geometry_{timestamp}.parquet - Geometry metrics per mode

Usage:
    python -m prism.assessments.mode_discovery --domain cheme
    python -m prism.assessments.mode_discovery --domain cheme --max-modes 10
    python -m prism.assessments.mode_discovery --domain cheme --exclude FAULT

Example Output:
    data/cheme/assessments/cheme_modes_20260117_162500.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from prism.db.parquet_store import get_parquet_path, ensure_directories
from prism.utils.domain import require_domain
from prism.modules.modes import discover_modes


def get_assessment_output_path(domain: str, name: str) -> Path:
    """
    Get timestamped output path for assessment results.

    Format: data/{domain}/assessments/{domain}_{name}_{YYYYMMDD_HHMMSS}.parquet
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"data/{domain}/assessments")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{domain}_{name}_{timestamp}.parquet"


def compute_mode_geometry(
    observations: pl.DataFrame,
    signal_ids: List[str],
) -> Dict[str, Any]:
    """Compute basic geometry metrics for a mode's signals."""
    if len(signal_ids) < 2:
        return {}

    # Filter to mode signals
    filtered = observations.filter(
        pl.col('signal_id').is_in(signal_ids)
    ).select(['obs_date', 'signal_id', 'value'])

    if filtered.is_empty():
        return {}

    # Pivot to matrix
    filtered = filtered.group_by(['signal_id', 'obs_date']).agg(
        pl.col('value').last()
    )

    pivoted = filtered.pivot(
        on='signal_id',
        index='obs_date',
        values='value'
    ).sort('obs_date').drop_nulls()

    if pivoted.is_empty() or len(pivoted.columns) < 3:
        return {}

    # Convert to numpy for stats
    cols = [c for c in pivoted.columns if c != 'obs_date']
    matrix = np.column_stack([pivoted[c].to_numpy() for c in cols])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix.T)
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    return {
        'n_signals': len(cols),
        'n_observations': len(pivoted),
        'correlation_mean': float(np.nanmean(upper_tri)),
        'correlation_std': float(np.nanstd(upper_tri)),
        'correlation_min': float(np.nanmin(upper_tri)),
        'correlation_max': float(np.nanmax(upper_tri)),
    }


def run_mode_discovery(
    domain: str,
    max_modes: int = 10,
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run mode discovery assessment and save results to timestamped parquet.

    Args:
        domain: Domain identifier
        max_modes: Maximum modes to discover
        exclude_patterns: Signal patterns to exclude (e.g., ['FAULT'])
        verbose: Print progress

    Returns:
        Summary with output paths
    """
    ensure_directories(domain)

    if exclude_patterns is None:
        exclude_patterns = []

    # Load signal field (Laplace data)
    field_path = get_parquet_path('vector', 'signal_field', domain)
    if not Path(field_path).exists():
        raise FileNotFoundError(f"Signal field not found: {field_path}")

    if verbose:
        print("=" * 80)
        print("PRISM MODE DISCOVERY ASSESSMENT")
        print("=" * 80)
        print(f"Domain: {domain}")
        print(f"Max modes: {max_modes}")
        print(f"Exclude patterns: {exclude_patterns}")
        print()

    # Load field data
    if verbose:
        print("Step 1: Loading signal field data...")

    field_df = pl.read_parquet(field_path)

    # Get unique signals
    all_signals = field_df['signal_id'].unique().to_list()

    # Apply exclusions
    if exclude_patterns:
        original_count = len(all_signals)
        for pattern in exclude_patterns:
            all_signals = [ind for ind in all_signals if pattern not in ind]
        if verbose:
            excluded = original_count - len(all_signals)
            print(f"  Excluded {excluded} signals matching {exclude_patterns}")

    if verbose:
        print(f"  Analyzing {len(all_signals)} signals")

    # Discover modes
    if verbose:
        print()
        print("Step 2: Discovering behavioral modes from Laplace fingerprints...")

    modes_df = discover_modes(
        field_df,
        domain_id=domain,
        cohort_id='default',
        signals=all_signals,
        max_modes=max_modes,
    )

    if modes_df is None or len(modes_df) == 0:
        print("  No modes discovered (insufficient data)")
        return {'status': 'no_modes', 'n_modes': 0}

    n_modes = modes_df['mode_id'].nunique()

    if verbose:
        print(f"  Discovered {n_modes} behavioral modes")
        print()
        print("  Mode Summary:")
        print("  " + "-" * 70)
        print(f"  {'Mode':<6} {'Count':<8} {'Affinity':<10} {'Entropy':<10} {'Top Signals'}")
        print("  " + "-" * 70)

        for mode_id in sorted(modes_df['mode_id'].unique()):
            mode_data = modes_df[modes_df['mode_id'] == mode_id]
            count = len(mode_data)
            aff = mode_data['mode_affinity'].mean()
            ent = mode_data['mode_entropy'].mean()
            top_inds = mode_data['signal_id'].head(3).tolist()
            top_str = ', '.join(top_inds)
            if count > 3:
                top_str += f" (+{count-3} more)"
            print(f"  {mode_id:<6} {count:<8} {aff:<10.3f} {ent:<10.3f} {top_str}")

    # Load observations for geometry
    if verbose:
        print()
        print("Step 3: Computing geometry per mode...")

    obs_path = get_parquet_path('raw', 'observations', domain)
    observations = pl.read_parquet(obs_path)

    # Compute geometry for each mode
    geometry_records = []
    computed_at = datetime.now()

    for mode_id in sorted(modes_df['mode_id'].unique()):
        mode_data = modes_df[modes_df['mode_id'] == mode_id]
        signal_ids = mode_data['signal_id'].tolist()

        metrics = compute_mode_geometry(observations, signal_ids)

        if not metrics:
            continue

        record = {
            'domain_id': domain,
            'mode_id': int(mode_id),
            'n_signals': len(signal_ids),
            'mode_affinity_mean': float(mode_data['mode_affinity'].mean()),
            'mode_entropy_mean': float(mode_data['mode_entropy'].mean()),
            **metrics,
            'computed_at': computed_at,
        }
        geometry_records.append(record)

        if verbose:
            corr = metrics.get('correlation_mean', 0)
            print(f"  Mode {mode_id}: {len(signal_ids)} signals, corr_mean={corr:.3f}")

    # Save mode assignments
    modes_output = get_assessment_output_path(domain, 'modes')
    modes_pl = pl.from_pandas(modes_df)
    modes_pl.write_parquet(modes_output)

    # Save geometry
    geometry_output = get_assessment_output_path(domain, 'mode_geometry')
    if geometry_records:
        geometry_df = pl.DataFrame(geometry_records, infer_schema_length=None)
        geometry_df.write_parquet(geometry_output)

    if verbose:
        print()
        print("=" * 80)
        print("MODE DISCOVERY COMPLETE")
        print("=" * 80)
        print(f"Modes discovered: {n_modes}")
        print(f"Signals analyzed: {len(all_signals)}")
        print()
        print("Output files:")
        print(f"  Modes:    {modes_output}")
        print(f"  Geometry: {geometry_output}")
        print()
        print("Mode Assignments:")
        print("-" * 80)

        # Print full mode assignments
        for mode_id in sorted(modes_df['mode_id'].unique()):
            mode_data = modes_df[modes_df['mode_id'] == mode_id].sort_values('signal_id')
            signals = mode_data['signal_id'].tolist()
            print(f"\nMODE {mode_id} ({len(signals)} signals):")
            for ind in signals:
                row = mode_data[mode_data['signal_id'] == ind].iloc[0]
                print(f"  {ind:<25} affinity={row['mode_affinity']:.3f}")

    return {
        'status': 'success',
        'n_modes': n_modes,
        'n_signals': len(all_signals),
        'modes_output': str(modes_output),
        'geometry_output': str(geometry_output),
    }


def main():
    parser = argparse.ArgumentParser(
        description='PRISM Mode Discovery Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Discovers behavioral modes from Laplace field fingerprints.

Modes are DISCOVERED groupings based on gradient/divergence patterns,
unlike cohorts which are PREDEFINED physical/logical groupings.

Output saved to: data/{domain}/assessments/{domain}_modes_{timestamp}.parquet

Examples:
  python -m prism.assessments.mode_discovery --domain cheme
  python -m prism.assessments.mode_discovery --domain cheme --exclude FAULT
  python -m prism.assessments.mode_discovery --domain cheme --max-modes 5
"""
    )

    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to analyze (prompts if not specified)')
    parser.add_argument('--max-modes', type=int, default=10,
                        help='Maximum modes to discover (default: 10)')
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                        help='Signal patterns to exclude (e.g., FAULT)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Domain selection
    domain = require_domain(args.domain, "Select domain for mode discovery")
    os.environ["PRISM_DOMAIN"] = domain

    run_mode_discovery(
        domain=domain,
        max_modes=args.max_modes,
        exclude_patterns=args.exclude if args.exclude else None,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

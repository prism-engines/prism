"""
Stage 09: Dynamics Entry Point
==============================

DEPRECATED: This stage is now a thin wrapper around stage_08_ftle.
Stage 08 now computes stability classification directly, making
stage_09's redundant FTLE + classify_stability computation unnecessary.

For backward compatibility, this module delegates to stage_08 with
stability classification included.

Inputs:
    - observations.parquet

Output:
    - dynamics.parquet (same schema as ftle.parquet + stability column)
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from engines.entry_points.stage_08_ftle import run as _ftle_run


def run(
    observations_path: str,
    output_path: str = "dynamics.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    min_samples: int = 200,
    verbose: bool = True,
    intervention: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Run dynamics computation for all signals, per-cohort.

    DEPRECATED: Delegates to stage_08_ftle which now includes stability
    classification. This wrapper exists for backward compatibility.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for dynamics.parquet
        signal_column: Column with signal IDs (unused, kept for compat)
        value_column: Column with values (unused, kept for compat)
        index_column: Column with time index (unused, kept for compat)
        min_samples: Minimum samples for computation
        verbose: Print progress

    Returns:
        Dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 09: DYNAMICS (delegates to stage_08_ftle)")
        print("Per-signal per-cohort stability classification")
        print("=" * 70)

    # Delegate to stage_08 which now includes stability column
    result = _ftle_run(
        observations_path=observations_path,
        output_path=output_path,
        min_samples=min_samples,
        verbose=verbose,
        direction='forward',
        intervention=intervention,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 09: Dynamics (delegates to stage_08_ftle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEPRECATED: Now delegates to stage_08_ftle with stability classification.

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

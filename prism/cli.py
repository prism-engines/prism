"""
PRISM CLI

Command line interface for running PRISM manifests.

Usage:
    python -m prism run --manifest manifest.json
    python -m prism run --data ./observations.parquet --output ./results --engines rms,kurtosis,hurst

Examples:
    # Run from manifest file
    python -m prism run --manifest /path/to/manifest.json

    # Run with inline options
    python -m prism run \\
        --data /path/to/observations.parquet \\
        --output /tmp/results \\
        --engines rms,peak,kurtosis,hurst,entropy \\
        --pair-engines granger \\
        --symmetric-engines correlation,mutual_info \\
        --sql-engines zscore,statistics

    # List available engines
    python -m prism list
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_run(args):
    """Run engines on data."""
    from prism.runner import ManifestRunner, run_manifest

    if args.manifest:
        # Run from manifest file
        result = run_manifest(args.manifest)
    else:
        # Build manifest from CLI args
        if not args.data:
            print("Error: Either --manifest or --data is required")
            sys.exit(1)

        manifest = {
            'observations_path': str(Path(args.data).resolve()),
            'output_dir': str(Path(args.output).resolve()) if args.output else '/tmp/prism_output',
            'engines': {
                'signal': [],
                'pair': [],
                'symmetric_pair': [],
                'windowed': ['derivatives'],  # Always run derivatives
                'sql': [],
            },
            'params': {}
        }

        # Parse engine lists
        if args.engines:
            manifest['engines']['signal'] = [e.strip() for e in args.engines.split(',')]

        if args.pair_engines:
            manifest['engines']['pair'] = [e.strip() for e in args.pair_engines.split(',')]

        if args.symmetric_engines:
            manifest['engines']['symmetric_pair'] = [e.strip() for e in args.symmetric_engines.split(',')]

        if args.windowed_engines:
            manifest['engines']['windowed'] = [e.strip() for e in args.windowed_engines.split(',')]

        if args.sql_engines:
            manifest['engines']['sql'] = [e.strip() for e in args.sql_engines.split(',')]

        # Add manifold if requested
        if args.manifold:
            manifest['engines']['windowed'].append('manifold')

        runner = ManifestRunner(manifest)
        result = runner.run()

    print(f"\nResults written to: {result.get('output_dir', 'unknown')}")
    return 0


def cmd_list(args):
    """List available engines."""
    from prism.python_runner import SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
    from prism.sql_runner import SQL_ENGINES

    print("=" * 50)
    print("PRISM AVAILABLE ENGINES")
    print("=" * 50)

    print("\n[SIGNAL ENGINES] (one value per signal)")
    for eng in sorted(SIGNAL_ENGINES):
        print(f"  - {eng}")

    print("\n[PAIR ENGINES] (directional A→B)")
    for eng in sorted(PAIR_ENGINES):
        print(f"  - {eng}")

    print("\n[SYMMETRIC PAIR ENGINES] (A↔B)")
    for eng in sorted(SYMMETRIC_PAIR_ENGINES):
        print(f"  - {eng}")

    print("\n[WINDOWED ENGINES] (observation-level)")
    for eng in sorted(WINDOWED_ENGINES):
        print(f"  - {eng}")

    print("\n[SQL ENGINES] (DuckDB)")
    for eng in sorted(SQL_ENGINES):
        print(f"  - {eng}")

    return 0


def cmd_validate(args):
    """Validate output parquets."""
    from prism.validate_outputs import validate_directory

    if not args.output_dir:
        print("Error: --output-dir is required")
        sys.exit(1)

    validate_directory(args.output_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='prism',
        description='PRISM - Signal Analysis Engine Runner'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run command
    run_parser = subparsers.add_parser('run', help='Run engines on data')
    run_parser.add_argument('--manifest', '-m', help='Path to manifest JSON file')
    run_parser.add_argument('--data', '-d', help='Path to observations.parquet')
    run_parser.add_argument('--output', '-o', help='Output directory')
    run_parser.add_argument('--engines', '-e', help='Comma-separated signal engines')
    run_parser.add_argument('--pair-engines', help='Comma-separated pair engines')
    run_parser.add_argument('--symmetric-engines', help='Comma-separated symmetric pair engines')
    run_parser.add_argument('--windowed-engines', help='Comma-separated windowed engines')
    run_parser.add_argument('--sql-engines', help='Comma-separated SQL engines (DuckDB)')
    run_parser.add_argument('--manifold', action='store_true', help='Enable manifold computation')
    run_parser.set_defaults(func=cmd_run)

    # list command
    list_parser = subparsers.add_parser('list', help='List available engines')
    list_parser.set_defaults(func=cmd_list)

    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate output parquets')
    validate_parser.add_argument('--output-dir', required=True, help='Directory to validate')
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

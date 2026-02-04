"""
PRISM Command Line Interface

Usage:
    python -m prism <command> [args]

Commands:
    validate    Check prerequisites and validate input files
    signal      Compute signal vector from manifest
    status      Show pipeline status

Examples:
    python -m prism validate /path/to/data
    python -m prism signal /path/to/manifest.yaml
    python -m prism status /path/to/data

SafeCLI:
    Standardized argument parsing for entry points with safety checks:
    1. Named arguments (no positional ambiguity)
    2. Input file validation (must exist)
    3. Output file protection (can't overwrite inputs)
    4. Overwrite confirmation for non-default outputs
    5. Clear help text with INPUT/OUTPUT labels

    Usage in entry points:
        from prism.cli import SafeCLI

        cli = SafeCLI("State Geometry Engine")
        cli.add_input('signal_vector', '-s', help='signal_vector.parquet')
        cli.add_input('state_vector', '-t', help='state_vector.parquet')
        cli.add_output('output', default='state_geometry.parquet')
        args = cli.parse()
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set


# ============================================================
# SAFE CLI FOR ENTRY POINTS
# ============================================================

class SafeCLI:
    """
    Safe command-line interface with input/output validation.

    Prevents accidental data destruction by:
    - Validating input files exist
    - Preventing output from overwriting inputs
    - Confirming overwrites of existing files
    """

    def __init__(self, description: str, allow_overwrite: bool = False):
        """
        Initialize CLI parser.

        Args:
            description: Program description for --help
            allow_overwrite: If True, skip overwrite confirmation (for scripts)
        """
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.defaults: dict = {}
        self.allow_overwrite = allow_overwrite

        # Add global flags
        self.parser.add_argument(
            '-y', '--yes',
            action='store_true',
            help='Skip confirmation prompts (for automated scripts)'
        )
        self.parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            default=True,
            help='Verbose output (default: True)'
        )
        self.parser.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='Suppress output'
        )

    def add_input(
        self,
        name: str,
        flag: Optional[str] = None,
        help: str = '',
        required: bool = True
    ):
        """
        Add an input file argument.

        Args:
            name: Argument name (e.g., 'signal_vector')
            flag: Optional short flag (e.g., '-s')
            help: Help text
            required: Whether argument is required
        """
        self.inputs.append(name)

        # Convert to flag format
        flag_name = f"--{name.replace('_', '-')}"
        flags = [flag, flag_name] if flag else [flag_name]

        self.parser.add_argument(
            *flags,
            required=required,
            metavar='FILE',
            help=f'[INPUT] {help}'
        )

    def add_output(
        self,
        name: str = 'output',
        default: str = 'output.parquet',
        help: str = ''
    ):
        """
        Add an output file argument.

        Args:
            name: Argument name
            default: Default output filename
            help: Help text (auto-generated if empty)
        """
        self.outputs.append(name)
        self.defaults[name] = default

        flag_name = f"--{name.replace('_', '-')}"

        if not help:
            help = f'Output path (default: {default})'

        self.parser.add_argument(
            '-o' if name == 'output' else flag_name,
            f'--{name.replace("_", "-")}' if name != 'output' else '--output',
            default=default,
            metavar='FILE',
            help=f'[OUTPUT] {help}'
        )

    def add_flag(self, name: str, help: str = '', short: Optional[str] = None):
        """Add a boolean flag."""
        flags = [f'--{name.replace("_", "-")}']
        if short:
            flags.insert(0, short)
        self.parser.add_argument(*flags, action='store_true', help=help)

    def add_option(
        self,
        name: str,
        default=None,
        type=str,
        help: str = '',
        choices: Optional[List] = None
    ):
        """Add an option with a value."""
        self.parser.add_argument(
            f'--{name.replace("_", "-")}',
            default=default,
            type=type,
            choices=choices,
            help=help
        )

    def parse(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse arguments with safety validation.

        Args:
            args: Arguments to parse (default: sys.argv)

        Returns:
            Parsed arguments namespace

        Raises:
            SystemExit: On validation failure
        """
        parsed = self.parser.parse_args(args)

        # Handle quiet flag
        if parsed.quiet:
            parsed.verbose = False

        # Collect input paths
        input_paths: Set[str] = set()
        for input_name in self.inputs:
            path = getattr(parsed, input_name, None)
            if path:
                # Resolve to absolute path for comparison
                abs_path = str(Path(path).resolve())
                input_paths.add(abs_path)

                # Validate input exists
                if not Path(path).exists():
                    self._error(f"Input file not found: {path}")

        # Validate outputs
        for output_name in self.outputs:
            path = getattr(parsed, output_name, None)
            if path:
                abs_path = str(Path(path).resolve())

                # Check: output can't be an input
                if abs_path in input_paths:
                    self._error(
                        f"Output '{path}' matches an input file!\n"
                        f"       This would destroy your input data.\n"
                        f"       Use -o/--output to specify a different output path."
                    )

                # Check: warn before overwriting existing non-default file
                default = self.defaults.get(output_name)
                if (
                    Path(path).exists()
                    and path != default
                    and not self.allow_overwrite
                    and not parsed.yes
                ):
                    self._confirm_overwrite(path)

        return parsed

    def _error(self, message: str):
        """Print error and exit."""
        print(f"\n❌ ERROR: {message}", file=sys.stderr)
        sys.exit(1)

    def _confirm_overwrite(self, path: str):
        """Ask user to confirm overwrite."""
        print(f"\n⚠️  WARNING: Output file '{path}' already exists.")
        try:
            response = input("   Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("   Aborted.")
                sys.exit(0)
        except EOFError:
            # Non-interactive mode
            self._error(
                f"Output file '{path}' exists and running non-interactively.\n"
                f"       Use -y/--yes to overwrite, or choose a different output path."
            )


# ============================================================
# PRISM MAIN CLI
# ============================================================


def cmd_validate(args):
    """Validate prerequisites and input data."""
    from prism.validation import (
        check_prerequisites,
        validate_input,
        PrerequisiteError,
        ValidationError,
    )

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    print(f"Validating: {data_dir}")
    print()

    errors = []

    # Check prerequisites for signal_vector stage
    try:
        result = check_prerequisites(
            'signal_vector',
            str(data_dir),
            raise_on_missing=False,
            verbose=True,
        )
        if not result['satisfied']:
            errors.append(f"Missing prerequisites: {result['missing']}")
    except Exception as e:
        errors.append(f"Prerequisite check failed: {e}")

    print()

    # Validate input data (if prerequisites present)
    if not errors or args.force:
        try:
            report = validate_input(
                str(data_dir),
                raise_on_error=False,
                verbose=True,
            )
            if not report.valid:
                errors.extend(report.errors)
        except Exception as e:
            errors.append(f"Input validation failed: {e}")

    # Summary
    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("\nValidation PASSED")
        return 0


def cmd_status(args):
    """Show pipeline status."""
    from prism.validation.prerequisites import print_pipeline_status

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    print_pipeline_status(str(data_dir))
    return 0


def cmd_signal(args):
    """Compute signal vector."""
    from prism.entry_points.signal_vector import run_from_manifest

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1

    try:
        run_from_manifest(
            str(manifest_path),
            verbose=not args.quiet,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main():
    """PRISM CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='prism',
        description='PRISM Signal Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m prism validate /path/to/data
    python -m prism signal /path/to/manifest.yaml
    python -m prism status /path/to/data
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Check prerequisites and validate input files',
    )
    validate_parser.add_argument(
        'data_dir',
        help='Directory containing pipeline files',
    )
    validate_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Continue validation even if prerequisites missing',
    )

    # status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show pipeline status',
    )
    status_parser.add_argument(
        'data_dir',
        help='Directory containing pipeline files',
    )

    # signal command
    signal_parser = subparsers.add_parser(
        'signal',
        help='Compute signal vector from manifest',
    )
    signal_parser.add_argument(
        'manifest',
        help='Path to manifest.yaml',
    )
    signal_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        'validate': cmd_validate,
        'status': cmd_status,
        'signal': cmd_signal,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

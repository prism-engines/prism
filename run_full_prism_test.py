#!/usr/bin/env python3
"""
Full PRISM Test Runner

Runs ALL engines on observations.parquet and validates outputs.

Usage:
    python run_full_prism_test.py /path/to/observations.parquet [output_dir]
    python run_full_prism_test.py /path/to/observations.parquet --quick   # 1K rows per signal

Expected outputs:
    - observations_enriched.parquet (1.5M+ rows, 20+ columns)
    - manifold.parquet (600K+ rows)
    - primitives.parquet (11 rows, 40+ columns)
    - primitives_pairs.parquet (20+ rows)
    - geometry.parquet (10+ rows)
    - zscore.parquet (1.5M+ rows)
    - statistics.parquet (11 rows)
    - correlation.parquet (10+ rows)
    - regime_assignment.parquet (1.5M+ rows)
    - physics.parquet (eigenvalue coherence, energy, thermodynamics)
    - dynamics.parquet (Lyapunov, RQA, attractor dimension)
    - topology.parquet (persistent homology, Betti numbers)
    - information_flow.parquet (transfer entropy, causal networks)
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


# Full manifest with ALL engines
FULL_MANIFEST = {
    "job_id": f"full-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "callback_url": None,

    "observations_path": None,  # Set by script
    "output_dir": None,  # Set by script

    "engines": {
        # Signal-level engines (one value per signal)
        "signal": [
            # Basic statistics
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
            "skewness",

            # Vibration analysis
            "envelope",
            "harmonics",
            "frequency_bands",
            "spectral",

            # Nonlinear dynamics
            "hurst",
            "entropy",
            "lyapunov",
            "garch",
            "attractor",
            "dmd",
        ],

        # Pair engines (directional A→B)
        "pair": [
            "granger",
            "transfer_entropy",
        ],

        # Symmetric pair engines (A↔B)
        "symmetric_pair": [
            "cointegration",
            "mutual_info",
            "correlation",
        ],

        # Windowed engines (observation-level)
        "windowed": [
            "derivatives",
            "manifold",
            "stability",
            "rolling_rms",
            "rolling_kurtosis",
            "rolling_entropy",
            "rolling_hurst",
            "rolling_volatility",
            "rolling_mean",
            "rolling_std",
        ],

        # SQL engines
        "sql": [
            "zscore",
            "statistics",
            "correlation",
            "regime_assignment",
        ],

        # Physics stack (L1-L4)
        "physics": True,

        # Dynamics engine (Lyapunov, RQA)
        "dynamics": True,

        # Topology engine (persistent homology, Betti numbers)
        "topology": True,

        # Information flow engine (transfer entropy, causality)
        "information_flow": True,
    },

    "params": {
        "harmonics": {"sample_rate": 12000},
        "frequency_bands": {
            "sample_rate": 12000,
            "bands": {
                "low": [0, 500],
                "mid": [500, 2000],
                "high": [2000, 6000]
            }
        },
        "rolling_rms": {"window": 500},
        "rolling_kurtosis": {"window": 500},
        "rolling_entropy": {"window": 500},
        "rolling_hurst": {"window": 500},
        "rolling_volatility": {"window": 500},
        "rolling_mean": {"window": 100},
        "rolling_std": {"window": 100},
        "granger": {"max_lag": 10},
        "manifold": {"n_components": 3},
        "physics": {"n_baseline": 100, "coherence_window": 50},
        "dynamics": {"window_size": 100, "step_size": 10},
        "topology": {"window_size": 100, "step_size": 20},
        "information_flow": {"window_size": 100, "step_size": 20},
    }
}


# Expected outputs and their validation criteria
EXPECTED_OUTPUTS = {
    # Python runner outputs
    "observations_enriched.parquet": {
        "min_rows": 1000,
        "required_columns": ["entity_id", "signal_id", "I", "y", "dy", "d2y", "curvature"],
        "min_columns": 15,
        "description": "Observation-level: derivatives, rolling metrics"
    },
    "manifold.parquet": {
        "min_rows": 1000,
        "required_columns": ["entity_id", "I", "manifold_x", "manifold_y", "manifold_z"],
        "min_columns": 5,
        "description": "Cross-signal PCA projection"
    },
    "primitives.parquet": {
        "min_rows": 1,
        "required_columns": ["entity_id", "signal_id", "hurst", "sample_entropy"],
        "min_columns": 30,
        "description": "Signal-level: all computed metrics"
    },
    "primitives_pairs.parquet": {
        "min_rows": 1,
        "required_columns": ["entity_id", "source_signal", "target_signal", "granger_fstat"],
        "min_columns": 5,
        "description": "Directional pairs: granger, transfer entropy"
    },
    "geometry.parquet": {
        "min_rows": 1,
        "required_columns": ["entity_id", "signal_a", "signal_b", "correlation"],
        "min_columns": 5,
        "description": "Symmetric pairs: correlation, cointegration, mutual info"
    },

    # SQL runner outputs
    "zscore.parquet": {
        "min_rows": 1000,
        "required_columns": ["entity_id", "signal_id", "I", "z_score"],
        "min_columns": 4,
        "description": "Z-scores at observation level"
    },
    "statistics.parquet": {
        "min_rows": 1,
        "required_columns": ["entity_id", "signal_id", "mean", "std"],
        "min_columns": 5,
        "description": "Basic statistics per signal"
    },
    "correlation.parquet": {
        "min_rows": 1,
        "required_columns": ["entity_id", "signal_a", "signal_b", "correlation"],
        "min_columns": 4,
        "description": "SQL correlation (may duplicate geometry)"
    },
    "regime_assignment.parquet": {
        "min_rows": 1000,
        "required_columns": ["entity_id", "signal_id", "I", "regime_id"],
        "min_columns": 4,
        "description": "Regime labels at observation level"
    },
    "physics.parquet": {
        "min_rows": 10,
        "required_columns": ["entity_id", "I", "state_distance", "coherence", "effective_dim", "energy_proxy"],
        "min_columns": 15,
        "description": "Physics stack L1-L4 metrics (eigenvalue-based coherence)"
    },
    "dynamics.parquet": {
        "min_rows": 10,
        "required_columns": ["entity_id", "I", "lyapunov_max", "correlation_dim", "determinism", "recurrence_rate"],
        "min_columns": 10,
        "description": "Dynamics: Lyapunov exponents, attractor dimension, RQA metrics"
    },
    "topology.parquet": {
        "min_rows": 10,
        "required_columns": ["entity_id", "observation_idx", "betti_0", "betti_1", "topological_complexity"],
        "min_columns": 10,
        "description": "Topology: persistent homology, Betti numbers, topological complexity"
    },
    "information_flow.parquet": {
        "min_rows": 10,
        "required_columns": ["entity_id", "I", "n_causal_edges", "hierarchy_score", "max_transfer_entropy"],
        "min_columns": 10,
        "description": "Information flow: transfer entropy, causal network metrics"
    },
}


def run_test(observations_path: str, output_dir: str = None, quick: bool = False):
    """Run full PRISM test."""
    import polars as pl

    observations_path = Path(observations_path).resolve()

    if not observations_path.exists():
        print(f"Error: {observations_path} does not exist")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path(f"/tmp/prism_full_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        output_dir = Path(output_dir)

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Quick mode: slice to 1K rows per signal for fast validation
    if quick:
        print("QUICK MODE: Slicing to 1,000 rows per signal...")
        obs = pl.read_parquet(observations_path)
        obs = obs.group_by(["entity_id", "signal_id"]).head(1000)
        small_path = output_dir / "observations_small.parquet"
        obs.write_parquet(small_path)
        observations_path = small_path
        print(f"  Sliced to {len(obs):,} rows")

    # Update manifest
    manifest = FULL_MANIFEST.copy()
    manifest["observations_path"] = str(observations_path)
    manifest["output_dir"] = str(output_dir)

    # Save manifest for reference
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("FULL PRISM TEST")
    print("=" * 70)
    print(f"Input:  {observations_path}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    # Count engines
    total_engines = (
        len(manifest["engines"]["signal"]) +
        len(manifest["engines"]["pair"]) +
        len(manifest["engines"]["symmetric_pair"]) +
        len(manifest["engines"]["windowed"]) +
        len(manifest["engines"]["sql"])
    )
    print(f"Engines to run: {total_engines}")
    print(f"  Signal:         {len(manifest['engines']['signal'])}")
    print(f"  Pair:           {len(manifest['engines']['pair'])}")
    print(f"  Symmetric Pair: {len(manifest['engines']['symmetric_pair'])}")
    print(f"  Windowed:       {len(manifest['engines']['windowed'])}")
    print(f"  SQL:            {len(manifest['engines']['sql'])}")
    print()

    # Run PRISM
    print("-" * 70)
    print("RUNNING PRISM...")
    print("-" * 70)

    try:
        from prism.runner import ManifestRunner
        runner = ManifestRunner(manifest)
        result = runner.run()
        print(f"\nRunner result: {result}")
    except Exception as e:
        print(f"\nError running PRISM: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Validate outputs
    print()
    print("-" * 70)
    print("VALIDATING OUTPUTS...")
    print("-" * 70)

    validate_outputs(output_dir)


def validate_outputs(output_dir: Path):
    """Validate all expected outputs."""
    import pandas as pd

    output_dir = Path(output_dir)

    passed = 0
    failed = 0
    warnings = 0

    # Check each expected output
    for filename, criteria in EXPECTED_OUTPUTS.items():
        filepath = output_dir / filename
        print(f"\n[{filename}]")
        print(f"  Expected: {criteria['description']}")

        if not filepath.exists():
            print(f"  ✗ FILE MISSING")
            failed += 1
            continue

        try:
            df = pd.read_parquet(filepath)
            rows, cols = df.shape

            # Check row count
            if rows < criteria["min_rows"]:
                print(f"  ✗ Too few rows: {rows} < {criteria['min_rows']}")
                failed += 1
            else:
                print(f"  ✓ Rows: {rows:,}")
                passed += 1

            # Check column count
            if cols < criteria["min_columns"]:
                print(f"  ⚠ Few columns: {cols} < {criteria['min_columns']} expected")
                warnings += 1
            else:
                print(f"  ✓ Columns: {cols}")
                passed += 1

            # Check required columns
            missing_cols = [c for c in criteria["required_columns"] if c not in df.columns]
            if missing_cols:
                print(f"  ✗ Missing columns: {missing_cols}")
                failed += 1
            else:
                print(f"  ✓ Required columns present")
                passed += 1

            # Check for all-null columns
            null_cols = [c for c in df.columns if df[c].isna().all()]
            if null_cols:
                print(f"  ⚠ All-null columns: {null_cols}")
                warnings += 1

            # Show actual columns
            print(f"  Columns: {list(df.columns)}")

        except Exception as e:
            print(f"  ✗ Error reading: {e}")
            failed += 1

    # Check for unexpected files
    all_files = list(output_dir.glob("*.parquet"))
    expected_files = set(EXPECTED_OUTPUTS.keys())
    actual_files = set(f.name for f in all_files)

    extra_files = actual_files - expected_files
    if extra_files:
        print(f"\n[EXTRA FILES]")
        for f in extra_files:
            print(f"  + {f}")

    missing_files = expected_files - actual_files
    if missing_files:
        print(f"\n[MISSING FILES]")
        for f in missing_files:
            print(f"  - {f}")

    # Summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Warnings: {warnings}")
    print()

    if failed == 0:
        print("★ ALL VALIDATION CHECKS PASSED ★")
        return True
    else:
        print(f"✗ {failed} CHECKS FAILED")
        return False


def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nExpected output files:")
    for filename, criteria in EXPECTED_OUTPUTS.items():
        print(f"  {filename}")
        print(f"    {criteria['description']}")
        print(f"    Required columns: {criteria['required_columns']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full PRISM Test Runner - runs all engines and validates outputs"
    )
    parser.add_argument("observations_path", help="Path to observations.parquet")
    parser.add_argument("output_dir", nargs="?", help="Output directory (default: /tmp/prism_full_test_TIMESTAMP)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode: slice to 1K rows per signal for fast validation")

    args = parser.parse_args()

    run_test(args.observations_path, args.output_dir, quick=args.quick)

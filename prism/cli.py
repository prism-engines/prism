"""
PRISM CLI

Architecture:
  ORTHON creates typology.parquet → PRISM reads it and computes everything else
  Signal Vector → State Vector → Geometry → Geometry Dynamics → Dynamics → SQL

Usage:
    python -m prism <data_dir>                    # Run full pipeline
    python -m prism signal-vector <data_dir>      # Run signal vector (aggregate)
    python -m prism signal-vector-temporal <data_dir>  # Run signal vector (temporal)
    python -m prism state-vector <data_dir>       # Run state vector
    python -m prism geometry <data_dir>           # Run geometry (state + signal + pairwise)
    python -m prism geometry-dynamics <data_dir>  # Run geometry dynamics
    python -m prism lyapunov <data_dir>           # Run Lyapunov exponents
    python -m prism dynamics <data_dir>           # Run dynamics (RQA, info flow)
    python -m prism sql <data_dir>                # Run SQL engines

Note: Typology is created by ORTHON, not PRISM. PRISM expects typology.parquet to exist.
"""

import sys
from pathlib import Path


def check_typology(data_dir: Path) -> bool:
    """Check that typology.parquet exists (created by ORTHON)."""
    typology_path = data_dir / 'typology.parquet'
    if not typology_path.exists():
        print(f"ERROR: typology.parquet not found in {data_dir}")
        print("       Typology is created by ORTHON, not PRISM.")
        print("       Run ORTHON first to generate typology.parquet")
        return False
    return True


def run_signal_vector(data_dir: Path, temporal: bool = True) -> dict:
    """Run signal vector (temporal by default - includes I column)."""
    typology_path = data_dir / 'typology.parquet'
    obs_path = data_dir / 'observations.parquet'
    output_path = data_dir / 'signal_vector.parquet'

    if temporal:
        from prism.signal_vector_temporal import compute_signal_vector_temporal_sql
        print(f"[SIGNAL VECTOR] → {output_path}")
        result = compute_signal_vector_temporal_sql(
            str(obs_path), str(typology_path), str(output_path)
        )
    else:
        # Deprecated: aggregate mode loses I column
        from prism.signal_vector.runner_legacy import run_signal_vector as run_sv
        print(f"[SIGNAL VECTOR (aggregate - deprecated)] → {output_path}")
        result = run_sv(str(data_dir))

    return {'signal_vector': output_path, 'rows': len(result)}


def run_state_vector(data_dir: Path) -> dict:
    """Run state vector."""
    from prism.engines.state_vector import compute_state_vector

    signal_vector_path = data_dir / 'signal_vector.parquet'
    typology_path = data_dir / 'typology.parquet'
    output_path = data_dir / 'state_vector.parquet'

    print(f"[STATE VECTOR] → {output_path}")
    result = compute_state_vector(
        str(signal_vector_path), str(typology_path), str(output_path)
    )
    return {'state_vector': output_path, 'rows': len(result)}


def run_geometry(data_dir: Path) -> dict:
    """Run geometry pipeline (state_geometry + signal_geometry + signal_pairwise)."""
    signal_vector_path = data_dir / 'signal_vector.parquet'
    state_vector_path = data_dir / 'state_vector.parquet'

    results = {}

    # State geometry (eigenvalues)
    print(f"[STATE GEOMETRY]")
    from prism.engines.state_geometry import compute_state_geometry
    try:
        state_geom = compute_state_geometry(
            str(signal_vector_path), str(state_vector_path),
            str(data_dir / 'state_geometry.parquet')
        )
        results['state_geometry'] = len(state_geom)
    except Exception as e:
        print(f"  Warning: {e}")
        results['state_geometry'] = 0

    # Signal geometry
    print(f"[SIGNAL GEOMETRY]")
    from prism.engines.signal_geometry import compute_signal_geometry
    try:
        sig_geom = compute_signal_geometry(
            str(signal_vector_path), str(state_vector_path),
            str(data_dir / 'signal_geometry.parquet')
        )
        results['signal_geometry'] = len(sig_geom)
    except Exception as e:
        print(f"  Warning: {e}")
        results['signal_geometry'] = 0

    # Signal pairwise
    print(f"[SIGNAL PAIRWISE]")
    from prism.engines.signal_pairwise import compute_signal_pairwise
    try:
        sig_pair = compute_signal_pairwise(
            str(signal_vector_path), str(state_vector_path),
            str(data_dir / 'signal_pairwise.parquet')
        )
        results['signal_pairwise'] = len(sig_pair)
    except Exception as e:
        print(f"  Warning: {e}")
        results['signal_pairwise'] = 0

    return results


def run_geometry_dynamics(data_dir: Path) -> dict:
    """Run geometry dynamics pipeline."""
    from prism.engines.geometry_dynamics import compute_all_dynamics

    state_geometry_path = data_dir / 'state_geometry.parquet'
    signal_geometry_path = data_dir / 'signal_geometry.parquet'
    signal_pairwise_path = data_dir / 'signal_pairwise.parquet'

    results = {}

    print(f"[GEOMETRY DYNAMICS]")
    try:
        dynamics_results = compute_all_dynamics(
            str(state_geometry_path),
            str(signal_geometry_path),
            str(signal_pairwise_path) if signal_pairwise_path.exists() else None,
            str(data_dir),
            verbose=True
        )
        results['geometry_dynamics'] = len(dynamics_results.get('geometry', []))
        results['signal_dynamics'] = len(dynamics_results.get('signal', []))
        results['pairwise_dynamics'] = len(dynamics_results.get('pairwise', []))
    except Exception as e:
        print(f"  Warning: {e}")
        results['geometry_dynamics'] = 0
        results['signal_dynamics'] = 0
        results['pairwise_dynamics'] = 0

    return results


def run_lyapunov(data_dir: Path) -> dict:
    """Run Lyapunov engine."""
    from prism.engines.lyapunov_engine import compute_lyapunov_for_signal_vector

    signal_vector_path = data_dir / 'signal_vector.parquet'
    observations_path = data_dir / 'observations.parquet'
    output_path = data_dir / 'lyapunov.parquet'

    print(f"[LYAPUNOV]")
    try:
        result = compute_lyapunov_for_signal_vector(
            str(signal_vector_path),
            str(observations_path),
            str(output_path),
            verbose=True,
        )
        return {'lyapunov': output_path, 'rows': len(result)}
    except Exception as e:
        print(f"  Warning: {e}")
        return {'lyapunov': None, 'rows': 0}


def run_dynamics(data_dir: Path) -> dict:
    """Run dynamics engines."""
    import polars as pl
    from prism.engines.dynamics_runner import run_dynamics
    from prism.engines.information_flow_runner import run_information_flow

    obs = pl.read_parquet(data_dir / 'observations.parquet')

    results = {}

    print(f"[DYNAMICS]")
    try:
        dyn = run_dynamics(obs, data_dir)
        results['dynamics'] = len(dyn) if len(dyn) > 0 else 0
    except Exception as e:
        print(f"  Warning: {e}")
        results['dynamics'] = 0

    print(f"[INFORMATION FLOW]")
    try:
        info = run_information_flow(obs, data_dir)
        results['information_flow'] = len(info) if len(info) > 0 else 0
    except Exception as e:
        print(f"  Warning: {e}")
        results['information_flow'] = 0

    return results


def run_sql(data_dir: Path) -> dict:
    """Run SQL engines (no classification - that's ORTHON's job)."""
    from prism.sql_runner import SQLRunner

    obs_path = data_dir / 'observations.parquet'

    print(f"[SQL ENGINES]")
    runner = SQLRunner(obs_path, data_dir, engines=['zscore', 'statistics', 'correlation'])
    result = runner.run()

    return result


def run_full_pipeline(data_dir: Path) -> dict:
    """Run full PRISM pipeline (requires typology.parquet from ORTHON)."""
    print("=" * 70)
    print("PRISM PIPELINE")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()

    # Check typology exists (created by ORTHON)
    if not check_typology(data_dir):
        return {'error': 'typology.parquet not found'}

    print("[TYPOLOGY] ✓ Found (created by ORTHON)")

    results = {}

    # 1. Signal Vector (temporal - includes I column)
    results['signal_vector'] = run_signal_vector(data_dir, temporal=True)

    # 2. State Vector
    results['state_vector'] = run_state_vector(data_dir)

    # 3. Geometry (state_geometry + signal_geometry + signal_pairwise)
    results['geometry'] = run_geometry(data_dir)

    # 4. Geometry Dynamics (geometry_dynamics + signal_dynamics + pairwise_dynamics)
    results['geometry_dynamics'] = run_geometry_dynamics(data_dir)

    # 5. Dynamics (Lyapunov, RQA, etc.)
    results['dynamics'] = run_dynamics(data_dir)

    # 6. SQL (no classification)
    results['sql'] = run_sql(data_dir)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return results


def main():
    """PRISM CLI entry point."""

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg1 = sys.argv[1]

    # Command mode
    commands = {
        'signal-vector': lambda d: run_signal_vector(d, temporal=False),
        'signal-vector-temporal': lambda d: run_signal_vector(d, temporal=True),
        'state-vector': lambda d: run_state_vector(d),
        'geometry': lambda d: run_geometry(d),
        'geometry-dynamics': lambda d: run_geometry_dynamics(d),
        'lyapunov': lambda d: run_lyapunov(d),
        'dynamics': lambda d: run_dynamics(d),
        'sql': lambda d: run_sql(d),
    }

    if arg1 in commands:
        if len(sys.argv) < 3:
            print(f"Usage: python -m prism {arg1} <data_dir>")
            sys.exit(1)
        data_dir = Path(sys.argv[2])
        result = commands[arg1](data_dir)
        print(f"Result: {result}")
        return 0

    # Default: full pipeline on data_dir
    data_dir = Path(arg1)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    result = run_full_pipeline(data_dir)
    print(f"\nResults: {result}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
PRISM Manifest Runner (Orchestrator)

Reads manifest from Orthon and coordinates Python and SQL runners.

DETERMINISTIC EXECUTION: Engines either RUN or DON'T EXIST in the manifest.
No runtime file checks. No conditional skipping.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import polars as pl

from prism.python_runner import PythonRunner, SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
from prism.sql_runner import SQLRunner, SQL_ENGINES


class ManifestRunner:
    """
    Orchestrates manifest execution.

    Coordinates:
    1. PythonRunner for python/ and python_windowed/ engines
    2. SQLRunner for sql/ engines
    3. Dynamics, Topology, Information Flow, Physics engines

    All engines read from observations.parquet. No cross-dependencies.
    """

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.observations_path = Path(manifest['observations_path'])
        self.output_dir = Path(manifest['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.engine_config = manifest.get('engines', {})
        self.params = manifest.get('params', {})

        # Load observations once, share with runners
        print(f"Loading observations from {self.observations_path}")
        self.obs_pd = pd.read_parquet(self.observations_path)
        self.obs_pl = pl.from_pandas(self.obs_pd)
        print(f"  Loaded {len(self.obs_pd):,} observations")

        # Check sampling uniformity (for logging)
        self._check_sampling()

        # Results collection
        self.results: Dict[str, Any] = {}

    def _check_sampling(self):
        """Check and report on sampling uniformity."""
        try:
            I_values = self.obs_pl.select('I').unique().sort('I').to_series().to_numpy()
            if len(I_values) > 1:
                dI = np.diff(I_values)
                cv = np.std(dI) / (np.mean(dI) + 1e-10)
                if cv > 0.1:
                    print(f"  Note: Non-uniform sampling detected (CV={cv:.2f})")
                    print(f"  Dynamical results assume uniform sampling - interpret with caution")
        except Exception:
            pass

    def run(self) -> dict:
        """Execute the manifest - deterministic, no conditionals."""
        print("=" * 60)
        print("PRISM MANIFEST RUNNER")
        print("=" * 60)
        print(f"Input:  {self.observations_path}")
        print(f"Output: {self.output_dir}")

        # 1. Run Python signal/pair engines
        python_engines = {
            'signal': self.engine_config.get('signal', []),
            'pair': self.engine_config.get('pair', []),
            'symmetric_pair': self.engine_config.get('symmetric_pair', []),
            'windowed': self.engine_config.get('windowed', []),
        }

        if any(python_engines.values()):
            print("\n" + "-" * 60)
            print("PYTHON RUNNER")
            print("-" * 60)
            python_runner = PythonRunner(
                obs=self.obs_pd,
                output_dir=self.output_dir,
                engines=python_engines,
                params=self.params
            )
            python_results = python_runner.run()
            self.results['python'] = python_results

        # 2. Run SQL engines
        sql_engines = self.engine_config.get('sql', [])

        if sql_engines:
            print("\n" + "-" * 60)
            print("SQL RUNNER")
            print("-" * 60)
            sql_runner = SQLRunner(
                observations_path=self.observations_path,
                output_dir=self.output_dir,
                engines=sql_engines,
                params=self.params
            )
            sql_results = sql_runner.run()
            self.results['sql'] = sql_results

        # 3. Run dynamics engine (Lyapunov, RQA, attractors)
        if self.engine_config.get('dynamics', False):
            self._run_dynamics()

        # 4. Run topology engine (persistent homology)
        if self.engine_config.get('topology', False):
            self._run_topology()

        # 5. Run information flow engine (transfer entropy, Granger)
        if self.engine_config.get('information_flow', False):
            self._run_information_flow()

        # 6. Run physics engine (state distance, coherence, energy)
        if self.engine_config.get('physics', False):
            self._run_physics()

        # Summary
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        self._print_summary()

        return {
            'status': 'complete',
            'output_dir': str(self.output_dir),
            'results': self.results
        }

    def _run_dynamics(self):
        """Compute dynamics (Lyapunov, RQA, attractors) for all entities."""
        print("\n" + "-" * 60)
        print("DYNAMICS ENGINE")
        print("-" * 60)

        try:
            from prism.engines.dynamics_runner import run_dynamics

            dynamics_params = self.params.get('dynamics', {})
            dynamics_df = run_dynamics(self.obs_pl, self.output_dir, dynamics_params)

            if not dynamics_df.is_empty():
                self.results['dynamics'] = {'rows': len(dynamics_df), 'cols': len(dynamics_df.columns)}
            else:
                self.results['dynamics'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in dynamics engine: {e}")
            self.results['dynamics'] = {'error': str(e)}

    def _run_topology(self):
        """Compute topology (persistent homology, Betti numbers) for all entities."""
        print("\n" + "-" * 60)
        print("TOPOLOGY ENGINE")
        print("-" * 60)

        try:
            from prism.engines.topology_runner import run_topology

            topology_params = self.params.get('topology', {})
            topology_df = run_topology(self.obs_pl, self.output_dir, topology_params)

            if not topology_df.is_empty():
                self.results['topology'] = {'rows': len(topology_df), 'cols': len(topology_df.columns)}
            else:
                self.results['topology'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in topology engine: {e}")
            self.results['topology'] = {'error': str(e)}

    def _run_information_flow(self):
        """Compute information flow (transfer entropy, Granger) for all entities."""
        print("\n" + "-" * 60)
        print("INFORMATION FLOW ENGINE")
        print("-" * 60)

        try:
            from prism.engines.information_flow_runner import run_information_flow

            info_params = self.params.get('information_flow', {})
            info_df = run_information_flow(self.obs_pl, self.output_dir, info_params)

            if not info_df.is_empty():
                self.results['information_flow'] = {'rows': len(info_df), 'cols': len(info_df.columns)}
            else:
                self.results['information_flow'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in information flow engine: {e}")
            self.results['information_flow'] = {'error': str(e)}

    def _run_physics(self):
        """Compute physics stack (state distance, coherence, energy) for all entities."""
        print("\n" + "-" * 60)
        print("PHYSICS STACK")
        print("-" * 60)

        # Physics engine needs rolling metrics from observations_enriched
        # If not available, compute directly from observations with limited metrics
        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        try:
            from prism.engines.signal.physics_stack import compute_physics_for_all_entities

            physics_params = self.params.get('physics', {})

            if obs_enriched_path.exists():
                # Use enriched observations (has rolling metrics)
                obs_enriched = pd.read_parquet(obs_enriched_path)
                print(f"  Using observations_enriched.parquet ({len(obs_enriched):,} rows)")
            else:
                # Use raw observations - physics will compute what it can
                print("  Using raw observations (no rolling metrics available)")
                obs_enriched = self.obs_pd.copy()
                # Add 'y' column expected by physics (alias for value)
                if 'y' not in obs_enriched.columns and 'value' in obs_enriched.columns:
                    obs_enriched['y'] = obs_enriched['value']

            physics_df = compute_physics_for_all_entities(
                obs_enriched=obs_enriched,
                n_baseline=physics_params.get('n_baseline', 100),
                coherence_window=physics_params.get('coherence_window', 50),
            )

            if not physics_df.empty:
                output_path = self.output_dir / 'physics.parquet'
                physics_df.to_parquet(output_path, index=False)
                print(f"  physics.parquet: {len(physics_df):,} rows x {len(physics_df.columns)} cols")
                self.results['physics'] = {'rows': len(physics_df), 'cols': len(physics_df.columns)}
            else:
                print("  Warning: no physics data computed")
                self.results['physics'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in physics engine: {e}")
            self.results['physics'] = {'error': str(e)}

    def _print_summary(self):
        """Print summary of outputs."""
        print("\nOutput files:")
        for f in sorted(self.output_dir.glob('*.parquet')):
            size = f.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name}: {size_str}")


def run_manifest(manifest_path: str) -> dict:
    """Load and run a manifest from a JSON file."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    runner = ManifestRunner(manifest)
    return runner.run()


def run_manifest_dict(manifest: dict) -> dict:
    """Run a manifest from a dictionary."""
    runner = ManifestRunner(manifest)
    return runner.run()


# Export engine lists for CLI
__all__ = [
    'ManifestRunner',
    'run_manifest',
    'run_manifest_dict',
    'SIGNAL_ENGINES',
    'PAIR_ENGINES',
    'SYMMETRIC_PAIR_ENGINES',
    'WINDOWED_ENGINES',
    'SQL_ENGINES',
]

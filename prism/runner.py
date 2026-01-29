"""
PRISM Manifest Runner (Orchestrator)

Reads manifest from Orthon and coordinates Python and SQL runners.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from prism.python_runner import PythonRunner, SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
from prism.sql_runner import SQLRunner, SQL_ENGINES


class ManifestRunner:
    """
    Orchestrates manifest execution.

    Coordinates:
    1. PythonRunner for python/ and python_windowed/ engines
    2. SQLRunner for sql/ engines
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
        self.obs = pd.read_parquet(self.observations_path)
        print(f"  Loaded {len(self.obs):,} observations")

        # Results collection
        self.results: Dict[str, Any] = {}

    def run(self) -> dict:
        """Execute the manifest."""
        print("=" * 60)
        print("PRISM MANIFEST RUNNER")
        print("=" * 60)
        print(f"Input:  {self.observations_path}")
        print(f"Output: {self.output_dir}")

        # Run Python engines
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
                obs=self.obs,
                output_dir=self.output_dir,
                engines=python_engines,
                params=self.params
            )
            python_results = python_runner.run()
            self.results['python'] = python_results

        # Run SQL engines
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

        # Run physics engine (requires observations_enriched)
        if self.engine_config.get('physics', False):
            self._run_physics()

        # Run dynamics engine (requires observations_enriched)
        if self.engine_config.get('dynamics', False):
            self._run_dynamics()

        # Run topology engine (requires observations_enriched)
        if self.engine_config.get('topology', False):
            self._run_topology()

        # Run information flow engine (requires observations_enriched)
        if self.engine_config.get('information_flow', False):
            self._run_information_flow()

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

    def _run_physics(self):
        """Compute physics stack for all entities."""
        print("\n" + "-" * 60)
        print("PHYSICS STACK")
        print("-" * 60)

        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        if not obs_enriched_path.exists():
            print("  Skipping: observations_enriched.parquet not found")
            return

        from prism.engines.python.physics import compute_physics_for_all_entities

        obs_enriched = pd.read_parquet(obs_enriched_path)
        physics_params = self.params.get('physics', {})

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

    def _run_dynamics(self):
        """Compute dynamics (Lyapunov, RQA) for all entities."""
        print("\n" + "-" * 60)
        print("DYNAMICS ENGINE")
        print("-" * 60)

        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        if not obs_enriched_path.exists():
            print("  Skipping: observations_enriched.parquet not found")
            return

        from prism.dynamics.engine import compute_dynamics_for_all_entities

        obs_enriched = pd.read_parquet(obs_enriched_path)
        dynamics_params = self.params.get('dynamics', {})

        dynamics_df = compute_dynamics_for_all_entities(
            obs_enriched=obs_enriched,
            window_size=dynamics_params.get('window_size', 100),
            step_size=dynamics_params.get('step_size', 10),
        )

        if not dynamics_df.empty:
            output_path = self.output_dir / 'dynamics.parquet'
            dynamics_df.to_parquet(output_path, index=False)
            print(f"  dynamics.parquet: {len(dynamics_df):,} rows x {len(dynamics_df.columns)} cols")
            self.results['dynamics'] = {'rows': len(dynamics_df), 'cols': len(dynamics_df.columns)}
        else:
            print("  Warning: no dynamics data computed")

    def _run_topology(self):
        """Compute topology (persistent homology, Betti numbers) for all entities."""
        print("\n" + "-" * 60)
        print("TOPOLOGY ENGINE")
        print("-" * 60)

        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        if not obs_enriched_path.exists():
            print("  Skipping: observations_enriched.parquet not found")
            return

        from prism.topology.engine import compute_topology_for_all_entities

        obs_enriched = pd.read_parquet(obs_enriched_path)
        topology_params = self.params.get('topology', {})

        topology_df = compute_topology_for_all_entities(
            obs_enriched=obs_enriched,
            window_size=topology_params.get('window_size', 100),
            step_size=topology_params.get('step_size', 20),
        )

        if not topology_df.empty:
            output_path = self.output_dir / 'topology.parquet'
            topology_df.to_parquet(output_path, index=False)
            print(f"  topology.parquet: {len(topology_df):,} rows x {len(topology_df.columns)} cols")
            self.results['topology'] = {'rows': len(topology_df), 'cols': len(topology_df.columns)}
        else:
            print("  Warning: no topology data computed")

    def _run_information_flow(self):
        """Compute information flow (transfer entropy, causality) for all entities."""
        print("\n" + "-" * 60)
        print("INFORMATION FLOW ENGINE")
        print("-" * 60)

        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        if not obs_enriched_path.exists():
            print("  Skipping: observations_enriched.parquet not found")
            return

        from prism.information.engine import compute_information_flow_for_all_entities

        obs_enriched = pd.read_parquet(obs_enriched_path)
        info_params = self.params.get('information_flow', {})

        info_df = compute_information_flow_for_all_entities(
            obs_enriched=obs_enriched,
            window_size=info_params.get('window_size', 100),
            step_size=info_params.get('step_size', 20),
        )

        if not info_df.empty:
            output_path = self.output_dir / 'information_flow.parquet'
            info_df.to_parquet(output_path, index=False)
            print(f"  information_flow.parquet: {len(info_df):,} rows x {len(info_df.columns)} cols")
            self.results['information_flow'] = {'rows': len(info_df), 'cols': len(info_df.columns)}
        else:
            print("  Warning: no information flow data computed")

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

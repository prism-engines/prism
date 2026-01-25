#!/usr/bin/env python3
"""
PRISM State Entry Point
=======================

Orchestrates state space and temporal dynamics calculations.

REQUIRES: vector.parquet, geometry.parquet, dynamics.parquet

Orchestrates state engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/state/.

Engines (11 total):
    Causality:
        - transfer_entropy: Information transfer between signals
        - granger: Granger causality relationships
        - coupled_inertia: Momentum coupling strength

    Dynamics:
        - tension_dynamics: Spring-like tension in feature space
        - energy_dynamics: Energy flow metrics
        - trajectory: State space trajectory analysis

    Pairwise:
        - cointegration: Long-run equilibrium relationships
        - cross_correlation: Lagged correlations
        - dtw: Dynamic time warping distance
        - dmd: Dynamic Mode Decomposition

    Aggregation:
        - cohort: Fleet-level cohort behavior

Output:
    data/state.parquet - entity indexed

Usage:
    python -m prism.entry_points.state
    python -m prism.entry_points.state --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS - All compute lives in engines
# =============================================================================

# Class-based engines
from prism.engines.state.transfer_entropy import TransferEntropyEngine
from prism.engines.state.granger import GrangerEngine
from prism.engines.state.coupled_inertia import CoupledInertiaEngine
from prism.engines.state.tension_dynamics import TensionDynamicsEngine
from prism.engines.state.energy_dynamics import EnergyDynamicsEngine
from prism.engines.state.cointegration import CointegrationEngine
from prism.engines.state.cross_correlation import CrossCorrelationEngine
from prism.engines.state.dtw import DTWEngine
from prism.engines.state.dmd import DMDEngine

# Function-based engines
from prism.engines.state.trajectory import compute_state_trajectory, compute_state_metrics
from prism.engines.state.cohort import run_cohort_state


# Engine registry - class-based
CLASS_ENGINES = {
    'transfer_entropy': TransferEntropyEngine,
    'granger': GrangerEngine,
    'coupled_inertia': CoupledInertiaEngine,
    'tension_dynamics': TensionDynamicsEngine,
    'energy_dynamics': EnergyDynamicsEngine,
    'cointegration': CointegrationEngine,
    'cross_correlation': CrossCorrelationEngine,
    'dtw': DTWEngine,
    'dmd': DMDEngine,
}

# Engine registry - function-based
FUNCTION_ENGINES = {
    'trajectory': compute_state_trajectory,
    'cohort': run_cohort_state,
}


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config.json or config.yaml from data directory."""
    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    yaml_path = data_path / 'config.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {}


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def build_feature_matrix(
    vector_df: pl.DataFrame,
    entity_id: str,
) -> tuple:
    """Build feature matrix for one entity."""
    entity_data = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_data) == 0:
        return None, None

    # Identify metric columns
    id_cols = {'entity_id', 'signal_id', 'window_idx', 'window_start', 'window_end', 'n_samples'}
    metric_cols = [c for c in entity_data.columns
                   if c not in id_cols
                   and entity_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not metric_cols:
        return None, None

    # Convert to pandas for engine compatibility
    pdf = entity_data.select(metric_cols).to_pandas()
    pdf = pdf.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return pdf, metric_cols


# =============================================================================
# ORCHESTRATOR - Routes to engines, no compute
# =============================================================================

def run_state_engines(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
    entity_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all state engines for one entity.

    Pure orchestration - all compute is in engines.
    """
    results = {'entity_id': entity_id}

    # Get engine config
    state_config = config.get('state', {})
    enabled_engines = state_config.get('enabled', list(CLASS_ENGINES.keys()) + list(FUNCTION_ENGINES.keys()))
    engine_params = state_config.get('params', {})

    # Build feature matrix
    feature_df, metric_cols = build_feature_matrix(vector_df, entity_id)
    if feature_df is None or len(feature_df) < 3:
        return results

    run_id = f"{entity_id}_state"

    # === CLASS-BASED ENGINES ===
    for engine_name, EngineClass in CLASS_ENGINES.items():
        if engine_name not in enabled_engines:
            continue

        try:
            engine = EngineClass()
            params = engine_params.get(engine_name, {})
            result = engine.run(df=feature_df, run_id=run_id, **params)

            # Flatten result into row
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if np.isfinite(v):
                            results[f"{engine_name}_{k}"] = float(v)
                    elif isinstance(v, str):
                        results[f"{engine_name}_{k}"] = v
                    elif isinstance(v, np.ndarray) and v.size <= 50:
                        results[f"{engine_name}_{k}_json"] = json.dumps(v.tolist())

        except Exception as e:
            logger.debug(f"{engine_name} failed for {entity_id}: {e}")

    # === FUNCTION-BASED ENGINES ===
    feature_matrix = feature_df.values

    for engine_name, compute_fn in FUNCTION_ENGINES.items():
        if engine_name not in enabled_engines:
            continue

        try:
            params = engine_params.get(engine_name, {})

            if engine_name == 'trajectory':
                # Trajectory analysis
                result = compute_fn(feature_matrix, **params)
                if hasattr(result, '__dict__'):
                    # StateTrajectory object - extract metrics
                    result = compute_state_metrics(result)
            elif engine_name == 'cohort':
                # Cohort analysis needs full dataframes
                result = compute_fn(
                    vector_df=vector_df.filter(pl.col('entity_id') == entity_id),
                    geometry_df=geometry_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in geometry_df.columns else geometry_df,
                    dynamics_df=dynamics_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in dynamics_df.columns else dynamics_df,
                    config=config,
                )
            else:
                result = compute_fn(feature_matrix, **params)

            # Flatten result
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if np.isfinite(v):
                            results[f"{engine_name}_{k}"] = float(v)
                    elif isinstance(v, str):
                        results[f"{engine_name}_{k}"] = v

        except Exception as e:
            logger.debug(f"{engine_name} failed for {entity_id}: {e}")

    return results


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_state(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute state metrics for all entities.

    Output: ONE ROW PER ENTITY
    """
    entities = vector_df['entity_id'].unique().to_list() if 'entity_id' in vector_df.columns else ['default']
    n_entities = len(entities)

    logger.info(f"Computing state for {n_entities} entities")
    logger.info(f"Class engines: {list(CLASS_ENGINES.keys())}")
    logger.info(f"Function engines: {list(FUNCTION_ENGINES.keys())}")

    results = []

    for i, entity_id in enumerate(entities):
        row = run_state_engines(vector_df, geometry_df, dynamics_df, entity_id, config)

        if len(row) > 1:  # More than just entity_id
            results.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for state")
        return pl.DataFrame({'entity_id': []})

    df = pl.DataFrame(results)
    logger.info(f"State: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM State")
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--engines', '-e', nargs='+',
                        choices=list(CLASS_ENGINES.keys()) + list(FUNCTION_ENGINES.keys()))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM State")
    logger.info("=" * 60)

    ensure_directory()
    data_path = Path(get_path(VECTOR)).parent

    # Check dependencies
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)
    dynamics_path = get_path(DYNAMICS)

    missing = []
    if not vector_path.exists():
        missing.append('vector')
    if not geometry_path.exists():
        missing.append('geometry')
    if not dynamics_path.exists():
        missing.append('dynamics')

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Run the pipeline in order: vector → geometry → dynamics → state")
        return 1

    output_path = data_path / 'state.parquet'
    if output_path.exists() and not args.force:
        logger.info("state.parquet exists, use --force to recompute")
        return 0

    # Load dependencies
    vector_df = read_parquet(vector_path)
    geometry_df = read_parquet(geometry_path)
    dynamics_df = read_parquet(dynamics_path)
    config = load_config(data_path)

    # Override enabled engines if specified
    if args.engines:
        if 'state' not in config:
            config['state'] = {}
        config['state']['enabled'] = args.engines

    logger.info(f"Vector: {len(vector_df)} rows")
    logger.info(f"Geometry: {len(geometry_df)} rows")
    logger.info(f"Dynamics: {len(dynamics_df)} rows")

    # Run
    start = time.time()
    state_df = compute_state(vector_df, geometry_df, dynamics_df, config)

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(state_df)} rows, {len(state_df.columns)} columns")

    # Save
    write_parquet_atomic(state_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

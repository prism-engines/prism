#!/usr/bin/env python3
"""
PRISM Physics Entry Point
=========================

Computes physics-inspired metrics using ALL physics engines.

Engines (8 total):
    - hamiltonian: Total energy (H = T + V), conservation detection
    - lagrangian: Action principle (L = T - V)
    - kinetic_energy: Energy of motion (T = 1/2 mv^2)
    - potential_energy: Energy of position (V = 1/2 k(x-x0)^2)
    - gibbs_free_energy: Spontaneity and equilibrium (G = H - TS)
    - angular_momentum: Cyclical dynamics (L = q x p)
    - momentum_flux: Flow dynamics (Navier-Stokes inspired)
    - derivatives: Velocity, acceleration, jerk analysis

Key insight:
    The Hamiltonian is the canary - when energy stops being conserved,
    something fundamental has changed in the system.

Usage:
    python -m prism.entry_points.physics
    python -m prism.entry_points.physics --force

Output:
    data/physics.parquet - Physics metrics per (entity, signal, window)
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS
# =============================================================================

def import_engines(config: Dict[str, Any]):
    """
    Import physics engines based on config selection.

    Config structure:
        engines:
          physics:
            hamiltonian: true
            lagrangian: false
            ...
    """
    engines = {}
    engine_config = config.get('engines', {}).get('physics', {})

    # If no config, enable all engines by default
    if not engine_config:
        engine_config = {k: True for k in [
            'hamiltonian', 'lagrangian', 'kinetic', 'potential',
            'gibbs', 'angular_momentum', 'momentum_flux', 'derivatives',
        ]}

    if engine_config.get('hamiltonian', True):
        try:
            from prism.engines.physics import compute_hamiltonian
            engines['hamiltonian'] = compute_hamiltonian
        except ImportError:
            pass

    if engine_config.get('lagrangian', True):
        try:
            from prism.engines.physics import compute_lagrangian
            engines['lagrangian'] = compute_lagrangian
        except ImportError:
            pass

    if engine_config.get('kinetic', True):
        try:
            from prism.engines.physics import compute_kinetic
            engines['kinetic'] = compute_kinetic
        except ImportError:
            pass

    if engine_config.get('potential', True):
        try:
            from prism.engines.physics import compute_potential
            engines['potential'] = compute_potential
        except ImportError:
            pass

    if engine_config.get('gibbs', True):
        try:
            from prism.engines.physics import compute_gibbs
            engines['gibbs'] = compute_gibbs
        except ImportError:
            pass

    if engine_config.get('angular_momentum', True):
        try:
            from prism.engines.physics import compute_angular_momentum
            engines['angular_momentum'] = compute_angular_momentum
        except ImportError:
            pass

    if engine_config.get('momentum_flux', True):
        try:
            from prism.engines.physics import compute_momentum_flux
            engines['momentum_flux'] = compute_momentum_flux
        except ImportError:
            pass

    if engine_config.get('derivatives', True):
        try:
            from prism.engines.physics import compute_derivatives
            engines['derivatives'] = compute_derivatives
        except ImportError:
            pass

    logger.info(f"  Loaded {len(engines)} physics engines from config")
    return engines


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples': 50,
    'window_size': None,
    'stride': None,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)

    return config


# =============================================================================
# RESULT FLATTENING
# =============================================================================

def flatten_result(result: Any, prefix: str) -> Dict[str, float]:
    """Flatten engine result (dataclass or dict) to flat float dict."""
    flat = {}

    if result is None:
        return flat

    if hasattr(result, '__dataclass_fields__'):
        # Dataclass result
        for k in result.__dataclass_fields__:
            v = getattr(result, k)
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif isinstance(v, bool):
                flat[f"{prefix}_{k}"] = 1.0 if v else 0.0
            elif isinstance(v, str):
                # Skip string fields
                pass
    elif isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif isinstance(v, np.ndarray) and v.size == 1:
                val = float(v.item())
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val

    return flat


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_physics(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute physics metrics for all signals.
    """
    min_samples = config.get('min_samples', 50)

    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    n_signals = len(signals)
    n_engines = len(engines)
    logger.info(f"Processing {n_signals} signals with {n_engines} engines")

    results = []

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        timestamps = np.array(row['timestamps'], dtype=float)

        # Sort and clean
        sort_idx = np.argsort(timestamps)
        values = values[sort_idx]
        timestamps = timestamps[sort_idx]

        valid = ~np.isnan(values)
        values = values[valid]
        timestamps = timestamps[valid]

        if len(values) < min_samples:
            continue

        row_data = {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_samples': len(values),
            'window_start': float(timestamps[0]),
            'window_end': float(timestamps[-1]),
        }

        # Run all physics engines
        for engine_name, compute_fn in engines.items():
            try:
                result = compute_fn(values)
                flat = flatten_result(result, engine_name)
                row_data.update(flat)
            except Exception as e:
                logger.debug(f"  {engine_name} failed: {e}")

        results.append(row_data)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Physics: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Physics - Energy and momentum computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Physics Engine")
    logger.info("=" * 60)

    ensure_directory()

    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("observations.parquet not found")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'physics.parquet'

    if output_path.exists() and not args.force:
        logger.info("physics.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    logger.info("Loading engines from config...")
    engines = import_engines(config)
    logger.info(f"Total engines: {len(engines)}")

    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    start = time.time()
    df = compute_physics(observations, config, engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())

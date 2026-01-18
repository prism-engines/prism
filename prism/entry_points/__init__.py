"""
PRISM Entry Points Registry
===========================

CLI entry points for the PRISM analysis pipeline.
Each entry point has a defined goal, inputs, outputs, and documentation.

Storage: Polars + Parquet (data/ directory)

Pipeline Architecture:

    CORE PIPELINE:
        fetch → characterize → signal_vector → laplace → geometry → state

    DYNAMICAL SYSTEMS:
        generate_dynamical → dynamic_vector → dynamic_state → physics

Usage:
    # Core pipeline
    python -m prism.entry_points.fetch --cmapss
    python -m prism.entry_points.characterize
    python -m prism.entry_points.signal_vector
    python -m prism.entry_points.laplace
    python -m prism.entry_points.geometry
    python -m prism.entry_points.state

    # Dynamical systems validation
    python -m prism.entry_points.generate_dynamical --system lorenz
    python -m prism.entry_points.dynamic_vector
    python -m prism.entry_points.physics
"""

from typing import Dict, Any, Optional

# =============================================================================
# ENTRY POINT REGISTRY
# =============================================================================

ENTRY_POINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # CORE PIPELINE
    # ==========================================================================
    'fetch': {
        'module': 'prism.entry_points.fetch',
        'goal': 'Fetch data from external sources (USGS, climate, C-MAPSS, etc.)',
        'inputs': ['APIs', 'fetchers/yaml/*.yaml'],
        'outputs': ['raw/observations.parquet'],
    },

    'characterize': {
        'module': 'prism.entry_points.characterize',
        'goal': 'Characterize each signal (6 axes, valid engines, return method)',
        'inputs': ['raw/observations.parquet'],
        'outputs': ['raw/characterization.parquet'],
    },

    'signal_vector': {
        'module': 'prism.entry_points.signal_vector',
        'goal': 'Compute vector metrics for each signal (51 metrics: entropy, hurst, lyapunov, RQA, etc.)',
        'inputs': ['raw/observations.parquet', 'raw/characterization.parquet'],
        'outputs': ['vector/signal.parquet'],
    },

    'laplace': {
        'module': 'prism.entry_points.laplace',
        'goal': 'Compute Laplace field (gradient, laplacian, divergence) on signal vectors',
        'inputs': ['vector/signal.parquet'],
        'outputs': ['vector/signal_field.parquet'],
    },

    'laplace_pairwise': {
        'module': 'prism.entry_points.laplace_pairwise',
        'goal': 'Compute pairwise geometry in Laplace field space (vectorized)',
        'inputs': ['vector/signal_field.parquet'],
        'outputs': ['geometry/laplace_pair.parquet'],
    },

    'geometry': {
        'module': 'prism.entry_points.geometry',
        'goal': 'Compute cohort geometry (PCA, MST, clustering, LOF) + modes + wavelet',
        'inputs': ['vector/signal_field.parquet'],
        'outputs': ['geometry/cohort.parquet', 'geometry/signal_pair.parquet'],
    },

    'state': {
        'module': 'prism.entry_points.state',
        'goal': 'Derive query-time state for signals and cohorts',
        'inputs': ['geometry/cohort.parquet'],
        'outputs': ['state/signal.parquet', 'state/cohort.parquet'],
    },

    # ==========================================================================
    # DYNAMICAL SYSTEMS VALIDATION
    # ==========================================================================
    'generate_dynamical': {
        'module': 'prism.entry_points.generate_dynamical',
        'goal': 'Generate test data from dynamical systems (Lorenz, Rossler, etc.)',
        'inputs': [],
        'outputs': ['raw/observations.parquet (dynamical)'],
    },

    'generate_pendulum_regime': {
        'module': 'prism.entry_points.generate_pendulum_regime',
        'goal': 'Generate double pendulum regime data for testing',
        'inputs': [],
        'outputs': ['raw/observations.parquet (pendulum)'],
    },

    'dynamic_vector': {
        'module': 'prism.entry_points.dynamic_vector',
        'goal': 'Compute vector metrics for dynamical system data',
        'inputs': ['raw/observations.parquet'],
        'outputs': ['vector/signal.parquet'],
    },

    'dynamic_state': {
        'module': 'prism.entry_points.dynamic_state',
        'goal': 'Compute state metrics for dynamical system data',
        'inputs': ['vector/signal.parquet'],
        'outputs': ['state/signal.parquet'],
    },

    'physics': {
        'module': 'prism.entry_points.physics',
        'goal': 'Validate physics laws (energy conservation, entropy increase, least action)',
        'inputs': ['state/signal.parquet'],
        'outputs': ['physics/conservation.parquet'],
    },

    # ==========================================================================
    # SUPERVISED LEARNING BRIDGE
    # ==========================================================================
    'hybrid': {
        'module': 'prism.entry_points.hybrid',
        'goal': 'Combine PRISM features with ML models for supervised prediction',
        'inputs': ['vector/signal_field.parquet', 'geometry/cohort.parquet'],
        'outputs': ['predictions (in-memory)'],
    },
}


def list_entry_points() -> None:
    """List all entry points."""
    print("=" * 70)
    print("PRISM ENTRY POINTS")
    print("=" * 70)
    for name, info in ENTRY_POINT_REGISTRY.items():
        print(f"\n{name}")
        print(f"  Goal: {info['goal']}")
        print(f"  Inputs: {info['inputs']}")
        print(f"  Outputs: {info['outputs']}")


def get_entry_point_info(name: str) -> Optional[Dict[str, Any]]:
    """Get info for a specific entry point."""
    return ENTRY_POINT_REGISTRY.get(name)


# =============================================================================
# PUBLIC API - Core Pipeline
# =============================================================================
try:
    from prism.entry_points.signal_vector import (
        UnivariateVector,
        UnivariateResult,
        compute_univariate,
        process_signal,
        process_all_signals,
        store_results,
    )
except ImportError:
    pass

try:
    from prism.entry_points.geometry import (
        load_cohort_members,
    )
except ImportError:
    pass

try:
    from prism.entry_points.laplace import (
        compute_laplace_field,
        WindowConfig,
    )
except ImportError:
    pass

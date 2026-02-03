"""
PRISM Engines
=============

Flat directory containing all compute engines.

Stage 1 - Signal Vector:
    signal/      - Per-signal engines (kurtosis, entropy, lyapunov, etc.)
    rolling/     - Rolling window engines
    sql/         - SQL-based engines

Stage 2 - State Vector:
    state_vector.py      - Centroid computation (WHERE)
    state_geometry.py    - Eigenvalue computation (SHAPE)
    signal_geometry.py   - Signal-to-centroid distances

Stage 3 - Pairwise:
    signal_pairwise.py   - Signal-to-signal relationships
    granger.py           - Granger causality
    transfer_entropy.py  - Transfer entropy
    correlation.py       - Correlation computation
    mutual_info.py       - Mutual information
    cointegration.py     - Cointegration tests

Stage 4 - Dynamics:
    geometry_dynamics.py         - Derivatives of geometry
    lyapunov_engine.py           - Lyapunov exponents
    dynamics_runner.py           - RQA, attractor reconstruction
    information_flow_runner.py   - Transfer entropy networks
"""

# Stage 2 - State Vector
from prism.engines.state_vector import compute_state_vector, compute_centroid
from prism.engines.state_geometry import compute_state_geometry, compute_eigenvalues
from prism.engines.signal_geometry import compute_signal_geometry

# Stage 3 - Pairwise
from prism.engines.signal_pairwise import compute_signal_pairwise

# Stage 4 - Dynamics
from prism.engines.geometry_dynamics import (
    compute_geometry_dynamics,
    compute_signal_dynamics,
    compute_pairwise_dynamics,
    compute_all_dynamics,
    compute_derivatives,
)
from prism.engines.lyapunov_engine import compute_lyapunov, compute_lyapunov_for_signal_vector
from prism.engines.dynamics_runner import run_dynamics, process_entity_dynamics
from prism.engines.information_flow_runner import run_information_flow, process_entity_information_flow

# Submodules
from prism.engines import signal
from prism.engines import rolling
from prism.engines import sql

__all__ = [
    # State Vector
    'compute_state_vector',
    'compute_centroid',
    'compute_state_geometry',
    'compute_eigenvalues',
    'compute_signal_geometry',
    # Pairwise
    'compute_signal_pairwise',
    # Dynamics
    'compute_geometry_dynamics',
    'compute_signal_dynamics',
    'compute_pairwise_dynamics',
    'compute_all_dynamics',
    'compute_derivatives',
    'compute_lyapunov',
    'compute_lyapunov_for_signal_vector',
    'run_dynamics',
    'process_entity_dynamics',
    'run_information_flow',
    'process_entity_information_flow',
    # Submodules
    'signal',
    'rolling',
    'sql',
]

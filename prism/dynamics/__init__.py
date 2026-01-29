"""
PRISM Dynamics Engine

Computes dynamical systems metrics for time series data:
- Lyapunov exponents (stability/chaos)
- Attractor dimension (complexity)
- Recurrence quantification (predictability)

Complements the geometric eigenvalue analysis in physics.py
"""

from .reconstruction import (
    embed_time_series,
    optimal_delay,
    optimal_embedding_dim,
)
from .lyapunov import (
    largest_lyapunov_exponent,
    lyapunov_spectrum,
)
from .dimension import (
    correlation_dimension,
    kaplan_yorke_dimension,
)
from .recurrence import (
    recurrence_matrix,
    rqa_metrics,
)
from .engine import (
    DynamicsEngine,
    compute_dynamics,
    compute_dynamics_for_entity,
)

__all__ = [
    # Reconstruction
    'embed_time_series',
    'optimal_delay',
    'optimal_embedding_dim',
    # Lyapunov
    'largest_lyapunov_exponent',
    'lyapunov_spectrum',
    # Dimension
    'correlation_dimension',
    'kaplan_yorke_dimension',
    # Recurrence
    'recurrence_matrix',
    'rqa_metrics',
    # Engine
    'DynamicsEngine',
    'compute_dynamics',
    'compute_dynamics_for_entity',
]

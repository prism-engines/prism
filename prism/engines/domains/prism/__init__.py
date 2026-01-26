"""
PRISM Domain Engines

PRISM-specific diagnostics and fleet-level analysis.

Engines by Category:

Fleet-level (cross-entity):
    - fleet_status: Aggregate counts/means across entities
    - entity_ranking: Rank entities by baseline_distance
    - leading_indicator: First entities to show divergence
    - correlated_trajectories: Entity pairs with similar trajectories
    - cohort: Cohort state analysis

Degradation Model:
    - barycenter: Behavioral center computation
    - pairwise: Pairwise signal analysis
    - energy_dynamics: Energy flow in feature space
    - tension_dynamics: Spring-like tension dynamics
"""

# Fleet-level engines
from .fleet_status import compute as compute_fleet_status
from .entity_ranking import compute as compute_entity_ranking
from .leading_indicator import compute as compute_leading_indicator
from .correlated_trajectories import compute as compute_correlated_trajectories
from .cohort import run_cohort_state

# Degradation model engines
from .barycenter import BarycenterEngine, compute_barycenter
from .pairwise import run_laplace_pairwise_vectorized as compute_pairwise
from .energy_dynamics import EnergyDynamicsEngine, compute_energy_dynamics
from .tension_dynamics import TensionDynamicsEngine, compute_tension_dynamics

__all__ = [
    # Fleet-level
    'compute_fleet_status',
    'compute_entity_ranking',
    'compute_leading_indicator',
    'compute_correlated_trajectories',
    'run_cohort_state',
    # Degradation model
    'BarycenterEngine',
    'compute_barycenter',
    'compute_pairwise',
    'EnergyDynamicsEngine',
    'compute_energy_dynamics',
    'TensionDynamicsEngine',
    'compute_tension_dynamics',
]

"""
PRISM Topology Engine

Computes topological data analysis (TDA) metrics:
- Persistent homology
- Betti numbers
- Topological complexity

Captures the SHAPE of system dynamics that geometry and dynamics cannot see.
"""

from .point_cloud import (
    time_delay_embedding,
    sliding_window_embedding,
    multivariate_point_cloud,
)
from .persistence import (
    PersistenceDiagram,
    compute_rips_persistence,
)
from .features import (
    betti_numbers,
    betti_curve,
    persistence_statistics,
    persistence_landscape,
    topological_complexity,
)
from .engine import (
    TopologyEngine,
    TopologyResult,
    compute_topology,
    compute_topology_for_all_entities,
)

__all__ = [
    # Point cloud
    'time_delay_embedding',
    'sliding_window_embedding',
    'multivariate_point_cloud',
    # Persistence
    'PersistenceDiagram',
    'compute_rips_persistence',
    # Features
    'betti_numbers',
    'betti_curve',
    'persistence_statistics',
    'persistence_landscape',
    'topological_complexity',
    # Engine
    'TopologyEngine',
    'TopologyResult',
    'compute_topology',
    'compute_topology_for_all_entities',
]

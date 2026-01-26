"""
PRISM Core Physics Engines - Classical Mechanics

Universal physics equations that apply to ANY time series
treated as physical trajectories.
"""

from prism.engines.core.physics.kinetic_energy import compute_kinetic_energy as compute_kinetic
from prism.engines.core.physics.potential_energy import compute as compute_potential
from prism.engines.core.physics.hamiltonian import compute as compute_hamilton
from prism.engines.core.physics.lagrangian import compute as compute_lagrange
from prism.engines.core.physics.momentum import compute as compute_momentum
from prism.engines.core.physics.work_energy import compute_mechanical_energy as compute_work_energy

__all__ = [
    'compute_kinetic',
    'compute_potential',
    'compute_hamilton',
    'compute_lagrange',
    'compute_momentum',
    'compute_work_energy',
]

"""
PRISM Fluid Mechanics Domain Engines

Field-specific engines for fluid mechanics analysis.
"""

from prism.engines.domains.fluid.reynolds import compute as compute_reynolds
from prism.engines.domains.fluid.pressure_drop import compute as compute_pressure_drop

__all__ = [
    'compute_reynolds',
    'compute_pressure_drop',
]

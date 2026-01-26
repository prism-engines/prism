"""
PRISM Thermodynamics Domain Engines

Field-specific engines for thermodynamic analysis.
"""

from prism.engines.domains.thermodynamics.gibbs_free_energy import (
    compute as compute_gibbs,
)

__all__ = [
    'compute_gibbs',
]

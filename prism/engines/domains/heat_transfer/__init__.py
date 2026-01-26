"""
PRISM Heat Transfer Domain Engines

Field-specific engines for heat transfer analysis.
"""

from prism.engines.domains.heat_transfer.fourier import (
    compute_heat_flux,
    compute_conduction_slab,
)

__all__ = [
    'compute_heat_flux',
    'compute_conduction_slab',
]

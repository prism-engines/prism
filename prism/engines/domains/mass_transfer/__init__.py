"""
PRISM Mass Transfer Domain Engines

Field-specific engines for mass transfer analysis (Fick's law).
"""

from prism.engines.domains.mass_transfer.fick import compute_molar_flux

__all__ = [
    'compute_molar_flux',
]

"""
PRISM Engineering Domain Engines

Dimensionless numbers and engineering calculations.
"""

from prism.engines.domains.engineering.dimensionless import (
    compute_prandtl,
    compute_schmidt,
    compute_nusselt,
    compute_sherwood,
    compute_peclet,
    compute_damkohler,
    compute_weber,
    compute_froude,
    compute_grashof,
    compute_rayleigh,
)


def compute_all_dimensionless(**kwargs):
    """Compute all dimensionless numbers."""
    results = {}
    for name, func in [
        ('prandtl', compute_prandtl),
        ('schmidt', compute_schmidt),
        ('nusselt', compute_nusselt),
        ('sherwood', compute_sherwood),
        ('peclet', compute_peclet),
        ('damkohler', compute_damkohler),
        ('weber', compute_weber),
        ('froude', compute_froude),
        ('grashof', compute_grashof),
        ('rayleigh', compute_rayleigh),
    ]:
        try:
            results[name] = func(**kwargs)
        except Exception:
            pass
    return results


__all__ = [
    'compute_prandtl',
    'compute_schmidt',
    'compute_nusselt',
    'compute_sherwood',
    'compute_peclet',
    'compute_damkohler',
    'compute_weber',
    'compute_froude',
    'compute_grashof',
    'compute_rayleigh',
    'compute_all_dimensionless',
]

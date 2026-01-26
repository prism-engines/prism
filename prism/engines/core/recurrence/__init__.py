"""
Recurrence Axis Engines
=======================

Computation engines for the Recurrence axis:
- rqa: Recurrence Quantification Analysis (DET, LAM, ENT, TT, etc.)
"""

from .rqa import compute as compute_rqa

__all__ = [
    'compute_rqa',
]

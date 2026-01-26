"""
Volatility Axis Engines
=======================

Computation engines for the Volatility axis:
- garch: GARCH(1,1) volatility model
- realized_vol: Realized volatility from squared returns
- bipower_variation: Jump-robust volatility (Barndorff-Nielsen & Shephard)
- hilbert_amplitude: Amplitude envelope via Hilbert transform
"""

from .garch import compute as compute_garch
from .realized_vol import compute as compute_realized_vol
from .bipower_variation import compute as compute_bipower_variation
from .hilbert_amplitude import compute as compute_hilbert_amplitude

__all__ = [
    'compute_garch',
    'compute_realized_vol',
    'compute_bipower_variation',
    'compute_hilbert_amplitude',
]

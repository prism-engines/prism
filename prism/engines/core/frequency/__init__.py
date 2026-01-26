"""
Frequency Axis Engines
======================

Computation engines for the Frequency axis:
- spectral: Spectral features (centroid, bandwidth, rolloff)
- wavelet: Multi-scale wavelet decomposition
"""

from .spectral import compute as compute_spectral
from .wavelet import compute as compute_wavelet

__all__ = [
    'compute_spectral',
    'compute_wavelet',
]

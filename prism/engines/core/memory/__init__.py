"""
Memory Axis Engines
===================

Computation engines for the Memory axis:
- hurst_dfa: Detrended Fluctuation Analysis
- hurst_rs: Rescaled Range Analysis
- acf_decay: Autocorrelation decay type
- spectral_slope: Power spectrum slope (β in S(f) ~ f^-β)
"""

from .hurst_dfa import compute as compute_hurst_dfa
from .hurst_rs import compute as compute_hurst_rs
from .acf_decay import compute as compute_acf_decay
from .spectral_slope import compute as compute_spectral_slope

__all__ = [
    'compute_hurst_dfa',
    'compute_hurst_rs',
    'compute_acf_decay',
    'compute_spectral_slope',
]

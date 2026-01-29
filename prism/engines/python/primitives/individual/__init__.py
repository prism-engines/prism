"""
Individual Signal Primitives

Single-signal computations: statistics, spectral, entropy, etc.
These are the atomic building blocks for all higher-level engines.
"""

from .statistics import (
    mean, std, variance, min_max, percentiles,
    skewness, kurtosis, rms, peak_to_peak, crest_factor,
    zero_crossings, mean_crossings,
)
from .calculus import (
    derivative, integral, curvature,
)
from .correlation import (
    autocorrelation, partial_autocorrelation,
)
from .spectral import (
    fft, psd, dominant_frequency,
    spectral_centroid, spectral_bandwidth, spectral_entropy,
    wavelet_coeffs,
)
from .hilbert import (
    envelope, hilbert_transform,
    instantaneous_frequency, instantaneous_amplitude,
)
from .entropy import (
    sample_entropy, permutation_entropy, approximate_entropy,
)
from .fractal import (
    hurst_exponent, dfa, hurst_r2,
)
from .stationarity import (
    stationarity_test, trend, changepoints, mann_kendall_test,
)

__all__ = [
    # Statistics (1-12)
    'mean', 'std', 'variance', 'min_max', 'percentiles',
    'skewness', 'kurtosis', 'rms', 'peak_to_peak', 'crest_factor',
    'zero_crossings', 'mean_crossings',
    # Calculus (13-14)
    'derivative', 'integral', 'curvature',
    # Correlation (15-16)
    'autocorrelation', 'partial_autocorrelation',
    # Spectral (17-23)
    'fft', 'psd', 'dominant_frequency',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_entropy',
    'wavelet_coeffs',
    # Hilbert (24-27)
    'envelope', 'hilbert_transform',
    'instantaneous_frequency', 'instantaneous_amplitude',
    # Entropy (28-30)
    'sample_entropy', 'permutation_entropy', 'approximate_entropy',
    # Fractal (31-32)
    'hurst_exponent', 'dfa', 'hurst_r2',
    # Stationarity (33-35)
    'stationarity_test', 'trend', 'changepoints', 'mann_kendall_test',
]

"""
Pairwise Signal Primitives

Two-signal computations: correlation, coherence, causality, etc.
"""

from .correlation import (
    correlation, covariance,
    cross_correlation, lag_at_max_xcorr,
    partial_correlation,
)
from .spectral import (
    coherence, cross_spectral_density, phase_spectrum,
    wavelet_coherence,
)
from .information import (
    mutual_information, transfer_entropy,
)
from .causality import (
    granger_causality, convergent_cross_mapping,
)
from .distance import (
    dynamic_time_warping, euclidean_distance, cosine_similarity, manhattan_distance,
)
from .regression import (
    linear_regression, ratio, product, difference, sum_signals,
)

__all__ = [
    # Correlation (36-40)
    'correlation', 'covariance',
    'cross_correlation', 'lag_at_max_xcorr',
    'partial_correlation',
    # Spectral (41-43)
    'coherence', 'cross_spectral_density', 'phase_spectrum',
    'wavelet_coherence',
    # Information (44-45)
    'mutual_information', 'transfer_entropy',
    # Causality (46-48)
    'granger_causality', 'convergent_cross_mapping',
    # Distance (49-51)
    'dynamic_time_warping', 'euclidean_distance', 'cosine_similarity', 'manhattan_distance',
    # Regression (52-55)
    'linear_regression', 'ratio', 'product', 'difference', 'sum_signals',
]

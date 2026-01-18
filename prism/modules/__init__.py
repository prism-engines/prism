"""
PRISM Modules Package
=====================

Reusable computation modules that can be called inline by entry points.
These are NOT standalone entry points - they're building blocks.

Modules:
    characterize: Inline characterization of signals
    laplace: Inline Laplace field computation
    modes: Behavioral mode discovery from Laplace signatures
    wavelet_microscope: Frequency-band degradation detection
    prefilter: O(n) Laplacian pre-filter for flat/duplicate signals
"""

from prism.modules.characterize import (
    characterize_signal,
    CharacterizationResult,
)

from prism.modules.laplace import (
    compute_laplace_for_series,
    compute_gradient,
    compute_laplacian,
)

from prism.modules.modes import (
    discover_modes,
    extract_laplace_fingerprint,
    extract_cohort_fingerprints,
    find_optimal_modes,
    compute_mode_scores,
    compute_affinity_weighted_features,
    compute_affinity_dynamics,
    run_modes,
)

from prism.modules.wavelet_microscope import (
    compute_wavelet_decomposition,
    compute_band_snr_evolution,
    identify_degradation_band,
    run_wavelet_microscope,
    extract_wavelet_features,
)

from prism.modules.prefilter import (
    laplacian_prefilter,
    identify_duplicates,
    smart_filter,
)

__all__ = [
    # Characterize
    'characterize_signal',
    'CharacterizationResult',
    # Laplace
    'compute_laplace_for_series',
    'compute_gradient',
    'compute_laplacian',
    # Modes
    'discover_modes',
    'extract_laplace_fingerprint',
    'extract_cohort_fingerprints',
    'find_optimal_modes',
    'compute_mode_scores',
    'compute_affinity_weighted_features',
    'compute_affinity_dynamics',
    'run_modes',
    # Wavelet Microscope
    'compute_wavelet_decomposition',
    'compute_band_snr_evolution',
    'identify_degradation_band',
    'run_wavelet_microscope',
    'extract_wavelet_features',
    # Prefilter
    'laplacian_prefilter',
    'identify_duplicates',
    'smart_filter',
]

"""
Structure Engines (PR #10)

Five engines that compose primitives into complete structural analysis:
1. CovarianceEngine - Covariance and correlation matrices
2. EigenvalueEngine - Eigenvalue decomposition with significance testing
3. KoopmanEngine - Dynamic Mode Decomposition (DMD)
4. SpectralEngine - PSD, coherence, cross-spectral density
5. WaveletEngine - Wavelet analysis and coherence
"""

from .covariance_engine import CovarianceEngine, CovarianceConfig
from .eigenvalue_engine import EigenvalueEngine, EigenvalueConfig
from .koopman_engine import KoopmanEngine, KoopmanConfig
from .spectral_engine import SpectralEngine, SpectralConfig
from .wavelet_engine import WaveletEngine, WaveletConfig
from .structure_runner import StructureRunner, StructureRunnerConfig, run_structure_analysis

__all__ = [
    # Engines
    'CovarianceEngine',
    'EigenvalueEngine',
    'KoopmanEngine',
    'SpectralEngine',
    'WaveletEngine',
    # Configs
    'CovarianceConfig',
    'EigenvalueConfig',
    'KoopmanConfig',
    'SpectralConfig',
    'WaveletConfig',
    # Runner
    'StructureRunner',
    'StructureRunnerConfig',
    'run_structure_analysis',
]

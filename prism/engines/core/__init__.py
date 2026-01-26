"""
PRISM Core Engines
==================

Universal, domain-agnostic mathematical engines.
These compute properties valid for ANY time series.

Categories:
    - memory: Long-range dependence (Hurst, ACF, DFA)
    - information: Complexity measures (entropy)
    - frequency: Spectral analysis (FFT, wavelet)
    - dynamics: Nonlinear dynamics (Lyapunov, attractors)
    - state: Pairwise relationships (Granger, DTW, cointegration)
    - geometry: Structural analysis (PCA, MST, clustering)
    - laplace: Laplace transforms and operators
    - volatility: Variance modeling (GARCH, realized vol)
    - recurrence: Recurrence quantification
    - typology: Signal classification (trend, seasonality)
    - pointwise: Point-by-point transforms
    - momentum: Randomness tests
    - detection: Spike and step detection
    - discontinuity: Structural breaks
    - events: Heaviside and Dirac functions
    - windowed: Windowed versions of above
    - physics: Classical mechanics (Hamiltonian, Lagrangian)
    - electrochemistry: Universal electrochemical laws
    - phase_equilibria: Thermodynamic equilibrium
    - balances: Energy and mass balances
    - fields: Navier-Stokes
    - utils: Parallel processing utilities
"""

# Subdirectories are imported lazily when needed
# This avoids circular imports and speeds up startup

CORE_ENGINE_CATEGORIES = [
    'memory',
    'information',
    'frequency',
    'dynamics',
    'state',
    'geometry',
    'laplace',
    'volatility',
    'recurrence',
    'typology',
    'pointwise',
    'momentum',
    'detection',
    'discontinuity',
    'events',
    'windowed',
    'physics',
    'electrochemistry',
    'phase_equilibria',
    'balances',
    'fields',
    'utils',
]

__all__ = ['CORE_ENGINE_CATEGORIES']

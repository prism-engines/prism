"""
Signal Typology Engines
=======================

Computation engines for the Signal Typology framework.

These engines compute the raw metrics that get normalized
into the 9-axis typology profile.

Engines:
    - derivative_stats: Derivative mean, std, kurtosis, zero crossing
    - cusum: CUSUM and level shift detection
    - rolling_volatility: Rolling std ratio for volatility clustering
    - stationarity: ADF test, KPSS test, unit root detection
    - trend: Mann-Kendall test, Sen's slope
    - seasonality: STL decomposition, seasonal strength
    - distribution: Skewness, kurtosis, normality tests
    - takens: Takens embedding, phase space reconstruction

Note: Runs test is in engines/momentum/runs_test.py (shared engine)
"""

from .derivative_stats import compute as compute_derivative_stats
from .cusum import compute as compute_cusum
from .rolling_volatility import compute as compute_rolling_volatility
from .stationarity import compute_stationarity, classify_stationarity
from .trend import compute_trend, classify_trend, compute_trend_change
from .seasonality import compute_seasonality, classify_seasonality, decompose_multiple_seasonalities
from .distribution import compute_distribution, classify_tail_behavior, fit_distribution
from .takens import compute_takens_embedding, compute_embedding_quality

__all__ = [
    # Original engines
    'compute_derivative_stats',
    'compute_cusum',
    'compute_rolling_volatility',
    # Stationarity
    'compute_stationarity',
    'classify_stationarity',
    # Trend
    'compute_trend',
    'classify_trend',
    'compute_trend_change',
    # Seasonality
    'compute_seasonality',
    'classify_seasonality',
    'decompose_multiple_seasonalities',
    # Distribution
    'compute_distribution',
    'classify_tail_behavior',
    'fit_distribution',
    # Takens embedding
    'compute_takens_embedding',
    'compute_embedding_quality',
]

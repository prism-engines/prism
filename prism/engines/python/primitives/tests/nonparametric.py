"""
Nonparametric Test Primitives (98)

Mann-Kendall trend test and other non-parametric tests.
"""

import numpy as np
from typing import Tuple


def mann_kendall(
    series: np.ndarray
) -> Tuple[str, float, float, float]:
    """
    Mann-Kendall trend test for monotonic trend.

    Parameters
    ----------
    series : np.ndarray
        Time series values

    Returns
    -------
    trend : str
        'increasing', 'decreasing', or 'no trend'
    p_value : float
        p-value for the trend
    tau : float
        Kendall's tau (correlation coefficient)
    slope : float
        Sen's slope (robust trend magnitude)

    Notes
    -----
    Non-parametric test for monotonic trend.
    Does NOT assume normality or linearity.

    Tests H₀: No monotonic trend

    Kendall's tau ∈ [-1, 1]:
    - τ > 0: Increasing trend
    - τ < 0: Decreasing trend
    - τ ≈ 0: No trend

    Sen's slope: Median of all pairwise slopes (robust to outliers)

    Physical interpretation:
    "Is this metric systematically increasing or decreasing over time?"

    Essential for:
    - Detecting slow degradation
    - Identifying regime drift
    - Tracking long-term changes
    """
    series = np.asarray(series).flatten()
    series = series[~np.isnan(series)]

    if len(series) < 4:
        return 'no trend', np.nan, np.nan, np.nan

    try:
        import pymannkendall as mk
        result = mk.original_test(series)
        return result.trend, float(result.p), float(result.Tau), float(result.slope)
    except ImportError:
        # Fallback implementation
        return _mann_kendall_simple(series)


def _mann_kendall_simple(series: np.ndarray) -> Tuple[str, float, float, float]:
    """Simple Mann-Kendall implementation without pymannkendall."""
    from scipy import stats

    n = len(series)

    # Compute S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    # Account for ties
    unique, counts = np.unique(series, return_counts=True)
    tp = np.sum(counts * (counts - 1) * (2 * counts + 5))

    var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18

    # Z-statistic
    if var_s == 0:
        z = 0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Sen's slope (median of pairwise slopes)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((series[j] - series[i]) / (j - i))

    sen_slope = np.median(slopes) if slopes else 0

    # Determine trend
    if p_value < 0.05:
        if tau > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no trend'

    return trend, float(p_value), float(tau), float(sen_slope)

"""
Stationarity Primitives (33-35)

Stationarity tests, trend detection, changepoints.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Tuple, List, Optional


def stationarity_test(
    signal: np.ndarray,
    test: str = 'adf'
) -> Tuple[float, float]:
    """
    Test for stationarity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    test : str
        'adf' (Augmented Dickey-Fuller) or 'kpss'

    Returns
    -------
    tuple
        (test_statistic, p_value)

    Notes
    -----
    ADF: H0 = unit root (non-stationary)
         p < 0.05 → reject H0 → stationary

    KPSS: H0 = stationary
          p < 0.05 → reject H0 → non-stationary
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]

    if len(signal) < 20:
        return np.nan, np.nan

    try:
        if test == 'adf':
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(signal, autolag='AIC')
            return float(result[0]), float(result[1])

        elif test == 'kpss':
            from statsmodels.tsa.stattools import kpss
            result = kpss(signal, regression='c')
            return float(result[0]), float(result[1])

        else:
            raise ValueError(f"Unknown test: {test}")

    except ImportError:
        # Fallback: simple variance ratio test
        n = len(signal)
        half = n // 2
        var1 = np.var(signal[:half])
        var2 = np.var(signal[half:])

        if var1 == 0 or var2 == 0:
            return np.nan, np.nan

        f_stat = var1 / var2
        # Approximate p-value
        p_value = 2 * min(
            scipy_stats.f.cdf(f_stat, half-1, n-half-1),
            1 - scipy_stats.f.cdf(f_stat, half-1, n-half-1)
        )
        return float(f_stat), float(p_value)


def trend(signal: np.ndarray) -> Tuple[float, float]:
    """
    Detect linear trend.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (slope, p_value)

    Notes
    -----
    Uses linear regression against time index.
    p < 0.05 indicates significant trend.
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 3:
        return np.nan, np.nan

    x = np.arange(n)
    result = scipy_stats.linregress(x, signal)

    return float(result.slope), float(result.pvalue)


def changepoints(
    signal: np.ndarray,
    method: str = 'pelt',
    penalty: float = None,
    n_bkps: int = None
) -> List[int]:
    """
    Detect changepoints (regime changes).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        'pelt' (optimal), 'binseg' (binary segmentation), or 'window'
    penalty : float, optional
        Penalty for adding breakpoints (for 'pelt')
    n_bkps : int, optional
        Number of breakpoints (for 'binseg')

    Returns
    -------
    list
        Indices of changepoints

    Notes
    -----
    Changepoints indicate structural breaks in the signal.
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return []

    try:
        import ruptures as rpt

        if method == 'pelt':
            algo = rpt.Pelt(model='rbf')
            if penalty is None:
                penalty = np.log(n) * np.var(signal)
            bkps = algo.fit(signal).predict(pen=penalty)

        elif method == 'binseg':
            algo = rpt.Binseg(model='l2')
            if n_bkps is None:
                n_bkps = max(1, n // 100)
            bkps = algo.fit(signal).predict(n_bkps=n_bkps)

        else:  # window
            algo = rpt.Window(width=max(10, n // 20), model='l2')
            if n_bkps is None:
                n_bkps = max(1, n // 100)
            bkps = algo.fit(signal).predict(n_bkps=n_bkps)

        # Remove last element (always equals n)
        if bkps and bkps[-1] == n:
            bkps = bkps[:-1]

        return [int(b) for b in bkps]

    except ImportError:
        # Fallback: simple CUSUM-like detection
        cumsum = np.cumsum(signal - np.mean(signal))
        diff = np.abs(np.diff(cumsum))

        # Find peaks above threshold
        threshold = np.mean(diff) + 2 * np.std(diff)
        bkps = np.where(diff > threshold)[0].tolist()

        return bkps


def mann_kendall_test(signal: np.ndarray) -> Tuple[float, float, str]:
    """
    Mann-Kendall trend test.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (statistic, p_value, trend_direction)

    Notes
    -----
    Non-parametric test for monotonic trend.
    trend_direction: 'increasing', 'decreasing', or 'no trend'
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 4:
        return np.nan, np.nan, 'insufficient data'

    try:
        import pymannkendall as mk
        result = mk.original_test(signal)
        return float(result.z), float(result.p), result.trend

    except ImportError:
        # Simplified implementation
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = signal[j] - signal[i]
                s += np.sign(diff)

        # Variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # P-value (two-tailed)
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

        if p < 0.05:
            direction = 'increasing' if z > 0 else 'decreasing'
        else:
            direction = 'no trend'

        return float(z), float(p), direction

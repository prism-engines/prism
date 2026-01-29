"""
Stationarity Test Primitives (103-105)

ADF, KPSS, Phillips-Perron tests.
"""

import numpy as np
from typing import Tuple, Optional


def adf_test(
    signal: np.ndarray,
    max_lag: int = None,
    regression: str = 'c'
) -> Tuple[float, float, int, dict]:
    """
    Augmented Dickey-Fuller test for unit root.

    Parameters
    ----------
    signal : np.ndarray
        Time series
    max_lag : int, optional
        Maximum lag order (default: auto via AIC)
    regression : str
        'c': constant only
        'ct': constant and trend
        'n': no constant or trend

    Returns
    -------
    tuple
        (adf_statistic, p_value, used_lag, critical_values)

    Notes
    -----
    H0: Unit root exists (non-stationary)
    H1: No unit root (stationary)

    p < 0.05: Reject H0, evidence of stationarity
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return np.nan, np.nan, 0, {}

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(signal, maxlag=max_lag, regression=regression)
        return (
            float(result[0]),  # ADF statistic
            float(result[1]),  # p-value
            int(result[2]),    # used lag
            result[4]          # critical values
        )
    except ImportError:
        # Fallback: simple implementation
        return _adf_simple(signal, max_lag, regression)


def kpss_test(
    signal: np.ndarray,
    regression: str = 'c',
    nlags: str = 'auto'
) -> Tuple[float, float, int, dict]:
    """
    KPSS test for stationarity.

    Parameters
    ----------
    signal : np.ndarray
        Time series
    regression : str
        'c': level stationarity
        'ct': trend stationarity
    nlags : str or int
        'auto' or number of lags

    Returns
    -------
    tuple
        (kpss_statistic, p_value, used_lags, critical_values)

    Notes
    -----
    H0: Series is stationary
    H1: Series has a unit root (non-stationary)

    p < 0.05: Reject H0, evidence of non-stationarity

    Note: Opposite interpretation from ADF!
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return np.nan, np.nan, 0, {}

    try:
        from statsmodels.tsa.stattools import kpss
        result = kpss(signal, regression=regression, nlags=nlags)
        return (
            float(result[0]),  # KPSS statistic
            float(result[1]),  # p-value
            int(result[2]),    # used lags
            result[3]          # critical values
        )
    except ImportError:
        # Fallback: simple implementation
        return _kpss_simple(signal, regression)


def philips_perron_test(
    signal: np.ndarray,
    regression: str = 'c'
) -> Tuple[float, float, dict]:
    """
    Phillips-Perron test for unit root.

    Parameters
    ----------
    signal : np.ndarray
        Time series
    regression : str
        'c': constant only
        'ct': constant and trend
        'n': no constant or trend

    Returns
    -------
    tuple
        (pp_statistic, p_value, critical_values)

    Notes
    -----
    H0: Unit root exists (non-stationary)
    H1: No unit root (stationary)

    Similar to ADF but uses non-parametric correction for
    serial correlation. More robust to heteroskedasticity.
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 10:
        return np.nan, np.nan, {}

    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(signal, trend=regression)
        return (
            float(pp.stat),
            float(pp.pvalue),
            pp.critical_values
        )
    except ImportError:
        # Fallback: use ADF as approximation
        adf_stat, adf_p, _, crit = adf_test(signal, regression=regression)
        return adf_stat, adf_p, crit


# Fallback implementations

def _adf_simple(
    signal: np.ndarray,
    max_lag: int = None,
    regression: str = 'c'
) -> Tuple[float, float, int, dict]:
    """Simple ADF implementation without statsmodels."""
    n = len(signal)

    if max_lag is None:
        max_lag = int(np.floor(12 * (n / 100) ** (1/4)))
        max_lag = min(max_lag, n // 3)

    # First difference
    diff = np.diff(signal)

    # Lagged level
    level_lag = signal[:-1]

    # Select lag using AIC
    best_aic = np.inf
    best_lag = 1

    for lag in range(1, max_lag + 1):
        y = diff[lag:]
        X = np.column_stack([
            np.ones(len(y)),
            level_lag[lag:n-1]
        ])

        # Add lagged differences
        for i in range(1, lag + 1):
            X = np.column_stack([X, diff[lag-i:n-1-i]])

        if regression == 'ct':
            X = np.column_stack([X, np.arange(len(y))])

        if len(y) < X.shape[1] + 2:
            continue

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            ssr = np.sum(resid ** 2)
            aic = len(y) * np.log(ssr / len(y) + 1e-10) + 2 * X.shape[1]

            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        except:
            continue

    # Final regression with best lag
    lag = best_lag
    y = diff[lag:]
    X = np.column_stack([
        np.ones(len(y)),
        level_lag[lag:n-1]
    ])

    for i in range(1, lag + 1):
        X = np.column_stack([X, diff[lag-i:n-1-i]])

    if regression == 'ct':
        X = np.column_stack([X, np.arange(len(y))])

    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = np.sum(resid ** 2) / (len(y) - X.shape[1])

        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * XtX_inv[1, 1])

        adf_stat = beta[1] / se

        # Approximate p-value using normal (conservative)
        from scipy import stats
        p_value = 2 * stats.norm.cdf(adf_stat)  # One-sided

        # Critical values (approximate)
        crit_values = {
            '1%': -3.51,
            '5%': -2.89,
            '10%': -2.58
        }

        return float(adf_stat), float(p_value), lag, crit_values

    except:
        return np.nan, np.nan, 0, {}


def _kpss_simple(
    signal: np.ndarray,
    regression: str = 'c'
) -> Tuple[float, float, int, dict]:
    """Simple KPSS implementation without statsmodels."""
    n = len(signal)

    # Detrend
    if regression == 'c':
        signal_detrended = signal - np.mean(signal)
    else:  # 'ct'
        t = np.arange(n)
        slope, intercept = np.polyfit(t, signal, 1)
        signal_detrended = signal - (slope * t + intercept)

    # Cumulative sum
    S = np.cumsum(signal_detrended)

    # Long-run variance estimate (Newey-West)
    nlags = int(np.floor(4 * (n / 100) ** (1/4)))

    gamma = []
    for lag in range(nlags + 1):
        if lag == 0:
            gamma.append(np.mean(signal_detrended ** 2))
        else:
            gamma.append(np.mean(signal_detrended[:-lag] * signal_detrended[lag:]))

    # Bartlett weights
    lrv = gamma[0]
    for lag in range(1, nlags + 1):
        weight = 1 - lag / (nlags + 1)
        lrv += 2 * weight * gamma[lag]

    if lrv <= 0:
        return np.nan, np.nan, nlags, {}

    # KPSS statistic
    kpss_stat = np.sum(S ** 2) / (n ** 2 * lrv)

    # Critical values
    if regression == 'c':
        crit_values = {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}
    else:
        crit_values = {'10%': 0.119, '5%': 0.146, '2.5%': 0.176, '1%': 0.216}

    # Approximate p-value (interpolate from critical values)
    if kpss_stat < crit_values['10%']:
        p_value = 0.10
    elif kpss_stat < crit_values['5%']:
        p_value = 0.05
    elif kpss_stat < crit_values['1%']:
        p_value = 0.01
    else:
        p_value = 0.01  # < 0.01

    return float(kpss_stat), p_value, nlags, crit_values

"""
Correlation Primitives (15-16)

Autocorrelation and partial autocorrelation.
"""

import numpy as np
from typing import Optional, Union


def autocorrelation(
    signal: np.ndarray,
    lag: Optional[int] = None,
    normalized: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute autocorrelation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    lag : int, optional
        Specific lag to compute (if None, compute all)
    normalized : bool
        If True, normalize to [-1, 1]

    Returns
    -------
    float or np.ndarray
        Autocorrelation at lag (scalar) or all lags (array)

    Notes
    -----
    R(k) = E[(x_t - μ)(x_{t+k} - μ)] / σ²
    Measures self-similarity at different time shifts.
    """
    signal = np.asarray(signal)
    signal = signal - np.nanmean(signal)
    n = len(signal)

    if lag is not None:
        # Single lag
        if lag >= n:
            return np.nan
        numerator = np.sum(signal[:n-lag] * signal[lag:])
        if normalized:
            denominator = np.sum(signal ** 2)
            return float(numerator / denominator) if denominator != 0 else 0.0
        return float(numerator / (n - lag))

    # All lags
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[n-1:]  # Keep positive lags only

    if normalized and acf[0] != 0:
        acf = acf / acf[0]

    return acf


def partial_autocorrelation(
    signal: np.ndarray,
    max_lag: int = 20
) -> np.ndarray:
    """
    Compute partial autocorrelation function (PACF).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    np.ndarray
        PACF values for lags 0 to max_lag

    Notes
    -----
    PACF at lag k measures correlation between x_t and x_{t-k}
    after removing the linear dependence on x_{t-1}, ..., x_{t-k+1}.
    Useful for identifying AR model order.
    """
    try:
        from statsmodels.tsa.stattools import pacf
        signal = np.asarray(signal)
        signal = signal[~np.isnan(signal)]

        if len(signal) < max_lag + 2:
            return np.full(max_lag + 1, np.nan)

        return pacf(signal, nlags=max_lag, method='ols')
    except ImportError:
        # Fallback: Yule-Walker approximation
        signal = np.asarray(signal)
        signal = signal[~np.isnan(signal)]
        n = len(signal)

        max_lag = min(max_lag, n - 2)
        pacf_vals = np.zeros(max_lag + 1)
        pacf_vals[0] = 1.0

        for k in range(1, max_lag + 1):
            # Levinson-Durbin recursion
            acf = autocorrelation(signal)[:k+1]
            if len(acf) < k + 1:
                break

            # Solve Yule-Walker equations
            try:
                r = acf[1:k+1]
                R = np.zeros((k, k))
                for i in range(k):
                    for j in range(k):
                        R[i, j] = acf[abs(i-j)]
                phi = np.linalg.solve(R, r)
                pacf_vals[k] = phi[-1]
            except:
                pacf_vals[k] = np.nan

        return pacf_vals

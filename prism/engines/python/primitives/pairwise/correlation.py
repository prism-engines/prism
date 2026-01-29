"""
Pairwise Correlation Primitives (36-40, 47)

Correlation, covariance, cross-correlation.
"""

import numpy as np
from typing import Optional, Union, Tuple


def correlation(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> float:
    """
    Compute Pearson correlation coefficient.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals

    Returns
    -------
    float
        Correlation coefficient in [-1, 1]

    Notes
    -----
    r = cov(a, b) / (std(a) * std(b))
    Measures linear relationship strength.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    # Remove NaNs
    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]

    if len(sig_a) < 2:
        return np.nan

    return float(np.corrcoef(sig_a, sig_b)[0, 1])


def covariance(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    ddof: int = 0
) -> float:
    """
    Compute covariance.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    float
        Covariance

    Notes
    -----
    cov(a, b) = E[(a - μ_a)(b - μ_b)]
    Measures joint variability.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]

    if len(sig_a) < 2:
        return np.nan

    return float(np.cov(sig_a, sig_b, ddof=ddof)[0, 1])


def cross_correlation(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    max_lag: int = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute cross-correlation at multiple lags.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    max_lag : int, optional
        Maximum lag (default: len/4)
    normalize : bool
        If True, normalize to [-1, 1]

    Returns
    -------
    np.ndarray
        Cross-correlation values for lags -max_lag to +max_lag

    Notes
    -----
    R_{ab}(τ) = E[(a(t) - μ_a)(b(t+τ) - μ_b)]
    Positive lag: a leads b
    Negative lag: b leads a
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    if max_lag is None:
        max_lag = n // 4

    # Center signals
    sig_a = sig_a - np.nanmean(sig_a)
    sig_b = sig_b - np.nanmean(sig_b)

    # Full cross-correlation
    xcorr = np.correlate(sig_a, sig_b, mode='full')

    # Extract relevant portion
    mid = len(xcorr) // 2
    xcorr = xcorr[mid - max_lag : mid + max_lag + 1]

    if normalize:
        norm = np.sqrt(np.sum(sig_a**2) * np.sum(sig_b**2))
        if norm > 0:
            xcorr = xcorr / norm

    return xcorr


def lag_at_max_xcorr(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    max_lag: int = None
) -> int:
    """
    Find lag at maximum cross-correlation.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    max_lag : int, optional
        Maximum lag to search

    Returns
    -------
    int
        Lag at maximum correlation
        Positive: a leads b
        Negative: b leads a
    """
    xcorr = cross_correlation(sig_a, sig_b, max_lag, normalize=True)
    n = min(len(sig_a), len(sig_b))

    if max_lag is None:
        max_lag = n // 4

    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[np.argmax(xcorr)])


def partial_correlation(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    controls: np.ndarray
) -> float:
    """
    Compute partial correlation controlling for other variables.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Variables to correlate
    controls : np.ndarray
        Control variables (1D or 2D)

    Returns
    -------
    float
        Partial correlation

    Notes
    -----
    Correlation between a and b after removing linear effect of controls.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()
    controls = np.asarray(controls)

    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)

    n = min(len(sig_a), len(sig_b), len(controls))
    sig_a, sig_b = sig_a[:n], sig_b[:n]
    controls = controls[:n]

    try:
        import pingouin as pg
        import pandas as pd

        df = pd.DataFrame({
            'a': sig_a,
            'b': sig_b,
        })
        for i in range(controls.shape[1]):
            df[f'c{i}'] = controls[:, i]

        covar = [f'c{i}' for i in range(controls.shape[1])]
        result = pg.partial_corr(data=df, x='a', y='b', covar=covar)
        return float(result['r'].values[0])

    except ImportError:
        # Manual calculation
        # Regress a and b on controls, get residuals
        X = np.column_stack([np.ones(n), controls])

        try:
            beta_a = np.linalg.lstsq(X, sig_a, rcond=None)[0]
            beta_b = np.linalg.lstsq(X, sig_b, rcond=None)[0]

            resid_a = sig_a - X @ beta_a
            resid_b = sig_b - X @ beta_b

            return correlation(resid_a, resid_b)
        except:
            return np.nan

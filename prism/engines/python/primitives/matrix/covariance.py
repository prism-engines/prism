"""
Matrix Covariance Primitives (56-57)

Covariance and correlation matrices.
"""

import numpy as np
from typing import Tuple, Optional


def covariance_matrix(
    signals: np.ndarray,
    ddof: int = 1,
    rowvar: bool = False
) -> np.ndarray:
    """
    Compute covariance matrix of signals.

    Parameters
    ----------
    signals : np.ndarray
        Matrix of signals (n_samples x n_signals) or (n_signals x n_samples)
    ddof : int
        Delta degrees of freedom (1 for sample, 0 for population)
    rowvar : bool
        If True, each row is a variable. If False, each column is a variable.

    Returns
    -------
    np.ndarray
        Covariance matrix (n_signals x n_signals)

    Notes
    -----
    Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)]
    Diagonal elements are variances.
    Off-diagonal elements are covariances.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        return np.array([[np.var(signals, ddof=ddof)]])

    # Handle NaN values
    if np.any(np.isnan(signals)):
        # Use pairwise complete observations
        if rowvar:
            n_vars = signals.shape[0]
        else:
            n_vars = signals.shape[1]

        cov = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i, n_vars):
                if rowvar:
                    x, y = signals[i], signals[j]
                else:
                    x, y = signals[:, i], signals[:, j]

                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > ddof:
                    cov[i, j] = np.cov(x[mask], y[mask], ddof=ddof)[0, 1]
                    cov[j, i] = cov[i, j]
                else:
                    cov[i, j] = np.nan
                    cov[j, i] = np.nan

        return cov

    return np.cov(signals, rowvar=rowvar, ddof=ddof)


def correlation_matrix(
    signals: np.ndarray,
    rowvar: bool = False
) -> np.ndarray:
    """
    Compute Pearson correlation matrix of signals.

    Parameters
    ----------
    signals : np.ndarray
        Matrix of signals (n_samples x n_signals) or (n_signals x n_samples)
    rowvar : bool
        If True, each row is a variable. If False, each column is a variable.

    Returns
    -------
    np.ndarray
        Correlation matrix (n_signals x n_signals)

    Notes
    -----
    Corr(X, Y) = Cov(X, Y) / (σ_X * σ_Y)
    Diagonal elements are 1.
    Off-diagonal elements in [-1, 1].
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        return np.array([[1.0]])

    # Handle NaN values
    if np.any(np.isnan(signals)):
        if rowvar:
            n_vars = signals.shape[0]
        else:
            n_vars = signals.shape[1]

        corr = np.eye(n_vars)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if rowvar:
                    x, y = signals[i], signals[j]
                else:
                    x, y = signals[:, i], signals[:, j]

                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 2:
                    corr[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]
                    corr[j, i] = corr[i, j]
                else:
                    corr[i, j] = np.nan
                    corr[j, i] = np.nan

        return corr

    return np.corrcoef(signals, rowvar=rowvar)

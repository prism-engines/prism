"""
Pairwise Regression Primitives (52-55)

Linear regression, ratio, product, difference.
"""

import numpy as np
from scipy import stats
from typing import Tuple


def linear_regression(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute linear regression: sig_b = slope * sig_a + intercept.

    Parameters
    ----------
    sig_a : np.ndarray
        Independent variable (x)
    sig_b : np.ndarray
        Dependent variable (y)

    Returns
    -------
    tuple
        (slope, intercept, r_squared, p_value)

    Notes
    -----
    Uses ordinary least squares.
    p_value tests H0: slope = 0.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]

    if len(sig_a) < 3:
        return np.nan, np.nan, np.nan, np.nan

    result = stats.linregress(sig_a, sig_b)

    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue ** 2),
        float(result.pvalue)
    )


def ratio(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute element-wise ratio sig_a / sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    epsilon : float
        Small value to prevent division by zero

    Returns
    -------
    np.ndarray
        Ratio signal

    Notes
    -----
    Useful for efficiency metrics, normalized comparisons.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    # Avoid division by zero
    sig_b_safe = np.where(np.abs(sig_b) < epsilon, epsilon * np.sign(sig_b + epsilon), sig_b)

    return sig_a / sig_b_safe


def product(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> np.ndarray:
    """
    Compute element-wise product sig_a * sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals

    Returns
    -------
    np.ndarray
        Product signal

    Notes
    -----
    Useful for power = force * velocity, etc.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] * sig_b[:n]


def difference(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> np.ndarray:
    """
    Compute element-wise difference sig_a - sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals

    Returns
    -------
    np.ndarray
        Difference signal

    Notes
    -----
    Useful for error signals, residuals, imbalances.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] - sig_b[:n]


def sum_signals(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> np.ndarray:
    """
    Compute element-wise sum sig_a + sig_b.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals

    Returns
    -------
    np.ndarray
        Sum signal
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    return sig_a[:n] + sig_b[:n]

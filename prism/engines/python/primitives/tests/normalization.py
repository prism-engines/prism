"""
Normalization Primitives (100-102)

Z-score, min-max, robust scaling.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


def z_score(
    data: np.ndarray,
    axis: int = None,
    ddof: int = 0
) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute (None for global)
    ddof : int
        Degrees of freedom for std calculation

    Returns
    -------
    np.ndarray
        Z-scored data

    Notes
    -----
    z = (x - μ) / σ

    Transforms to standard normal if input is normal.
    """
    data = np.asarray(data, dtype=float)

    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, ddof=ddof, keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    return (data - mean) / std


def z_score_significance(
    value: float,
    mean: float,
    std: float
) -> Tuple[float, float]:
    """
    Compute z-score and associated p-value.

    Parameters
    ----------
    value : float
        Observed value
    mean : float
        Population/baseline mean
    std : float
        Population/baseline standard deviation

    Returns
    -------
    z : float
        Z-score (number of standard deviations from mean)
    p_value : float
        Two-sided p-value

    Notes
    -----
    z = (x - μ) / σ

    Interpretation:
    - |z| < 1: Within normal range (68%)
    - |z| < 2: Within 95%
    - |z| < 3: Within 99.7%
    - |z| > 3: Highly unusual

    Physical interpretation:
    "How many standard deviations is this from normal?"

    Simple and intuitive. Use for quick anomaly flagging.
    """
    if std <= 0:
        return np.nan, np.nan

    z = (value - mean) / std
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(z), float(p_value)


def min_max_scale(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (0, 1),
    axis: int = None
) -> np.ndarray:
    """
    Scale data to a specified range.

    Parameters
    ----------
    data : np.ndarray
        Input data
    feature_range : tuple
        Desired range (min, max)
    axis : int, optional
        Axis along which to compute

    Returns
    -------
    np.ndarray
        Scaled data

    Notes
    -----
    x_scaled = (x - min) / (max - min) * (new_max - new_min) + new_min
    """
    data = np.asarray(data, dtype=float)
    new_min, new_max = feature_range

    min_val = np.nanmin(data, axis=axis, keepdims=True)
    max_val = np.nanmax(data, axis=axis, keepdims=True)

    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)

    scaled = (data - min_val) / range_val
    return scaled * (new_max - new_min) + new_min


def robust_scale(
    data: np.ndarray,
    center: bool = True,
    scale_iqr: bool = True,
    axis: int = None,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> np.ndarray:
    """
    Robust scaling using median and IQR.

    Parameters
    ----------
    data : np.ndarray
        Input data
    center : bool
        If True, subtract median
    scale_iqr : bool
        If True, divide by IQR
    axis : int, optional
        Axis along which to compute
    quantile_range : tuple
        Quantiles for IQR (default: 25th and 75th percentiles)

    Returns
    -------
    np.ndarray
        Robustly scaled data

    Notes
    -----
    x_scaled = (x - median) / IQR

    Robust to outliers compared to z-score.
    """
    data = np.asarray(data, dtype=float)

    result = data.copy()

    if center:
        median = np.nanmedian(data, axis=axis, keepdims=True)
        result = result - median

    if scale_iqr:
        q_low, q_high = quantile_range
        q1 = np.nanpercentile(data, q_low, axis=axis, keepdims=True)
        q3 = np.nanpercentile(data, q_high, axis=axis, keepdims=True)
        iqr = q3 - q1

        # Avoid division by zero
        iqr = np.where(iqr == 0, 1, iqr)
        result = result / iqr

    return result


def normalize_l2(
    data: np.ndarray,
    axis: int = None
) -> np.ndarray:
    """
    Normalize to unit L2 norm.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to normalize

    Returns
    -------
    np.ndarray
        L2-normalized data

    Notes
    -----
    x_norm = x / ||x||_2
    """
    data = np.asarray(data, dtype=float)

    norm = np.linalg.norm(data, axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1, norm)

    return data / norm


def normalize_l1(
    data: np.ndarray,
    axis: int = None
) -> np.ndarray:
    """
    Normalize to unit L1 norm.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to normalize

    Returns
    -------
    np.ndarray
        L1-normalized data

    Notes
    -----
    x_norm = x / ||x||_1
    """
    data = np.asarray(data, dtype=float)

    norm = np.sum(np.abs(data), axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1, norm)

    return data / norm

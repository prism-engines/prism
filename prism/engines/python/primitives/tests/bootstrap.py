"""
Bootstrap Primitives (106-107)

Bootstrap confidence intervals and permutation tests.
"""

import numpy as np
from typing import Tuple, Callable, Optional


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    method: str = 'percentile'
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    statistic : callable
        Function to compute statistic (default: mean)
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    n_bootstrap : int
        Number of bootstrap samples
    method : str
        'percentile': percentile method
        'bca': bias-corrected and accelerated

    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)

    Notes
    -----
    Percentile method: use percentiles of bootstrap distribution
    BCa: adjusts for bias and skewness (more accurate but slower)
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        return np.nan, np.nan, np.nan

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence

    if method == 'percentile':
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == 'bca':
        # Bias correction
        z0 = stats_norm_ppf(np.mean(bootstrap_stats < point_estimate))

        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats[i] = statistic(jackknife_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        # Adjusted percentiles
        z_alpha_lower = stats_norm_ppf(alpha / 2)
        z_alpha_upper = stats_norm_ppf(1 - alpha / 2)

        alpha_lower = stats_norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha_upper = stats_norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        lower = np.percentile(bootstrap_stats, 100 * alpha_lower)
        upper = np.percentile(bootstrap_stats, 100 * alpha_upper)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(point_estimate), float(lower), float(upper)


def bootstrap_mean(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    tuple
        (mean, lower_bound, upper_bound)
    """
    return bootstrap_ci(data, np.mean, confidence, n_bootstrap)


def bootstrap_std(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    tuple
        (std, lower_bound, upper_bound)
    """
    return bootstrap_ci(data, np.std, confidence, n_bootstrap)


def permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic: Callable = None,
    n_permutations: int = 1000,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Permutation test for comparing two samples.

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Two samples to compare
    statistic : callable, optional
        Test statistic function (default: difference in means)
    n_permutations : int
        Number of permutations
    alternative : str
        'two-sided', 'greater', or 'less'

    Returns
    -------
    tuple
        (observed_statistic, p_value)

    Notes
    -----
    Non-parametric test with no distributional assumptions.
    H0: samples come from same distribution
    H1: samples differ (in location, spread, or shape)
    """
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    n1, n2 = len(sample1), len(sample2)

    if n1 < 1 or n2 < 1:
        return np.nan, np.nan

    # Default statistic: difference in means
    if statistic is None:
        statistic = lambda x, y: np.mean(x) - np.mean(y)

    # Observed statistic
    observed = statistic(sample1, sample2)

    # Combined data
    combined = np.concatenate([sample1, sample2])

    # Permutation distribution
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm1 = combined[:n1]
        perm2 = combined[n1:]
        perm_stats[i] = statistic(perm1, perm2)

    # P-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(perm_stats >= observed)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(observed), float(p_value)


def block_bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    block_size: int = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Block bootstrap for time series (preserves autocorrelation).

    Parameters
    ----------
    data : np.ndarray
        Time series data
    statistic : callable
        Function to compute statistic
    block_size : int, optional
        Size of blocks (default: auto-selected)
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)

    Notes
    -----
    Standard bootstrap assumes i.i.d. data.
    Block bootstrap preserves serial correlation by
    resampling contiguous blocks.
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 4:
        return np.nan, np.nan, np.nan

    # Auto block size (Politis & Romano rule of thumb)
    if block_size is None:
        block_size = max(1, int(n ** (1/3)))

    # Point estimate
    point_estimate = statistic(data)

    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_size))

    # Bootstrap
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample blocks with replacement
        boot_sample = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            boot_sample.extend(data[start:start + block_size])

        # Truncate to original length
        boot_sample = np.array(boot_sample[:n])
        bootstrap_stats[i] = statistic(boot_sample)

    # Percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point_estimate), float(lower), float(upper)


# Helper functions

def stats_norm_ppf(p: float) -> float:
    """Standard normal quantile function (percent point function)."""
    from scipy import stats
    return stats.norm.ppf(p)


def stats_norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    from scipy import stats
    return stats.norm.cdf(x)

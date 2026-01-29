"""
Hypothesis Test Primitives (96-99)

t-test, F-test, chi-squared, non-parametric tests.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional


def t_test(
    sample: np.ndarray,
    popmean: float = 0.0
) -> Tuple[float, float]:
    """
    One-sample t-test.

    Parameters
    ----------
    sample : np.ndarray
        Sample data
    popmean : float
        Population mean under null hypothesis

    Returns
    -------
    tuple
        (t_statistic, p_value)

    Notes
    -----
    H0: sample mean = popmean
    H1: sample mean ≠ popmean (two-sided)
    """
    sample = np.asarray(sample).flatten()
    sample = sample[~np.isnan(sample)]

    if len(sample) < 2:
        return np.nan, np.nan

    result = stats.ttest_1samp(sample, popmean)
    return float(result.statistic), float(result.pvalue)


def t_test_paired(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> Tuple[float, float]:
    """
    Paired samples t-test.

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Paired samples (same length)

    Returns
    -------
    tuple
        (t_statistic, p_value)

    Notes
    -----
    H0: mean(sample1 - sample2) = 0
    H1: mean(sample1 - sample2) ≠ 0

    Use when samples are dependent (before/after, matched pairs).
    """
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()

    n = min(len(sample1), len(sample2))
    sample1, sample2 = sample1[:n], sample2[:n]

    mask = ~(np.isnan(sample1) | np.isnan(sample2))
    sample1, sample2 = sample1[mask], sample2[mask]

    if len(sample1) < 2:
        return np.nan, np.nan

    result = stats.ttest_rel(sample1, sample2)
    return float(result.statistic), float(result.pvalue)


def t_test_independent(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Independent samples t-test.

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Independent samples
    equal_var : bool
        If True, assume equal variance (Student's t)
        If False, use Welch's t-test

    Returns
    -------
    tuple
        (t_statistic, p_value)

    Notes
    -----
    H0: mean(sample1) = mean(sample2)
    H1: mean(sample1) ≠ mean(sample2)

    Use Welch's (equal_var=False) when variances may differ.
    """
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    if len(sample1) < 2 or len(sample2) < 2:
        return np.nan, np.nan

    result = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return float(result.statistic), float(result.pvalue)


def f_test(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> Tuple[float, float]:
    """
    F-test for equality of variances.

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Two samples

    Returns
    -------
    tuple
        (f_statistic, p_value)

    Notes
    -----
    H0: var(sample1) = var(sample2)
    H1: var(sample1) ≠ var(sample2)

    Assumes normally distributed data.
    """
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    if len(sample1) < 2 or len(sample2) < 2:
        return np.nan, np.nan

    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)

    if var2 == 0:
        return np.inf if var1 > 0 else 1.0, np.nan

    f_stat = var1 / var2
    df1, df2 = len(sample1) - 1, len(sample2) - 1

    # Two-sided p-value
    p = 2 * min(stats.f.cdf(f_stat, df1, df2),
                1 - stats.f.cdf(f_stat, df1, df2))

    return float(f_stat), float(p)


def chi_squared_test(
    observed: np.ndarray,
    expected: np.ndarray = None
) -> Tuple[float, float, int]:
    """
    Chi-squared goodness of fit test.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequencies
    expected : np.ndarray, optional
        Expected frequencies (uniform if None)

    Returns
    -------
    tuple
        (chi2_statistic, p_value, degrees_of_freedom)

    Notes
    -----
    H0: observed frequencies match expected
    H1: observed frequencies differ from expected
    """
    observed = np.asarray(observed).flatten()
    observed = observed[~np.isnan(observed)]

    if len(observed) < 2:
        return np.nan, np.nan, 0

    if expected is None:
        expected = np.full(len(observed), np.sum(observed) / len(observed))
    else:
        expected = np.asarray(expected).flatten()

    if len(expected) != len(observed):
        return np.nan, np.nan, 0

    result = stats.chisquare(observed, expected)
    df = len(observed) - 1

    return float(result.statistic), float(result.pvalue), df


def mannwhitney_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Independent samples
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    tuple
        (u_statistic, p_value)

    Notes
    -----
    Non-parametric alternative to independent t-test.
    H0: distributions are identical
    H1: one distribution is stochastically greater
    """
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    if len(sample1) < 2 or len(sample2) < 2:
        return np.nan, np.nan

    result = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def kruskal_test(
    *samples: np.ndarray
) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test.

    Parameters
    ----------
    samples : np.ndarray
        Two or more independent samples

    Returns
    -------
    tuple
        (h_statistic, p_value)

    Notes
    -----
    Non-parametric alternative to one-way ANOVA.
    H0: all populations have identical distributions
    H1: at least one population differs
    """
    cleaned_samples = []
    for s in samples:
        s = np.asarray(s).flatten()
        s = s[~np.isnan(s)]
        if len(s) >= 2:
            cleaned_samples.append(s)

    if len(cleaned_samples) < 2:
        return np.nan, np.nan

    result = stats.kruskal(*cleaned_samples)
    return float(result.statistic), float(result.pvalue)


def anova(
    *samples: np.ndarray
) -> Tuple[float, float]:
    """
    One-way ANOVA.

    Parameters
    ----------
    samples : np.ndarray
        Two or more independent samples

    Returns
    -------
    tuple
        (f_statistic, p_value)

    Notes
    -----
    H0: all population means are equal
    H1: at least one population mean differs

    Assumes normality and equal variances.
    """
    cleaned_samples = []
    for s in samples:
        s = np.asarray(s).flatten()
        s = s[~np.isnan(s)]
        if len(s) >= 2:
            cleaned_samples.append(s)

    if len(cleaned_samples) < 2:
        return np.nan, np.nan

    result = stats.f_oneway(*cleaned_samples)
    return float(result.statistic), float(result.pvalue)


def shapiro_test(
    sample: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Shapiro-Wilk test for normality.

    Parameters
    ----------
    sample : np.ndarray
        Sample values

    Returns
    -------
    statistic : float
        Shapiro-Wilk statistic
    p_value : float
        p-value
    is_normal : bool
        True if sample appears normal (p > 0.05)

    Notes
    -----
    Tests H₀: Sample comes from normal distribution

    High p-value → Fail to reject → Sample is plausibly normal
    Low p-value → Reject → Sample is NOT normal

    Physical interpretation:
    "Can we use parametric tests, or do we need non-parametric?"

    If not normal:
    - Use non-parametric tests (Mann-Kendall, permutation, etc.)
    - Be careful with t-tests and z-scores
    """
    sample = np.asarray(sample).flatten()
    sample = sample[~np.isnan(sample)]

    if len(sample) < 3:
        return np.nan, np.nan, False

    # Shapiro-Wilk has max sample size
    if len(sample) > 5000:
        sample = np.random.choice(sample, 5000, replace=False)

    try:
        statistic, p_value = stats.shapiro(sample)
        is_normal = p_value > 0.05
    except Exception:
        return np.nan, np.nan, False

    return float(statistic), float(p_value), is_normal


def levene_test(
    *samples
) -> Tuple[float, float]:
    """
    Levene's test for equality of variances.

    Parameters
    ----------
    *samples : np.ndarray
        Two or more samples to compare

    Returns
    -------
    statistic : float
        Levene's test statistic
    p_value : float
        p-value

    Notes
    -----
    Tests H₀: All groups have equal variance

    More robust than F-test (doesn't assume normality).

    Physical interpretation:
    "Has the variability changed across conditions/time?"
    """
    # Clean each sample
    cleaned = []
    for s in samples:
        s = np.asarray(s).flatten()
        s = s[~np.isnan(s)]
        if len(s) >= 2:
            cleaned.append(s)

    if len(cleaned) < 2:
        return np.nan, np.nan

    try:
        statistic, p_value = stats.levene(*cleaned)
    except Exception:
        return np.nan, np.nan

    return float(statistic), float(p_value)

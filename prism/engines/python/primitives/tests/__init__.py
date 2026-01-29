"""
Statistical Test Primitives (96-107)

Hypothesis tests, normalization, stationarity, bootstrap, null models.
"""

from .hypothesis import (
    t_test,
    t_test_paired,
    t_test_independent,
    f_test,
    chi_squared_test,
    mannwhitney_test,
    kruskal_test,
    anova,
    shapiro_test,
    levene_test,
)

from .nonparametric import (
    mann_kendall,
)

from .normalization import (
    z_score,
    z_score_significance,
    min_max_scale,
    robust_scale,
)

from .stationarity_tests import (
    adf_test,
    kpss_test,
    philips_perron_test,
)

from .bootstrap import (
    bootstrap_ci,
    bootstrap_mean,
    bootstrap_std,
    permutation_test,
)

from .null_models import (
    surrogate_test,
    marchenko_pastur_test,
    significance_summary,
)

__all__ = [
    # 96: t-test
    't_test',
    't_test_paired',
    't_test_independent',
    # 97: z-score
    'z_score',
    'z_score_significance',
    # 98: Mann-Kendall
    'mann_kendall',
    # 99: F-test and variance tests
    'f_test',
    'levene_test',
    # Hypothesis tests (additional)
    'chi_squared_test',
    'mannwhitney_test',
    'kruskal_test',
    'anova',
    # 100-101: Stationarity tests
    'adf_test',
    'kpss_test',
    'philips_perron_test',
    # 102: Normality test
    'shapiro_test',
    # 103: Normalization
    'min_max_scale',
    'robust_scale',
    # 104-105: Bootstrap
    'bootstrap_ci',
    'bootstrap_mean',
    'bootstrap_std',
    'permutation_test',
    # 106-107: Null models
    'surrogate_test',
    'marchenko_pastur_test',
    'significance_summary',
]

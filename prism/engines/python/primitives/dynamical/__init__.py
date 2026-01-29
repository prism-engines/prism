"""
Dynamical Primitives (86-95)

Lyapunov exponents, correlation dimension, recurrence quantification analysis.
"""

from .lyapunov import (
    lyapunov_rosenstein,
    lyapunov_kantz,
    lyapunov_spectrum,
)

from .dimension import (
    correlation_dimension,
    correlation_integral,
    information_dimension,
    kaplan_yorke_dimension,
)

from .rqa import (
    recurrence_matrix,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_rqa,
    max_diagonal_line,
    divergence_rqa,
)

__all__ = [
    # 86-87: Lyapunov
    'lyapunov_rosenstein',
    'lyapunov_kantz',
    'lyapunov_spectrum',
    # 88-89: Dimension
    'correlation_dimension',
    'correlation_integral',
    'information_dimension',
    'kaplan_yorke_dimension',
    # 90-95: RQA
    'recurrence_matrix',
    'recurrence_rate',
    'determinism',
    'laminarity',
    'trapping_time',
    'entropy_rqa',
    'max_diagonal_line',
    'divergence_rqa',
]

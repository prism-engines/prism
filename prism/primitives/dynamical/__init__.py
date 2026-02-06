"""
Dynamical Primitives (86-107)

Lyapunov exponents, correlation dimension, recurrence quantification analysis,
finite-time Lyapunov (FTLE), saddle detection, trajectory sensitivity.
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

from .ftle import (
    ftle_local_linearization,
    ftle_direct_perturbation,
    compute_cauchy_green_tensor,
    detect_lcs_ridges,
)

from .saddle import (
    estimate_jacobian_local,
    classify_jacobian_eigenvalues,
    detect_saddle_points,
    compute_separatrix_distance,
    compute_basin_stability,
)

from .sensitivity import (
    compute_variable_sensitivity,
    compute_directional_sensitivity,
    compute_sensitivity_evolution,
    detect_sensitivity_transitions,
    compute_influence_matrix,
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
    # 96-99: FTLE
    'ftle_local_linearization',
    'ftle_direct_perturbation',
    'compute_cauchy_green_tensor',
    'detect_lcs_ridges',
    # 100-104: Saddle Detection
    'estimate_jacobian_local',
    'classify_jacobian_eigenvalues',
    'detect_saddle_points',
    'compute_separatrix_distance',
    'compute_basin_stability',
    # 105-109: Trajectory Sensitivity
    'compute_variable_sensitivity',
    'compute_directional_sensitivity',
    'compute_sensitivity_evolution',
    'detect_sensitivity_transitions',
    'compute_influence_matrix',
]

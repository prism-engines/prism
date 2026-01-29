"""
Matrix Primitives (56-65)

All-signals computations that operate on matrices of multiple signals.
"""

from .covariance import (
    covariance_matrix,
    correlation_matrix,
)

from .decomposition import (
    eigendecomposition,
    svd,
    pca_loadings,
    factor_scores,
)

from .dmd import (
    dynamic_mode_decomposition,
)

from .information import (
    mutual_information_matrix,
    transfer_entropy_matrix,
    granger_matrix,
)

from .graph import (
    distance_matrix,
    temporal_distance_matrix,
    adjacency_matrix,
    laplacian_matrix,
    recurrence_matrix,
)

__all__ = [
    # 56-57: Covariance
    'covariance_matrix',
    'correlation_matrix',
    # 58-62: Decomposition
    'eigendecomposition',
    'svd',
    'pca_loadings',
    'factor_scores',
    # 60: DMD
    'dynamic_mode_decomposition',
    # 61-63: Information matrices
    'mutual_information_matrix',
    'transfer_entropy_matrix',
    'granger_matrix',
    # 64-65: Graph/Distance matrices
    'distance_matrix',
    'temporal_distance_matrix',
    'adjacency_matrix',
    'laplacian_matrix',
    'recurrence_matrix',
]

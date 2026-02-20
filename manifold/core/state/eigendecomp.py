"""
Eigendecomposition Engine (State Geometry).

Delegates to pmtvs state_eigendecomp and bootstrap_effective_dim.
"""

import warnings

import numpy as np
import polars as pl
from scipy import stats
from typing import Dict, Any, Optional, List, Literal

from pmtvs import (
    zscore_normalize,
    covariance_matrix,
    eigendecomposition,
    condition_number as _condition_number,
    shannon_entropy,
    effective_dimension,
    linear_regression,
)


def compute(
    signal_matrix: np.ndarray,
    centroid: np.ndarray = None,
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 2,
) -> Dict[str, Any]:
    """
    Compute cohort geometry (eigenvalues) from signal matrix.

    Args:
        signal_matrix: 2D array (n_signals, n_features)
        centroid: Pre-computed centroid. If None, computed from data.
        norm_method: Normalization before SVD
        min_signals: Minimum valid signals required

    Returns:
        dict with eigenvalues, effective_dim, derived metrics, loadings
    """
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    # Remove rows with any NaN
    valid_mask = np.all(np.isfinite(signal_matrix), axis=1)
    n_valid = int(valid_mask.sum())

    if n_valid < min_signals:
        return _empty_result(D)

    matrix = signal_matrix[valid_mask]

    # Normalize
    if norm_method == "zscore":
        matrix, _ = zscore_normalize(matrix, axis=0)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Covariance and eigendecomposition
    try:
        cov = covariance_matrix(matrix)
        eigenvalues, eigenvectors = eigendecomposition(cov)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    except (np.linalg.LinAlgError, ValueError):
        return _empty_result(D)

    # Ensure non-negative (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Derived metrics
    total_var = float(np.sum(eigenvalues))
    explained = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)

    eff_dim = float(effective_dimension(eigenvalues))

    # Eigenvalue entropy
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) > 0:
        p = pos / pos.sum()
        eig_entropy = float(-np.sum(p * np.log2(p + 1e-30)))
        max_ent = np.log2(len(p)) if len(p) > 1 else 1.0
        eig_entropy_norm = eig_entropy / max_ent if max_ent > 0 else 0.0
    else:
        eig_entropy = np.nan
        eig_entropy_norm = np.nan

    # Condition number
    cond_num = float(_condition_number(cov)) if total_var > 0 else np.nan

    # Eigenvalue ratios
    ratio_2_1 = float(eigenvalues[1] / eigenvalues[0]) if len(eigenvalues) > 1 and eigenvalues[0] > 0 else np.nan
    ratio_3_1 = float(eigenvalues[2] / eigenvalues[0]) if len(eigenvalues) > 2 and eigenvalues[0] > 0 else np.nan

    # Principal components (rows) and signal loadings
    principal_components = eigenvectors.T
    signal_loadings = matrix @ eigenvectors

    return {
        'eigenvalues': eigenvalues,
        'explained_ratio': explained,
        'total_variance': total_var,
        'effective_dim': eff_dim,
        'eigenvalue_entropy': eig_entropy,
        'eigenvalue_entropy_normalized': eig_entropy_norm,
        'condition_number': cond_num,
        'ratio_2_1': ratio_2_1,
        'ratio_3_1': ratio_3_1,
        'principal_components': principal_components,
        'signal_loadings': signal_loadings,
        'n_signals': n_valid,
        'n_features': D,
    }


def _empty_result(D: int) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'eigenvalues': np.full(D, np.nan),
        'explained_ratio': np.full(D, np.nan),
        'total_variance': np.nan,
        'effective_dim': np.nan,
        'eigenvalue_entropy': np.nan,
        'eigenvalue_entropy_normalized': np.nan,
        'condition_number': np.nan,
        'ratio_2_1': np.nan,
        'ratio_3_1': np.nan,
        'principal_components': None,
        'signal_loadings': None,
        'n_signals': 0,
        'n_features': D,
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'signal_0_end'],
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 3,
) -> pl.DataFrame:
    """
    Compute cohort geometry from signal_vector.parquet.

    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by
        norm_method: Normalization method
        min_signals: Minimum signals per group

    Returns:
        DataFrame with eigenvalues for each group
    """
    if feature_columns is None:
        feature_columns = [
            col for col in signal_vector.columns
            if col not in ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id', 'cohort']
            and signal_vector[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    results = []

    for group_keys, group_df in signal_vector.group_by(group_cols):
        matrix = group_df.select(feature_columns).to_numpy()
        geom = compute(matrix, norm_method=norm_method, min_signals=min_signals)

        row = dict(zip(group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
        row['effective_dim'] = geom['effective_dim']
        row['total_variance'] = geom['total_variance']
        row['condition_number'] = geom['condition_number']
        row['eigenvalue_entropy'] = geom['eigenvalue_entropy']
        row['eigenvalue_entropy_normalized'] = geom['eigenvalue_entropy_normalized']
        row['ratio_2_1'] = geom['ratio_2_1']
        row['ratio_3_1'] = geom['ratio_3_1']
        row['n_signals'] = geom['n_signals']

        eig = geom['eigenvalues']
        for i in range(min(5, len(eig))):
            row[f'eigenvalue_{i}'] = eig[i] if not np.isnan(eig[i]) else None

        exp_ratio = geom['explained_ratio']
        for i in range(min(3, len(exp_ratio))):
            row[f'explained_ratio_{i}'] = exp_ratio[i] if not np.isnan(exp_ratio[i]) else None

        results.append(row)

    return pl.DataFrame(results).sort(group_cols)


def enforce_eigenvector_continuity(
    eigenvectors_current: np.ndarray,
    eigenvectors_previous: np.ndarray,
) -> np.ndarray:
    """Ensure eigenvectors maintain consistent orientation across windows."""
    if eigenvectors_previous is None:
        return eigenvectors_current

    if eigenvectors_current.shape != eigenvectors_previous.shape:
        return eigenvectors_current

    corrected = eigenvectors_current.copy()
    n_vecs = corrected.shape[1] if corrected.ndim == 2 else 1

    if corrected.ndim == 1:
        if np.dot(corrected, eigenvectors_previous) < 0:
            corrected *= -1
        return corrected

    for j in range(n_vecs):
        dot = np.dot(eigenvectors_previous[:, j], corrected[:, j])
        if dot < 0:
            corrected[:, j] *= -1

    return corrected


def bootstrap_effective_dim(
    signal_matrix: np.ndarray,
    confidence_level: float = 0.95,
    **_kwargs,
) -> Dict[str, float]:
    """Jackknife confidence interval for effective dimensionality.

    Uses leave-one-out jackknife instead of bootstrap.  Row-resampling
    with replacement loses ~35% of unique rows per resample, which
    systematically deflates eigenvalue spread and biases effective_dim
    downward.  Jackknife avoids this: every resample has n-1 unique
    rows, so covariance rank is preserved and the CI reliably brackets
    the full-sample point estimate.
    """
    nan_result = {
        'effective_dim_mean': np.nan, 'effective_dim_std': np.nan,
        'effective_dim_lower': np.nan, 'effective_dim_upper': np.nan,
    }
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    valid_mask = np.all(np.isfinite(signal_matrix), axis=1)
    matrix = signal_matrix[valid_mask]
    n = len(matrix)

    if n < 4:
        return nan_result

    dims = []
    for i in range(n):
        sample = np.delete(matrix, i, axis=0)
        try:
            cov = covariance_matrix(sample)
            eigs, _ = eigendecomposition(cov)
            eigs = np.real(np.maximum(eigs, 0.0))
            dims.append(float(effective_dimension(eigs)))
        except (np.linalg.LinAlgError, ValueError):
            pass

    if not dims:
        return nan_result

    dims_arr = np.array(dims)
    jk_mean = float(dims_arr.mean())
    jk_var = ((n - 1) / n) * float(np.sum((dims_arr - jk_mean) ** 2))
    jk_std = float(np.sqrt(jk_var))

    alpha = (1 - confidence_level) / 2
    t_crit = float(stats.t.ppf(1 - alpha, df=n - 1))

    return {
        'effective_dim_mean': jk_mean,
        'effective_dim_std': jk_std,
        'effective_dim_lower': jk_mean - t_crit * jk_std,
        'effective_dim_upper': jk_mean + t_crit * jk_std,
    }


def enforce_eigenvector_continuity_sequence(
    eigenvectors_sequence: list,
) -> list:
    """Ensure eigenvector continuity across a sequence of windows."""
    if not eigenvectors_sequence:
        return eigenvectors_sequence

    corrected = [eigenvectors_sequence[0]]

    for i in range(1, len(eigenvectors_sequence)):
        prev = corrected[i - 1]
        curr = eigenvectors_sequence[i]

        if prev is None or curr is None:
            corrected.append(curr)
            continue

        corrected.append(enforce_eigenvector_continuity(curr, prev))

    return corrected


def compute_effective_dim_trend(
    effective_dims: np.ndarray,
) -> Dict[str, float]:
    """Compute trend statistics on effective dimension over time."""
    valid = ~np.isnan(effective_dims)
    if np.sum(valid) < 4:
        return {
            'eff_dim_slope': np.nan,
            'eff_dim_r2': np.nan,
        }

    x = np.arange(len(effective_dims))[valid].astype(float)
    y = effective_dims[valid]

    slope, _, r2, _ = linear_regression(x, y)

    return {
        'eff_dim_slope': slope,
        'eff_dim_r2': r2,
    }

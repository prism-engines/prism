"""
Matrix Decomposition Primitives (58-62)

Eigendecomposition, SVD, PCA loadings, factor scores.
"""

import numpy as np
from typing import Tuple, Optional


def eigendecomposition(
    matrix: np.ndarray,
    n_components: int = None,
    symmetric: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix (n x n)
    n_components : int, optional
        Number of eigenvalues/vectors to return (default: all)
    symmetric : bool
        If True, assume matrix is symmetric (faster, real eigenvalues)

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
        eigenvalues: 1D array of eigenvalues (sorted descending by magnitude)
        eigenvectors: 2D array, columns are eigenvectors

    Notes
    -----
    For covariance matrices: eigenvalues = variance explained
    eigenvectors = principal directions
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Handle NaN
    if np.any(np.isnan(matrix)):
        return np.full(matrix.shape[0], np.nan), np.full(matrix.shape, np.nan)

    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort by magnitude (descending)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    return eigenvalues.real, eigenvectors.real


def svd(
    matrix: np.ndarray,
    n_components: int = None,
    full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Singular Value Decomposition.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (m x n)
    n_components : int, optional
        Number of singular values/vectors to return
    full_matrices : bool
        If True, return full U and Vh matrices

    Returns
    -------
    tuple
        (U, S, Vh) where matrix ≈ U @ diag(S) @ Vh
        U: Left singular vectors (m x k)
        S: Singular values (k,)
        Vh: Right singular vectors (k x n)

    Notes
    -----
    Related to eigendecomposition:
    - U contains eigenvectors of matrix @ matrix.T
    - Vh contains eigenvectors of matrix.T @ matrix
    - S^2 are eigenvalues of both
    """
    matrix = np.asarray(matrix)

    if np.any(np.isnan(matrix)):
        m, n = matrix.shape
        k = min(m, n) if n_components is None else n_components
        return (
            np.full((m, k), np.nan),
            np.full(k, np.nan),
            np.full((k, n), np.nan)
        )

    U, S, Vh = np.linalg.svd(matrix, full_matrices=full_matrices)

    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        Vh = Vh[:n_components, :]

    return U, S, Vh


def pca_loadings(
    signals: np.ndarray,
    n_components: int = None,
    center: bool = True,
    scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA loadings (principal component weights).

    Parameters
    ----------
    signals : np.ndarray
        Data matrix (n_samples x n_signals)
    n_components : int, optional
        Number of components (default: all)
    center : bool
        If True, center data (subtract mean)
    scale : bool
        If True, standardize data (divide by std)

    Returns
    -------
    tuple
        (loadings, explained_variance_ratio, cumulative_variance)
        loadings: (n_signals x n_components) - weights for each component
        explained_variance_ratio: fraction of variance per component
        cumulative_variance: cumulative variance explained

    Notes
    -----
    Loadings show how each original signal contributes to each PC.
    Large absolute loading = signal strongly influences that PC.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape

    # Preprocessing
    if center:
        signals = signals - np.nanmean(signals, axis=0)
    if scale:
        std = np.nanstd(signals, axis=0)
        std[std == 0] = 1
        signals = signals / std

    # Handle NaN by replacing with column means
    col_means = np.nanmean(signals, axis=0)
    nan_mask = np.isnan(signals)
    signals = signals.copy()
    for j in range(n_signals):
        signals[nan_mask[:, j], j] = col_means[j]

    # SVD
    U, S, Vh = np.linalg.svd(signals, full_matrices=False)

    # Loadings are the right singular vectors
    loadings = Vh.T

    # Explained variance
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else explained_variance

    cumulative_variance = np.cumsum(explained_variance_ratio)

    if n_components is not None:
        loadings = loadings[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
        cumulative_variance = cumulative_variance[:n_components]

    return loadings, explained_variance_ratio, cumulative_variance


def factor_scores(
    signals: np.ndarray,
    n_components: int = None,
    center: bool = True,
    scale: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute factor scores (projections onto principal components).

    Parameters
    ----------
    signals : np.ndarray
        Data matrix (n_samples x n_signals)
    n_components : int, optional
        Number of components (default: all)
    center : bool
        If True, center data
    scale : bool
        If True, standardize data

    Returns
    -------
    tuple
        (scores, explained_variance_ratio)
        scores: (n_samples x n_components) - coordinates in PC space
        explained_variance_ratio: fraction of variance per component

    Notes
    -----
    Factor scores are the projections of each observation onto the PCs.
    Useful for dimensionality reduction and visualization.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape

    # Store means for projection
    means = np.nanmean(signals, axis=0)
    stds = np.nanstd(signals, axis=0) if scale else np.ones(n_signals)
    stds[stds == 0] = 1

    # Preprocessing
    signals_centered = signals - means
    if scale:
        signals_centered = signals_centered / stds

    # Handle NaN
    nan_mask = np.isnan(signals_centered)
    signals_centered = signals_centered.copy()
    for j in range(n_signals):
        col_mean = np.nanmean(signals_centered[:, j])
        signals_centered[nan_mask[:, j], j] = col_mean if not np.isnan(col_mean) else 0

    # SVD for scores
    U, S, Vh = np.linalg.svd(signals_centered, full_matrices=False)

    # Scores = U * S
    scores = U * S

    # Explained variance
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else explained_variance

    if n_components is not None:
        scores = scores[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

    return scores, explained_variance_ratio

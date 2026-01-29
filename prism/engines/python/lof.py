"""
LOF Engine.

Local Outlier Factor for anomaly detection in phase space.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def compute(y: np.ndarray, n_neighbors: int = 20, embedding_dim: int = 3) -> dict:
    """
    Compute Local Outlier Factor scores.

    Args:
        y: Signal values
        n_neighbors: Number of neighbors for LOF
        embedding_dim: Embedding dimension for phase space

    Returns:
        dict with lof_score, outlier_fraction
    """
    result = {
        'lof_score': np.nan,
        'outlier_fraction': np.nan
    }

    if len(y) < n_neighbors * 2 + embedding_dim:
        return result

    try:
        # Create time-delay embedding
        n = len(y) - embedding_dim + 1
        X = np.array([y[i:i+embedding_dim] for i in range(n)])

        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, n-1), contamination='auto')
        labels = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_

        result = {
            'lof_score': float(np.mean(scores)),
            'outlier_fraction': float(np.sum(labels == -1) / len(labels))
        }

    except Exception:
        pass

    return result

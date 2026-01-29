"""
Manifold Engine.

Computes PCA projection at every time slice (cross-signal).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute(observations: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Compute manifold projection at each time slice.

    Args:
        observations: DataFrame with entity_id, signal_id, I, y columns
        n_components: Number of PCA components

    Returns:
        DataFrame with manifold_x, manifold_y, manifold_z, manifold_velocity, manifold_acceleration
    """
    results = []

    for entity_id, group in observations.groupby('entity_id'):
        pivot = group.pivot(index='I', columns='signal_id', values='y').dropna()
        if len(pivot) < 10 or len(pivot.columns) < 2:
            continue

        n_comp = min(n_components, len(pivot.columns))
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(pivot.values)

        # Pad to 3 components
        if projected.shape[1] < 3:
            pad = np.zeros((projected.shape[0], 3 - projected.shape[1]))
            projected = np.hstack([projected, pad])

        # Manifold velocity
        velocity = np.zeros(len(projected))
        velocity[1:] = np.sqrt(np.sum(np.diff(projected, axis=0)**2, axis=1))

        # Manifold acceleration
        acceleration = np.zeros(len(projected))
        acceleration[1:] = np.diff(velocity)

        for i, I_val in enumerate(pivot.index):
            results.append({
                'entity_id': entity_id,
                'I': I_val,
                'manifold_x': projected[i, 0],
                'manifold_y': projected[i, 1],
                'manifold_z': projected[i, 2],
                'manifold_velocity': velocity[i],
                'manifold_acceleration': acceleration[i],
            })

    return pd.DataFrame(results)

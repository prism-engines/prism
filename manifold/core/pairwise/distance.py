"""
Distance Engine.

Computes pairwise distance measures:
- Dynamic Time Warping (DTW)

Thin wrapper over primitives/pairwise/distance.py.
"""

import numpy as np

from manifold.primitives.pairwise.distance import dynamic_time_warping


def compute_dtw(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two time series.

    Args:
        x: First time series
        y: Second time series

    Returns:
        DTW distance (float)
    """
    return dynamic_time_warping(x, y)

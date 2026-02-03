"""
Mutual Information Engine.

Computes mutual information between signal pairs.
"""

import numpy as np
from typing import Dict


def compute(y_a: np.ndarray, y_b: np.ndarray, n_bins: int = None) -> Dict[str, float]:
    """
    Compute mutual information between two signals (symmetric).

    Args:
        y_a: First signal values
        y_b: Second signal values
        n_bins: Number of bins for histograms (auto if None)

    Returns:
        dict with mutual_info, normalized_mi, h_a, h_b, h_joint
    """
    result = {
        'mutual_info': np.nan,
        'normalized_mi': np.nan,
        'h_a': np.nan,
        'h_b': np.nan,
        'h_joint': np.nan
    }

    # Handle NaN values
    y_a = np.asarray(y_a).flatten()
    y_b = np.asarray(y_b).flatten()

    # Align lengths
    n = min(len(y_a), len(y_b))
    if n < 50:
        return result

    y_a, y_b = y_a[:n], y_b[:n]

    # Remove pairs with NaN in either signal
    valid_mask = ~(np.isnan(y_a) | np.isnan(y_b))
    y_a = y_a[valid_mask]
    y_b = y_b[valid_mask]
    n = len(y_a)

    if n < 50:
        return result

    # Check for constant signals
    if np.std(y_a) < 1e-10 or np.std(y_b) < 1e-10:
        result['mutual_info'] = 0.0
        result['normalized_mi'] = 0.0
        return result

    try:
        # Determine number of bins using Sturges' rule if not provided
        if n_bins is None:
            n_bins = min(30, max(5, int(1 + np.log2(n))))

        # Joint histogram
        H_2d, x_edges, y_edges = np.histogram2d(y_a, y_b, bins=n_bins)

        # Normalize to get joint probability
        p_xy = H_2d / n

        # Marginal probabilities
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Compute entropies using natural log for nats (standard)
        def entropy(p):
            """Compute entropy H = -sum(p * log(p)) for p > 0."""
            p_pos = p[p > 0]
            return -np.sum(p_pos * np.log(p_pos))

        H_a = entropy(p_x)
        H_b = entropy(p_y)
        H_joint = entropy(p_xy.flatten())

        # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        mi = H_a + H_b - H_joint

        # Ensure non-negative (can be slightly negative due to numerical issues)
        mi = max(0.0, mi)

        result['mutual_info'] = float(mi)
        result['h_a'] = float(H_a)
        result['h_b'] = float(H_b)
        result['h_joint'] = float(H_joint)

        # Normalized mutual information
        # Use geometric mean normalization: NMI = MI / sqrt(H(X) * H(Y))
        if H_a > 0 and H_b > 0:
            nmi = mi / np.sqrt(H_a * H_b)
            result['normalized_mi'] = float(min(1.0, nmi))  # Cap at 1.0

    except Exception:
        pass

    return result

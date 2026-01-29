"""
Mutual Information Engine.

Computes mutual information between signal pairs.
"""

import numpy as np


def compute(y_a: np.ndarray, y_b: np.ndarray) -> dict:
    """
    Compute mutual information between two signals (symmetric).

    Args:
        y_a: First signal values
        y_b: Second signal values

    Returns:
        dict with mutual_info, normalized_mi
    """
    result = {
        'mutual_info': np.nan,
        'normalized_mi': np.nan,
    }

    n = min(len(y_a), len(y_b))
    if n < 50:
        return result

    y_a, y_b = y_a[:n], y_b[:n]

    try:
        # Subsample for efficiency
        step = max(1, n // 2000)
        y_a_sub = y_a[::step]
        y_b_sub = y_b[::step]
        n_sub = len(y_a_sub)

        # Histogram-based mutual information
        bins = min(30, int(np.sqrt(n_sub)))

        # Joint histogram
        H_ab, _, _ = np.histogram2d(y_a_sub, y_b_sub, bins=bins)
        H_ab = H_ab / n_sub + 1e-10

        # Marginals
        p_a = np.sum(H_ab, axis=1)
        p_b = np.sum(H_ab, axis=0)

        # Marginal entropies
        H_a = -np.sum(p_a * np.log(p_a + 1e-10))
        H_b = -np.sum(p_b * np.log(p_b + 1e-10))

        # Joint entropy
        H_ab_flat = H_ab.flatten()
        H_joint = -np.sum(H_ab_flat * np.log(H_ab_flat + 1e-10))

        # Mutual information
        mi = H_a + H_b - H_joint
        result['mutual_info'] = float(max(0, mi))

        # Normalized
        if H_a > 0 and H_b > 0:
            result['normalized_mi'] = float(mi / np.sqrt(H_a * H_b))
    except Exception:
        pass

    return result

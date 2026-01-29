"""
Transfer Entropy Engine.

Computes transfer entropy between signal pairs.
"""

import numpy as np


def compute(y_source: np.ndarray, y_target: np.ndarray) -> dict:
    """
    Compute transfer entropy from source to target.

    Args:
        y_source: Source signal values
        y_target: Target signal values

    Returns:
        dict with transfer_entropy, normalized_te
    """
    n = min(len(y_source), len(y_target))
    if n < 50:
        return {'transfer_entropy': np.nan, 'normalized_te': np.nan}

    y_source, y_target = y_source[:n], y_target[:n]
    n_bins = 5

    src_bins = np.digitize(y_source, np.percentile(y_source, np.linspace(0, 100, n_bins + 1)[1:-1]))
    tgt_bins = np.digitize(y_target, np.percentile(y_target, np.linspace(0, 100, n_bins + 1)[1:-1]))

    lag = 1
    src_past, tgt_past, tgt_future = src_bins[:-lag], tgt_bins[:-lag], tgt_bins[lag:]

    def calc_prob(arrays):
        combined = np.column_stack(arrays)
        unique, counts = np.unique(combined, axis=0, return_counts=True)
        return counts / len(combined)

    p1 = calc_prob([tgt_future, tgt_past])
    p2 = calc_prob([tgt_past])
    p3 = calc_prob([tgt_future, tgt_past, src_past])
    p4 = calc_prob([tgt_past, src_past])

    h1 = -np.sum(p1 * np.log(p1 + 1e-10))
    h2 = -np.sum(p2 * np.log(p2 + 1e-10))
    h3 = -np.sum(p3 * np.log(p3 + 1e-10))
    h4 = -np.sum(p4 * np.log(p4 + 1e-10))

    te = max(0, (h1 - h2) - (h3 - h4))
    return {
        'transfer_entropy': float(te),
        'normalized_te': float(te / (np.log(n_bins) + 1e-10))
    }

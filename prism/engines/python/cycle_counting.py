"""
Cycle Counting Engine.

Rainflow cycle counting for fatigue analysis.
Counts stress/strain cycles and their ranges.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Perform simplified rainflow cycle counting.

    Args:
        y: Signal values (stress, strain, load)

    Returns:
        dict with n_cycles, max_range, mean_range, damage_index
    """
    result = {
        'n_cycles': 0,
        'max_range': np.nan,
        'mean_range': np.nan,
        'damage_index': np.nan
    }

    if len(y) < 4:
        return result

    try:
        # Find peaks and valleys (turning points)
        dy = np.diff(y)
        sign_changes = np.where(dy[:-1] * dy[1:] < 0)[0] + 1

        if len(sign_changes) < 2:
            return result

        turning_points = y[sign_changes]

        # Simple range-pair counting
        ranges = []
        points = list(turning_points)

        while len(points) >= 4:
            # Check for closed cycle
            s1, s2, s3 = points[0], points[1], points[2]
            s4 = points[3] if len(points) > 3 else points[2]

            r1 = abs(s2 - s1)
            r2 = abs(s3 - s2)

            if r2 <= r1:
                ranges.append(r2)
                points.pop(2)
                points.pop(1)
            else:
                points.pop(0)

        # Remaining points form half-cycles
        for i in range(len(points) - 1):
            ranges.append(abs(points[i+1] - points[i]) / 2)

        if ranges:
            ranges = np.array(ranges)

            # Simple Palmgren-Miner damage (assuming S-N slope of 3)
            damage = np.sum((ranges / (np.max(ranges) + 1e-10)) ** 3)

            result = {
                'n_cycles': len(ranges),
                'max_range': float(np.max(ranges)),
                'mean_range': float(np.mean(ranges)),
                'damage_index': float(damage)
            }

    except Exception:
        pass

    return result

"""
Divergence Engine.

Computes information-theoretic divergence measures:
- KL divergence
- JS divergence

Thin wrapper over primitives/information/divergence.py.
"""

import numpy as np

from manifold.primitives.information.divergence import (
    kl_divergence as _kl_divergence,
    js_divergence as _js_divergence,
)


def kl_divergence(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute KL divergence from x to y.

    Args:
        x: Source distribution samples
        y: Target distribution samples

    Returns:
        KL(x || y)
    """
    return _kl_divergence(x, y)


def js_divergence(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between x and y.

    Args:
        x: First distribution samples
        y: Second distribution samples

    Returns:
        JS(x, y)
    """
    return _js_divergence(x, y)

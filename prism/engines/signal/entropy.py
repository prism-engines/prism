"""
Entropy Engine.

Computes sample entropy, permutation entropy, and approximate entropy.

Entropy measures quantify the irregularity/complexity of a signal:
- Sample entropy: Higher = more irregular/complex
- Permutation entropy: Higher = more random ordering
- Approximate entropy: Similar to sample entropy, slightly different algorithm
"""

import math
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy measures of signal.

    Args:
        y: Signal values

    Returns:
        dict with:
            - 'sample_entropy': Irregularity measure (0 = regular, higher = irregular)
            - 'permutation_entropy': Ordering complexity (0 = ordered, 1 = random)
            - 'approximate_entropy': Similar to sample entropy
    """
    result = {
        'sample_entropy': np.nan,
        'permutation_entropy': np.nan,
        'approximate_entropy': np.nan
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 50:
        return result

    # Try antropy library first (most accurate)
    try:
        from antropy import sample_entropy, perm_entropy, app_entropy

        result['sample_entropy'] = float(sample_entropy(y, order=2, metric='chebyshev'))
        result['permutation_entropy'] = float(perm_entropy(y, order=3, normalize=True))
        result['approximate_entropy'] = float(app_entropy(y, order=2, metric='chebyshev'))

        return result

    except ImportError:
        pass  # Fall back to manual implementation

    # Manual sample entropy
    result['sample_entropy'] = _sample_entropy(y)

    # Manual permutation entropy
    result['permutation_entropy'] = _permutation_entropy(y)

    # Approximate entropy (use sample entropy as approximation)
    result['approximate_entropy'] = result['sample_entropy']

    return result


def _sample_entropy(y: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Compute sample entropy.

    Args:
        y: Signal values
        m: Embedding dimension
        r_factor: Tolerance factor (multiplied by std)

    Returns:
        Sample entropy value
    """
    n = len(y)
    if n < 20:
        return np.nan

    # Subsample for very long signals
    if n > 1000:
        step = n // 1000
        y = y[::step]
        n = len(y)

    r = r_factor * np.std(y)
    if r <= 0:
        return np.nan

    def count_matches(template_len):
        """Count matching template pairs within tolerance r."""
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                # Chebyshev distance (max absolute difference)
                diff = np.max(np.abs(y[i:i + template_len] - y[j:j + template_len]))
                if diff <= r:
                    count += 1
        return count

    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0 or A == 0:
        return np.nan

    return float(-np.log(A / B))


def _permutation_entropy(y: np.ndarray, order: int = 3) -> float:
    """
    Compute permutation entropy.

    Args:
        y: Signal values
        order: Pattern length

    Returns:
        Normalized permutation entropy (0-1)
    """
    n = len(y)
    if n < order + 10:
        return np.nan

    # Count ordinal patterns
    patterns = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(y[i:i + order]))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    total = sum(patterns.values())
    if total == 0:
        return np.nan

    # Shannon entropy
    probs = [c / total for c in patterns.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)

    # Normalize by maximum entropy
    max_entropy = np.log(math.factorial(order))
    if max_entropy <= 0:
        return np.nan

    return float(entropy / max_entropy)

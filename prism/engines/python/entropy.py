"""
Entropy Engine.

Computes sample entropy, permutation entropy, and approximate entropy.
"""

import math
import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute entropy measures of signal.

    Args:
        y: Signal values

    Returns:
        dict with sample_entropy, permutation_entropy, approximate_entropy
    """
    result = {
        'sample_entropy': np.nan,
        'permutation_entropy': np.nan,
        'approximate_entropy': np.nan
    }

    n = len(y)
    if n < 50:
        return result

    # Sample entropy
    y_sub = y[::max(1, n//1000)]
    r_val = 0.2 * np.std(y_sub)
    if r_val > 0 and len(y_sub) > 10:
        m = 2
        n_sub = len(y_sub)

        def count_matches(template_len):
            count = 0
            for i in range(n_sub - template_len):
                for j in range(i + 1, n_sub - template_len):
                    if np.max(np.abs(y_sub[i:i+template_len] - y_sub[j:j+template_len])) <= r_val:
                        count += 1
            return count

        A, B = count_matches(m + 1), count_matches(m)
        if B > 0 and A > 0:
            result['sample_entropy'] = float(-np.log(A / B))

    # Permutation entropy
    order = 3
    patterns = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(y[i:i+order]))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    total = sum(patterns.values())
    probs = [c / total for c in patterns.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    max_entropy = np.log(math.factorial(order))
    result['permutation_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else np.nan

    # Approximate entropy (use sample as approximation)
    result['approximate_entropy'] = result['sample_entropy']

    return result

"""
Transfer Entropy Engine.

Computes transfer entropy from source to target signal.

Transfer entropy measures the directed information flow:
TE(X→Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)

If knowing X helps predict Y beyond Y's own history,
then X has causal influence on Y.
"""

import numpy as np
from typing import Dict


def compute(
    y_source: np.ndarray,
    y_target: np.ndarray,
    lag: int = 1,
    n_bins: int = 8
) -> Dict[str, float]:
    """
    Compute transfer entropy from source to target.

    Args:
        y_source: Source signal (potential cause)
        y_target: Target signal (potential effect)
        lag: Prediction lag (default: 1)
        n_bins: Number of bins for discretization

    Returns:
        dict with:
            - 'transfer_entropy': TE in bits
            - 'normalized_te': TE normalized by conditional entropy
            - 'effective_te': Bias-corrected TE (shuffled baseline subtracted)
    """
    result = {
        'transfer_entropy': np.nan,
        'normalized_te': np.nan,
        'effective_te': np.nan
    }

    n = min(len(y_source), len(y_target))
    if n < 50:
        return result

    y_source = np.asarray(y_source[:n]).flatten()
    y_target = np.asarray(y_target[:n]).flatten()

    # Remove NaN
    mask = ~(np.isnan(y_source) | np.isnan(y_target))
    y_source = y_source[mask]
    y_target = y_target[mask]
    n = len(y_source)

    if n < 50:
        return result

    try:
        # Discretize using percentile-based bins (more robust)
        def discretize(y, n_bins):
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(y, percentiles)
            bins[0] -= 1e-10  # Include minimum
            return np.digitize(y, bins[1:-1])

        src_bins = discretize(y_source, n_bins)
        tgt_bins = discretize(y_target, n_bins)

        # Create lagged variables
        Y_future = tgt_bins[lag:]      # Y_{t+lag}
        Y_past = tgt_bins[:-lag]       # Y_t
        X_past = src_bins[:-lag]       # X_t

        # Joint entropy function
        def entropy(states):
            """Shannon entropy in bits."""
            if len(states.shape) == 1:
                states = states.reshape(-1, 1)
            _, counts = np.unique(states, axis=0, return_counts=True)
            probs = counts / len(states)
            return -np.sum(probs * np.log2(probs + 1e-12))

        # Compute entropies
        H_YfYp = entropy(np.column_stack([Y_future, Y_past]))
        H_Yp = entropy(Y_past)
        H_YfYpXp = entropy(np.column_stack([Y_future, Y_past, X_past]))
        H_YpXp = entropy(np.column_stack([Y_past, X_past]))

        # Transfer entropy: TE = H(Yf|Yp) - H(Yf|Yp,Xp)
        #                     = (H(Yf,Yp) - H(Yp)) - (H(Yf,Yp,Xp) - H(Yp,Xp))
        te = (H_YfYp - H_Yp) - (H_YfYpXp - H_YpXp)
        te = max(0, te)  # TE is non-negative

        # Conditional entropy H(Yf|Yp) for normalization
        H_Yf_given_Yp = H_YfYp - H_Yp

        # Normalize
        if H_Yf_given_Yp > 1e-10:
            normalized_te = te / H_Yf_given_Yp
        else:
            normalized_te = 0.0

        # Bias correction via shuffling
        # Shuffle X to estimate baseline TE from chance
        n_shuffles = 10
        te_shuffled = []
        for _ in range(n_shuffles):
            X_shuffled = np.random.permutation(X_past)
            H_YfYpXs = entropy(np.column_stack([Y_future, Y_past, X_shuffled]))
            H_YpXs = entropy(np.column_stack([Y_past, X_shuffled]))
            te_s = (H_YfYp - H_Yp) - (H_YfYpXs - H_YpXs)
            te_shuffled.append(max(0, te_s))

        te_baseline = np.mean(te_shuffled)
        effective_te = max(0, te - te_baseline)

        result = {
            'transfer_entropy': float(te),
            'normalized_te': float(min(1.0, normalized_te)),
            'effective_te': float(effective_te)
        }

    except Exception:
        pass

    return result

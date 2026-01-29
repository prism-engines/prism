"""
Phase Space Reconstruction

Implements Takens' embedding theorem for reconstructing attractors
from scalar time series data.

References:
    Takens, F. (1981). "Detecting strange attractors in turbulence"
"""

import numpy as np
from scipy.stats import entropy


def embed_time_series(x: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """
    Takens' embedding theorem: reconstruct attractor from scalar time series.

    Parameters
    ----------
    x : array, shape (n_samples,)
        Scalar time series
    tau : int
        Time delay (in samples)
    dim : int
        Embedding dimension

    Returns
    -------
    embedded : array, shape (n_samples - (dim-1)*tau, dim)
        Embedded trajectory in reconstructed phase space

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> embedded = embed_time_series(x, tau=10, dim=3)
    >>> embedded.shape
    (980, 3)
    """
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Time series too short for tau={tau}, dim={dim}")

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau : i * tau + n]
    return embedded


def optimal_delay(x: np.ndarray, max_tau: int = 100) -> int:
    """
    Estimate optimal time delay using first minimum of mutual information.

    The mutual information between x(t) and x(t+tau) quantifies the
    information shared between the original and delayed signal.
    The first minimum indicates the delay where the signals are
    maximally independent while still related.

    Parameters
    ----------
    x : array
        Time series
    max_tau : int
        Maximum delay to consider

    Returns
    -------
    tau : int
        Optimal time delay

    References
    ----------
    Fraser, A. M., & Swinney, H. L. (1986). "Independent coordinates
    for strange attractors from mutual information"
    """
    # Ensure we don't exceed data length
    max_tau = min(max_tau, len(x) // 4)
    if max_tau < 2:
        return 1

    n_bins = min(20, int(np.sqrt(len(x) / 5)))
    bins = np.linspace(x.min(), x.max(), n_bins + 1)

    mi = []
    for tau in range(1, max_tau):
        # Discretize into bins
        x1 = np.digitize(x[:-tau], bins=bins) - 1
        x2 = np.digitize(x[tau:], bins=bins) - 1

        # Clip to valid range
        x1 = np.clip(x1, 0, n_bins - 1)
        x2 = np.clip(x2, 0, n_bins - 1)

        # Joint and marginal distributions
        joint = np.histogram2d(x1, x2, bins=n_bins)[0]
        joint = joint / joint.sum()

        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        # Only include non-zero probabilities
        px_nz = px[px > 0]
        py_nz = py[py > 0]
        joint_nz = joint[joint > 0]

        mi_val = entropy(px_nz) + entropy(py_nz) - entropy(joint_nz)
        mi.append(mi_val)

    if len(mi) < 3:
        return max(1, max_tau // 4)

    # First local minimum
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i + 1

    # Fallback: use 1/4 of max
    return max(1, max_tau // 4)


def optimal_embedding_dim(x: np.ndarray, tau: int, max_dim: int = 10) -> int:
    """
    Estimate optimal embedding dimension using false nearest neighbors (FNN).

    A point's nearest neighbor in dimension d may be a "false" neighbor
    if it's not close in dimension d+1. The algorithm finds the dimension
    where the FNN percentage drops below threshold.

    Parameters
    ----------
    x : array
        Time series
    tau : int
        Time delay
    max_dim : int
        Maximum dimension to consider

    Returns
    -------
    dim : int
        Optimal embedding dimension

    References
    ----------
    Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992).
    "Determining embedding dimension for phase-space reconstruction
    using a geometrical construction"
    """
    from scipy.spatial import KDTree

    fnn_threshold = 10.0  # Ratio threshold for false neighbors
    fnn_target = 0.01  # Target FNN percentage

    fnn_ratios = []

    for dim in range(1, max_dim):
        try:
            embedded_d = embed_time_series(x, tau, dim)
            embedded_d1 = embed_time_series(x, tau, dim + 1)
        except ValueError:
            # Time series too short
            return dim

        n_points = min(len(embedded_d), len(embedded_d1))
        if n_points < 10:
            return dim

        embedded_d = embedded_d[:n_points]
        embedded_d1 = embedded_d1[:n_points]

        # Find nearest neighbors in d dimensions
        tree = KDTree(embedded_d)
        distances, indices = tree.query(embedded_d, k=2)

        # Check if they're still neighbors in d+1 dimensions
        n_false = 0
        n_valid = 0

        for i in range(n_points):
            j = indices[i, 1]  # Nearest neighbor index
            dist_d = distances[i, 1]

            if dist_d > 0 and j < n_points:
                dist_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[j])

                if (dist_d1 / dist_d) > fnn_threshold:
                    n_false += 1
                n_valid += 1

        if n_valid > 0:
            fnn_ratio = n_false / n_valid
            fnn_ratios.append(fnn_ratio)

            if fnn_ratio < fnn_target:
                return dim + 1
        else:
            fnn_ratios.append(1.0)

    # Return dimension with minimum FNN if none below threshold
    if fnn_ratios:
        return np.argmin(fnn_ratios) + 2
    return max_dim

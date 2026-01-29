"""
Network Community Primitives (78, 85)

Modularity and community detection.
"""

import numpy as np
from typing import Tuple, List, Optional


def modularity(
    adjacency: np.ndarray,
    communities: np.ndarray,
    resolution: float = 1.0
) -> float:
    """
    Compute modularity of a community partition.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    communities : np.ndarray
        Community labels for each node
    resolution : float
        Resolution parameter (γ)

    Returns
    -------
    float
        Modularity in [-0.5, 1]

    Notes
    -----
    Q = (1/2m) Σ_{ij} (A_{ij} - γ k_i k_j / 2m) δ(c_i, c_j)

    where:
    - m = total edge weight
    - k_i = degree of node i
    - c_i = community of node i
    - δ = Kronecker delta

    Q > 0.3 typically indicates significant community structure.
    """
    adjacency = np.asarray(adjacency)
    communities = np.asarray(communities)
    n = adjacency.shape[0]

    # Total edge weight
    m = np.sum(adjacency) / 2
    if m == 0:
        return 0.0

    # Node degrees
    k = np.sum(adjacency, axis=1)

    # Compute modularity
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                Q += adjacency[i, j] - resolution * k[i] * k[j] / (2 * m)

    Q /= (2 * m)
    return float(Q)


def community_detection(
    adjacency: np.ndarray,
    method: str = 'louvain',
    resolution: float = 1.0,
    n_communities: int = None
) -> Tuple[np.ndarray, float]:
    """
    Detect communities in a network.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    method : str
        'louvain': Louvain algorithm (modularity optimization)
        'spectral': Spectral clustering
        'label_propagation': Label propagation
    resolution : float
        Resolution parameter for Louvain
    n_communities : int, optional
        Number of communities (for spectral only)

    Returns
    -------
    tuple
        (communities, modularity)
        communities: array of community labels
        modularity: modularity score of partition

    Notes
    -----
    Louvain: Fast, modularity-optimizing, hierarchical
    Spectral: Uses eigenvectors of Laplacian
    Label propagation: Fast, non-deterministic
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([]), 0.0

    if method == 'louvain':
        communities = _louvain(adjacency, resolution)
    elif method == 'spectral':
        if n_communities is None:
            n_communities = _estimate_n_communities(adjacency)
        communities = _spectral_clustering(adjacency, n_communities)
    elif method == 'label_propagation':
        communities = _label_propagation(adjacency)
    else:
        raise ValueError(f"Unknown method: {method}")

    mod = modularity(adjacency, communities, resolution)
    return communities, mod


def _louvain(
    adjacency: np.ndarray,
    resolution: float = 1.0,
    max_iter: int = 100
) -> np.ndarray:
    """Louvain community detection algorithm."""
    n = adjacency.shape[0]

    # Initialize: each node in its own community
    communities = np.arange(n)

    # Total edge weight
    m = np.sum(adjacency) / 2
    if m == 0:
        return communities

    # Node degrees
    k = np.sum(adjacency, axis=1)

    def delta_modularity(node, new_comm, old_comm):
        """Change in modularity from moving node to new_comm."""
        # Sum of weights to nodes in new community
        sum_in = np.sum(adjacency[node, communities == new_comm])

        # Sum of degrees in new community
        k_in = np.sum(k[communities == new_comm])

        # Sum of weights to nodes in old community (excluding self)
        mask_old = (communities == old_comm) & (np.arange(n) != node)
        sum_out = np.sum(adjacency[node, mask_old])

        # Sum of degrees in old community (excluding self)
        k_out = np.sum(k[mask_old])

        delta = (sum_in - sum_out) / m - resolution * k[node] * (k_in - k_out) / (2 * m**2)
        return delta

    # Iterate until convergence
    for _ in range(max_iter):
        changed = False
        order = np.random.permutation(n)

        for node in order:
            old_comm = communities[node]

            # Find neighboring communities
            neighbor_comms = np.unique(communities[adjacency[node] > 0])

            best_comm = old_comm
            best_delta = 0

            for new_comm in neighbor_comms:
                if new_comm != old_comm:
                    delta = delta_modularity(node, new_comm, old_comm)
                    if delta > best_delta:
                        best_delta = delta
                        best_comm = new_comm

            if best_comm != old_comm:
                communities[node] = best_comm
                changed = True

        if not changed:
            break

    # Renumber communities to be consecutive
    unique_comms = np.unique(communities)
    mapping = {old: new for new, old in enumerate(unique_comms)}
    communities = np.array([mapping[c] for c in communities])

    return communities


def _spectral_clustering(
    adjacency: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """Spectral clustering using Laplacian eigenvectors."""
    n = adjacency.shape[0]

    if n_clusters >= n:
        return np.arange(n)

    # Normalized Laplacian
    degrees = np.sum(adjacency, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    # Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Use k smallest non-trivial eigenvectors
    idx = np.argsort(eigenvalues)[1:n_clusters+1]
    embedding = eigenvectors[:, idx]

    # Normalize rows
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    embedding = embedding / row_norms

    # K-means clustering
    communities = _kmeans(embedding, n_clusters)

    return communities


def _label_propagation(
    adjacency: np.ndarray,
    max_iter: int = 100
) -> np.ndarray:
    """Label propagation community detection."""
    n = adjacency.shape[0]

    # Initialize: each node in its own community
    labels = np.arange(n)

    for _ in range(max_iter):
        changed = False
        order = np.random.permutation(n)

        for node in order:
            # Count neighbor labels (weighted by edge strength)
            neighbor_labels = labels[adjacency[node] > 0]
            neighbor_weights = adjacency[node, adjacency[node] > 0]

            if len(neighbor_labels) == 0:
                continue

            # Weighted vote
            unique_labels = np.unique(neighbor_labels)
            label_weights = np.array([
                np.sum(neighbor_weights[neighbor_labels == lab])
                for lab in unique_labels
            ])

            # Break ties randomly
            max_weight = np.max(label_weights)
            candidates = unique_labels[label_weights == max_weight]
            new_label = np.random.choice(candidates)

            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    # Renumber
    unique_labels = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([mapping[lab] for lab in labels])

    return labels


def _estimate_n_communities(adjacency: np.ndarray) -> int:
    """Estimate number of communities using eigengap heuristic."""
    n = adjacency.shape[0]

    # Normalized Laplacian eigenvalues
    degrees = np.sum(adjacency, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))

    # Find largest eigengap
    gaps = np.diff(eigenvalues[:min(n//2, 20)])
    if len(gaps) == 0:
        return 2

    k = np.argmax(gaps) + 1
    return max(2, min(k, n // 2))


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Simple k-means clustering."""
    n = len(X)

    # Initialize centers randomly
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign points to nearest center
        for i in range(n):
            dists = np.linalg.norm(X[i] - centers, axis=1)
            labels[i] = np.argmin(dists)

        # Update centers
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return labels

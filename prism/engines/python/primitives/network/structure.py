"""
Network Structure Primitives (75-77, 84)

Thresholding, density, clustering, components.
"""

import numpy as np
from typing import Tuple, List, Optional


def threshold_matrix(
    matrix: np.ndarray,
    threshold: float = None,
    percentile: float = None,
    keep: str = 'above',
    binary: bool = True
) -> np.ndarray:
    """
    Create binary or weighted adjacency from similarity/distance matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Similarity or distance matrix (n x n)
    threshold : float, optional
        Absolute threshold value
    percentile : float, optional
        Percentile threshold (0-100)
    keep : str
        'above': keep values above threshold (for similarity)
        'below': keep values below threshold (for distance)
    binary : bool
        If True, return binary adjacency (0/1)

    Returns
    -------
    np.ndarray
        Adjacency matrix

    Notes
    -----
    Exactly one of threshold or percentile must be specified.
    """
    matrix = np.asarray(matrix).copy()
    n = matrix.shape[0]

    # Compute threshold from percentile if needed
    if percentile is not None:
        # Exclude diagonal
        off_diag = matrix[~np.eye(n, dtype=bool)]
        off_diag = off_diag[~np.isnan(off_diag)]
        threshold = np.percentile(off_diag, percentile)

    if threshold is None:
        raise ValueError("Must specify threshold or percentile")

    # Zero the diagonal
    np.fill_diagonal(matrix, 0)

    # Apply threshold
    if keep == 'above':
        adj = np.where(matrix >= threshold, matrix, 0)
    else:  # 'below'
        adj = np.where(matrix <= threshold, matrix, 0)

    if binary:
        adj = (adj != 0).astype(float)

    return adj


def network_density(
    adjacency: np.ndarray,
    directed: bool = False
) -> float:
    """
    Compute network density.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    directed : bool
        If True, treat as directed graph

    Returns
    -------
    float
        Density in [0, 1]

    Notes
    -----
    Density = actual edges / possible edges

    Undirected: ρ = 2m / (n * (n-1))
    Directed: ρ = m / (n * (n-1))

    where m = number of edges, n = number of nodes
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if n < 2:
        return 0.0

    # Count edges
    edges = np.sum(adjacency != 0)
    np.fill_diagonal(adjacency, 0)  # Ensure no self-loops counted

    # Possible edges
    if directed:
        possible = n * (n - 1)
    else:
        possible = n * (n - 1) / 2
        edges = edges / 2  # Each edge counted twice in undirected

    return float(edges / possible) if possible > 0 else 0.0


def clustering_coefficient(
    adjacency: np.ndarray,
    node: int = None,
    weighted: bool = False
) -> float:
    """
    Compute clustering coefficient.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    node : int, optional
        Specific node (if None, returns global average)
    weighted : bool
        If True, use weighted clustering

    Returns
    -------
    float
        Clustering coefficient in [0, 1]

    Notes
    -----
    C_i = (triangles at i) / (possible triangles at i)

    Measures local density of connections.
    High C = neighbors tend to be connected to each other.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    # Ensure binary for unweighted
    if not weighted:
        adj = (adjacency != 0).astype(float)
    else:
        adj = adjacency.copy()
        # Normalize weights to [0, 1]
        max_weight = np.max(np.abs(adj))
        if max_weight > 0:
            adj = adj / max_weight

    # Zero diagonal
    np.fill_diagonal(adj, 0)

    def node_clustering(i):
        # Neighbors of i
        neighbors = np.where(adj[i, :] != 0)[0]
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count triangles
        if weighted:
            # Weighted: (w_ij * w_jk * w_ki)^(1/3) summed
            triangles = 0.0
            for j in neighbors:
                for l in neighbors:
                    if j < l and adj[j, l] != 0:
                        triangles += (adj[i, j] * adj[j, l] * adj[l, i]) ** (1/3)
        else:
            # Unweighted: count connections between neighbors
            neighbor_adj = adj[np.ix_(neighbors, neighbors)]
            triangles = np.sum(neighbor_adj) / 2

        possible = k * (k - 1) / 2
        return triangles / possible if possible > 0 else 0.0

    if node is not None:
        return float(node_clustering(node))

    # Global average
    coefficients = [node_clustering(i) for i in range(n)]
    return float(np.mean(coefficients))


def connected_components(
    adjacency: np.ndarray,
    return_labels: bool = True
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Find connected components in graph.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    return_labels : bool
        If True, also return component labels for each node

    Returns
    -------
    tuple
        (n_components, labels) where labels[i] = component ID of node i

    Notes
    -----
    Uses BFS to find components.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    # Make undirected for component finding
    adj = (adjacency != 0) | (adjacency.T != 0)
    np.fill_diagonal(adj, False)

    labels = np.full(n, -1)
    component_id = 0

    for start in range(n):
        if labels[start] >= 0:
            continue

        # BFS from start
        queue = [start]
        labels[start] = component_id

        while queue:
            node = queue.pop(0)
            neighbors = np.where(adj[node, :])[0]

            for neighbor in neighbors:
                if labels[neighbor] < 0:
                    labels[neighbor] = component_id
                    queue.append(neighbor)

        component_id += 1

    if return_labels:
        return component_id, labels
    else:
        return component_id, None


def assortativity(
    adjacency: np.ndarray,
    attribute: np.ndarray = None
) -> float:
    """
    Compute degree assortativity coefficient.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    attribute : np.ndarray, optional
        Node attributes (if None, uses degree)

    Returns
    -------
    float
        Assortativity coefficient in [-1, 1]

    Notes
    -----
    Measures tendency of similar nodes to connect.
    r > 0: assortative (high-degree to high-degree)
    r < 0: disassortative (high-degree to low-degree)
    r = 0: no correlation
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    adj = (adjacency != 0).astype(float)
    np.fill_diagonal(adj, 0)

    # Node degrees or attributes
    if attribute is None:
        attribute = np.sum(adj, axis=1)
    else:
        attribute = np.asarray(attribute)

    # Find edges
    edges = np.argwhere(adj > 0)

    if len(edges) == 0:
        return 0.0

    # Source and target attributes
    x = attribute[edges[:, 0]]
    y = attribute[edges[:, 1]]

    # Pearson correlation
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)

    if sx == 0 or sy == 0:
        return 0.0

    r = np.mean((x - mx) * (y - my)) / (sx * sy)
    return float(r)

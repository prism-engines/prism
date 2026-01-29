"""
Network Centrality Primitives (79-82)

Degree, betweenness, eigenvector, closeness centrality.
"""

import numpy as np
from typing import Optional


def centrality_degree(
    adjacency: np.ndarray,
    weighted: bool = False,
    normalized: bool = True,
    direction: str = 'both'
) -> np.ndarray:
    """
    Compute degree centrality.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted degree (strength)
    normalized : bool
        If True, normalize to [0, 1]
    direction : str
        'in': in-degree, 'out': out-degree, 'both': total degree

    Returns
    -------
    np.ndarray
        Degree centrality for each node

    Notes
    -----
    Simplest centrality measure.
    d_i = number (or sum of weights) of edges incident to node i
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if not weighted:
        adj = (adjacency != 0).astype(float)
    else:
        adj = np.abs(adjacency)

    np.fill_diagonal(adj, 0)

    if direction == 'in':
        degree = np.sum(adj, axis=0)  # Sum columns
    elif direction == 'out':
        degree = np.sum(adj, axis=1)  # Sum rows
    else:  # 'both'
        degree = np.sum(adj, axis=0) + np.sum(adj, axis=1)
        if not np.allclose(adj, adj.T):  # Directed
            pass  # Keep sum of in and out
        else:
            degree = degree / 2  # Undirected: each edge counted twice

    if normalized and n > 1:
        max_degree = n - 1 if not weighted else np.max(degree)
        if max_degree > 0:
            degree = degree / max_degree

    return degree


def centrality_betweenness(
    adjacency: np.ndarray,
    weighted: bool = False,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute betweenness centrality.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted shortest paths
    normalized : bool
        If True, normalize to [0, 1]

    Returns
    -------
    np.ndarray
        Betweenness centrality for each node

    Notes
    -----
    B_i = Σ_{s≠i≠t} σ_{st}(i) / σ_{st}

    where σ_{st} = number of shortest paths from s to t
          σ_{st}(i) = number of those passing through i

    High betweenness = node lies on many shortest paths (bridge/broker)
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if weighted:
        # Convert to distance (inverse of weight)
        with np.errstate(divide='ignore'):
            dist_matrix = np.where(adjacency != 0, 1.0 / np.abs(adjacency), np.inf)
    else:
        dist_matrix = np.where(adjacency != 0, 1.0, np.inf)

    np.fill_diagonal(dist_matrix, 0)

    betweenness = np.zeros(n)

    # Brandes' algorithm
    for s in range(n):
        # Single-source shortest paths
        dist = np.full(n, np.inf)
        dist[s] = 0
        sigma = np.zeros(n)  # Number of shortest paths
        sigma[s] = 1
        pred = [[] for _ in range(n)]  # Predecessors

        # BFS/Dijkstra
        stack = []
        if weighted:
            # Dijkstra
            import heapq
            heap = [(0, s)]
            while heap:
                d, v = heapq.heappop(heap)
                if d > dist[v]:
                    continue
                stack.append(v)
                for w in range(n):
                    if dist_matrix[v, w] < np.inf:
                        new_dist = dist[v] + dist_matrix[v, w]
                        if new_dist < dist[w]:
                            dist[w] = new_dist
                            sigma[w] = sigma[v]
                            pred[w] = [v]
                            heapq.heappush(heap, (new_dist, w))
                        elif new_dist == dist[w]:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
        else:
            # BFS
            queue = [s]
            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in range(n):
                    if adjacency[v, w] != 0:
                        if dist[w] == np.inf:
                            dist[w] = dist[v] + 1
                            sigma[w] = sigma[v]
                            pred[w] = [v]
                            queue.append(w)
                        elif dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)

        # Backpropagate
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalize
    if normalized and n > 2:
        # Undirected: each pair counted once
        scale = 2.0 / ((n - 1) * (n - 2))
        betweenness *= scale

    return betweenness


def centrality_eigenvector(
    adjacency: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    Compute eigenvector centrality.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    max_iter : int
        Maximum iterations for power method
    tolerance : float
        Convergence tolerance

    Returns
    -------
    np.ndarray
        Eigenvector centrality for each node (normalized to unit length)

    Notes
    -----
    x_i = (1/λ) Σ_j A_{ij} x_j

    Node's centrality proportional to centrality of its neighbors.
    High eigenvector centrality = connected to other high-centrality nodes.

    Uses power iteration to find principal eigenvector.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Use absolute values for centrality
    adj = np.abs(adjacency)
    np.fill_diagonal(adj, 0)

    # Power iteration
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)

        if norm == 0:
            return np.zeros(n)

        x_new = x_new / norm

        if np.linalg.norm(x_new - x) < tolerance:
            break

        x = x_new

    # Make all positive (principal eigenvector for non-negative matrix)
    if np.sum(x) < 0:
        x = -x

    return x


def centrality_closeness(
    adjacency: np.ndarray,
    weighted: bool = False,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute closeness centrality.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    weighted : bool
        If True, use weighted shortest paths
    normalized : bool
        If True, normalize to [0, 1]

    Returns
    -------
    np.ndarray
        Closeness centrality for each node

    Notes
    -----
    C_i = (n-1) / Σ_j d(i,j)

    where d(i,j) = shortest path distance from i to j.

    High closeness = on average close to all other nodes.
    For disconnected graphs, only considers reachable nodes.
    """
    adjacency = np.asarray(adjacency)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Compute all shortest paths using Floyd-Warshall
    if weighted:
        with np.errstate(divide='ignore'):
            dist = np.where(adjacency != 0, 1.0 / np.abs(adjacency), np.inf)
    else:
        dist = np.where(adjacency != 0, 1.0, np.inf)

    np.fill_diagonal(dist, 0)

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    closeness = np.zeros(n)

    for i in range(n):
        reachable = dist[i, :] < np.inf
        reachable[i] = False
        n_reachable = np.sum(reachable)

        if n_reachable > 0:
            avg_dist = np.sum(dist[i, reachable]) / n_reachable
            closeness[i] = 1.0 / avg_dist if avg_dist > 0 else 0.0

            if normalized:
                closeness[i] *= n_reachable / (n - 1)

    return closeness

"""
Causal Network Analysis

Analyzes the structure of causal relationships between signals.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CausalNetwork:
    """Container for causal network structure."""
    nodes: List[str]
    adjacency_matrix: np.ndarray  # Weighted edges
    edge_significance: np.ndarray  # P-values or confidence

    def get_drivers(self, threshold: float = 0.1) -> List[str]:
        """Find nodes that drive others but aren't driven."""
        n = len(self.nodes)
        out_degree = np.sum(self.adjacency_matrix > threshold, axis=1)
        in_degree = np.sum(self.adjacency_matrix > threshold, axis=0)

        drivers = []
        for i, node in enumerate(self.nodes):
            if out_degree[i] > 0 and in_degree[i] == 0:
                drivers.append(node)
        return drivers

    def get_sinks(self, threshold: float = 0.1) -> List[str]:
        """Find nodes that are driven but don't drive others."""
        n = len(self.nodes)
        out_degree = np.sum(self.adjacency_matrix > threshold, axis=1)
        in_degree = np.sum(self.adjacency_matrix > threshold, axis=0)

        sinks = []
        for i, node in enumerate(self.nodes):
            if in_degree[i] > 0 and out_degree[i] == 0:
                sinks.append(node)
        return sinks

    def get_hubs(self, threshold: float = 0.1) -> List[str]:
        """Find highly connected nodes."""
        n = len(self.nodes)
        total_degree = (
            np.sum(self.adjacency_matrix > threshold, axis=1) +
            np.sum(self.adjacency_matrix > threshold, axis=0)
        )

        mean_degree = np.mean(total_degree)
        if mean_degree == 0:
            return []

        hubs = [self.nodes[i] for i in range(n) if total_degree[i] > 2 * mean_degree]
        return hubs

    def feedback_loops(self, threshold: float = 0.1) -> List[Tuple[str, str]]:
        """Find bidirectional edges (feedback loops)."""
        loops = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if (self.adjacency_matrix[i, j] > threshold and
                    self.adjacency_matrix[j, i] > threshold):
                    loops.append((self.nodes[i], self.nodes[j]))
        return loops

    def causal_hierarchy_score(self) -> float:
        """
        Measure how hierarchical the causal structure is.

        0 = fully circular/bidirectional
        1 = perfectly hierarchical (DAG)
        """
        A = self.adjacency_matrix
        total = np.sum(A)
        if total == 0:
            return 1.0  # No edges = trivially hierarchical

        # Compare adjacency matrix with its transpose
        symmetry = np.sum(np.minimum(A, A.T)) / total
        return 1 - symmetry


def network_metrics(network: CausalNetwork, threshold: float = 0.1) -> Dict[str, float]:
    """
    Compute summary metrics for causal network.

    Parameters
    ----------
    network : CausalNetwork
    threshold : float
        Edge threshold for binary operations

    Returns
    -------
    metrics : dict
        Summary metrics
    """
    A = (network.adjacency_matrix > threshold).astype(float)
    n = len(network.nodes)

    # Density
    n_edges = np.sum(A)
    max_edges = n * (n - 1)
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Reciprocity (bidirectional edges)
    n_bidirectional = np.sum(np.minimum(A, A.T))
    reciprocity = n_bidirectional / n_edges if n_edges > 0 else 0.0

    # Degree statistics
    out_degree = np.sum(A, axis=1)
    in_degree = np.sum(A, axis=0)

    return {
        'n_nodes': n,
        'n_edges': int(n_edges),
        'density': float(density),
        'reciprocity': float(reciprocity),
        'hierarchy_score': float(network.causal_hierarchy_score()),
        'mean_out_degree': float(np.mean(out_degree)),
        'max_out_degree': float(np.max(out_degree)),
        'mean_in_degree': float(np.mean(in_degree)),
        'max_in_degree': float(np.max(in_degree)),
        'n_drivers': len(network.get_drivers(threshold)),
        'n_sinks': len(network.get_sinks(threshold)),
        'n_hubs': len(network.get_hubs(threshold)),
        'n_feedback_loops': len(network.feedback_loops(threshold)),
    }


def build_network_from_te(
    te_matrix: np.ndarray,
    signal_names: List[str],
    p_matrix: np.ndarray = None
) -> CausalNetwork:
    """
    Build CausalNetwork from transfer entropy matrix.

    Parameters
    ----------
    te_matrix : array
        Transfer entropy matrix
    signal_names : list
        Signal names
    p_matrix : array, optional
        P-value matrix for significance

    Returns
    -------
    network : CausalNetwork
    """
    if p_matrix is None:
        p_matrix = np.ones_like(te_matrix)

    return CausalNetwork(
        nodes=signal_names,
        adjacency_matrix=te_matrix,
        edge_significance=p_matrix
    )


def network_distance(net1: CausalNetwork, net2: CausalNetwork, threshold: float = 0.1) -> float:
    """
    Compute distance between two causal networks.

    Uses Hamming distance on binarized adjacency matrices.

    Parameters
    ----------
    net1, net2 : CausalNetwork
    threshold : float

    Returns
    -------
    distance : float
        Normalized distance in [0, 1]
    """
    A1 = (net1.adjacency_matrix > threshold).astype(int)
    A2 = (net2.adjacency_matrix > threshold).astype(int)

    n = len(net1.nodes)
    max_diff = n * (n - 1)

    if max_diff == 0:
        return 0.0

    diff = np.sum(A1 != A2)
    return diff / max_diff

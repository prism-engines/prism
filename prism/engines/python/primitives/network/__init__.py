"""
Network Primitives (75-85)

Graph metrics from adjacency matrices.
"""

from .structure import (
    threshold_matrix,
    network_density,
    clustering_coefficient,
    connected_components,
    assortativity,
)

from .centrality import (
    centrality_degree,
    centrality_betweenness,
    centrality_eigenvector,
    centrality_closeness,
)

from .paths import (
    shortest_paths,
    average_path_length,
    diameter,
)

from .community import (
    modularity,
    community_detection,
)

__all__ = [
    # 75: Thresholding
    'threshold_matrix',
    # 76: Density
    'network_density',
    # 77: Clustering
    'clustering_coefficient',
    # 78: Modularity
    'modularity',
    # 79-82: Centrality
    'centrality_degree',
    'centrality_betweenness',
    'centrality_eigenvector',
    'centrality_closeness',
    # 83: Paths
    'shortest_paths',
    'average_path_length',
    'diameter',
    # 84: Components
    'connected_components',
    # 85: Community
    'community_detection',
    # Additional
    'assortativity',
]

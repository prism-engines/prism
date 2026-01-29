"""
PRISM Information Flow Engine

Computes causal discovery and information-theoretic metrics:
- Transfer entropy
- Granger causality
- Causal network analysis

Captures WHO DRIVES WHOM - directional information flow between signals.
"""

from .entropy import (
    shannon_entropy,
    mutual_information,
    kraskov_mutual_information,
)
from .transfer_entropy import (
    transfer_entropy,
    transfer_entropy_matrix,
)
from .granger import (
    granger_causality,
    granger_causality_matrix,
)
from .network import (
    CausalNetwork,
    network_metrics,
)
from .engine import (
    InformationEngine,
    InformationResult,
    compute_information_flow,
    compute_information_flow_for_all_entities,
)

__all__ = [
    # Entropy
    'shannon_entropy',
    'mutual_information',
    'kraskov_mutual_information',
    # Transfer entropy
    'transfer_entropy',
    'transfer_entropy_matrix',
    # Granger
    'granger_causality',
    'granger_causality_matrix',
    # Network
    'CausalNetwork',
    'network_metrics',
    # Engine
    'InformationEngine',
    'InformationResult',
    'compute_information_flow',
    'compute_information_flow_for_all_entities',
]

"""
Information Primitives (108-116)

Entropy, mutual information, transfer entropy, partial information decomposition.
"""

from .entropy import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy,
    joint_entropy,
    conditional_entropy,
)

from .divergence import (
    cross_entropy,
    kl_divergence,
    js_divergence,
)

from .mutual import (
    mutual_information,
    conditional_mutual_information,
    multivariate_mutual_information,
    total_correlation,
    interaction_information,
)

from .transfer import (
    transfer_entropy,
    transfer_entropy_effective,
    net_transfer_entropy,
)

from .decomposition import (
    partial_information_decomposition,
    redundancy,
    synergy,
    information_atoms,
)

__all__ = [
    # 108: Entropy
    'shannon_entropy',
    'renyi_entropy',
    'tsallis_entropy',
    # 109-110: Joint/Conditional entropy
    'joint_entropy',
    'conditional_entropy',
    # 111-112: Mutual information
    'mutual_information',
    'conditional_mutual_information',
    'multivariate_mutual_information',
    'total_correlation',
    'interaction_information',
    # 113: Transfer entropy
    'transfer_entropy',
    'transfer_entropy_effective',
    'net_transfer_entropy',
    # 114-116: Partial information decomposition
    'partial_information_decomposition',
    'redundancy',
    'synergy',
    'information_atoms',
    # Divergences
    'cross_entropy',
    'kl_divergence',
    'js_divergence',
]

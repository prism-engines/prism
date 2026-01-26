"""
Information Axis Engines
========================

Computation engines for the Information axis:
- permutation_entropy: Bandt-Pompe permutation entropy
- sample_entropy: Sample entropy (SampEn)
- entropy_rate: Change rate of entropy
"""

from .permutation_entropy import compute as compute_permutation_entropy
from .sample_entropy import compute as compute_sample_entropy
from .entropy_rate import compute as compute_entropy_rate

__all__ = [
    'compute_permutation_entropy',
    'compute_sample_entropy',
    'compute_entropy_rate',
]

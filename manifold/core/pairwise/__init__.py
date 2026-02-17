"""
Pairwise Engines.

Signal-to-signal computations.
"""

from . import correlation
from . import causality
from . import distance
from . import divergence

__all__ = ['correlation', 'causality', 'distance', 'divergence']

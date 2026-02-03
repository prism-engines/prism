"""
Advanced Engines (PR #13)

Four engines for advanced analysis:
1. CausalityEngine - Causal network analysis via Granger/TE
2. TopologyEngine - Persistent homology and topological data analysis
3. EmergenceEngine - Synergy, redundancy via Partial Information Decomposition
4. IntegrationEngine - Unified health assessment combining all metrics

This completes the PRISM build.
"""

from .causality_engine import CausalityEngine, CausalityConfig, run_causality_engine
from .topology_engine import TopologyEngine, TopologyConfig, run_topology_engine
from .emergence_engine import EmergenceEngine, EmergenceConfig, run_emergence_engine
from .integration_engine import IntegrationEngine, IntegrationConfig, run_integration_engine

__all__ = [
    # Engines
    'CausalityEngine',
    'TopologyEngine',
    'EmergenceEngine',
    'IntegrationEngine',
    # Configs
    'CausalityConfig',
    'TopologyConfig',
    'EmergenceConfig',
    'IntegrationConfig',
    # Runner functions
    'run_causality_engine',
    'run_topology_engine',
    'run_emergence_engine',
    'run_integration_engine',
]

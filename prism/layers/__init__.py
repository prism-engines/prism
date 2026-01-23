"""
PRISM Analytical Layers
=======================

Pure orchestrators that call engines and produce meaning.
Contains ZERO computation - all computation lives in engines/.

ORTHON Four-Layer Framework:
    1. Signal Typology: What is it?
    2. Behavioral Geometry: How does it behave?
    3. Dynamical Systems: When/how does it change?
    4. Causal Mechanics: Why does it change?

Rule: If you see `np.` or `scipy.` in a layer file, STOP.
      That computation belongs in an engine.
"""

from .signal_typology import SignalTypologyLayer
from .causal_mechanics import (
    CausalMechanicsLayer,
    CausalMechanicsOutput,
    MechanicsVector,
    MechanicsTypology,
    analyze_mechanics,
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
    # Backwards compatibility
    SystemPhysicsLayer,
    SystemPhysicsOutput,
    PhysicsVector,
    PhysicsTypology,
    analyze_physics,
)
from .behavioral_geometry import (
    BehavioralGeometryLayer,
    BehavioralGeometryOutput,
    GeometryVector,
    GeometryTypology,
    analyze_geometry,
    TopologyClass,
    StabilityClass,
    LeadershipClass
)

__all__ = [
    # Signal Typology
    'SignalTypologyLayer',

    # Causal Mechanics (new names)
    'CausalMechanicsLayer',
    'CausalMechanicsOutput',
    'MechanicsVector',
    'MechanicsTypology',
    'analyze_mechanics',
    'EnergyClass',
    'EquilibriumClass',
    'FlowClass',
    'OrbitClass',
    'DominanceClass',

    # Causal Mechanics (backwards compatibility)
    'SystemPhysicsLayer',
    'SystemPhysicsOutput',
    'PhysicsVector',
    'PhysicsTypology',
    'analyze_physics',

    # Behavioral Geometry
    'BehavioralGeometryLayer',
    'BehavioralGeometryOutput',
    'GeometryVector',
    'GeometryTypology',
    'analyze_geometry',
    'TopologyClass',
    'StabilityClass',
    'LeadershipClass',
]

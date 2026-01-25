"""
Fluid Dynamics Domain — Incompressible Viscous Flow Analysis

Physics engines for analyzing velocity fields, validated against the
classic Ghia et al. (1982) lid-driven cavity benchmark.

Level 4 Analysis: Requires spatial velocity field data u(x,y,t), v(x,y,t).

Capabilities:
    - VORTICITY: Local rotation rate (ω = ∂v/∂x - ∂u/∂y)
    - DIVERGENCE: Mass conservation check (∇·u = 0 for incompressible)
    - Q_CRITERION: Vortex identification (coming soon)
    - TURBULENT_KE: Turbulent kinetic energy (coming soon)

Usage:
    from domains.fluid import vorticity, divergence
    from domains.fluid.benchmark_ghia import get_u_profile, validate_against_ghia

    # Compute vorticity from velocity field
    result = vorticity.compute_2d(u, v, dx, dy)
    print(f"Max vorticity: {result.max_vorticity}")

    # Check mass conservation
    div_result = divergence.compute_2d(u, v, dx, dy)
    print(f"Continuity satisfied: {div_result.continuity_satisfied}")

    # Validate against Ghia benchmark
    y_ref, u_ref = get_u_profile(Re=100)
    validation = validate_against_ghia(u_computed, y_computed, Re=100)
    print(f"Benchmark passed: {validation['passed']}")

References:
    - Ghia et al. (1982) - THE lid-driven cavity benchmark (10,000+ citations)
    - Batchelor, "An Introduction to Fluid Dynamics"
"""

from . import vorticity
from . import divergence
from . import benchmark_ghia

# Engine registry
ENGINES = {
    'vorticity': vorticity.compute_2d,
    'divergence': divergence.compute_2d,
    'divergence_3d': divergence.compute_3d,
}

# Capability mapping
CAPABILITIES = {
    'VORTICITY': vorticity,
    'DIVERGENCE': divergence,
}

# Requirements for each capability
REQUIREMENTS = {
    'VORTICITY': {
        'signals': ['u', 'v'],
        'grid': True,
        'constants': [],
    },
    'DIVERGENCE': {
        'signals': ['u', 'v'],
        'grid': True,
        'constants': [],
    },
}

# Re-export benchmark functions
get_u_profile = benchmark_ghia.get_u_profile
get_v_profile = benchmark_ghia.get_v_profile
validate_against_ghia = benchmark_ghia.validate_against_ghia
GHIA_PRIMARY_VORTEX = benchmark_ghia.GHIA_PRIMARY_VORTEX

__all__ = [
    'vorticity',
    'divergence',
    'benchmark_ghia',
    'ENGINES',
    'CAPABILITIES',
    'REQUIREMENTS',
    'get_u_profile',
    'get_v_profile',
    'validate_against_ghia',
    'GHIA_PRIMARY_VORTEX',
]

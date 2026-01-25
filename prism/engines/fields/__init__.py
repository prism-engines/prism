"""
PRISM Fields Engines

Real Navier-Stokes field analysis. Not inspired by. The real equations.

    dv/dt + (v . nabla)v = -nabla(p)/rho + nu * nabla^2(v) + f

Computes:
    - Vorticity: omega = curl(v)
    - Strain rate tensor: S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)
    - Q-criterion, lambda_2 criterion for vortex identification
    - Turbulent kinetic energy, dissipation, enstrophy
    - Energy spectrum with Kolmogorov k^(-5/3) validation
    - Kolmogorov, Taylor, and integral length scales

References:
    Pope (2000) "Turbulent Flows"
    Tennekes & Lumley (1972) "A First Course in Turbulence"
    Kolmogorov (1941) "Local structure of turbulence"
"""

from prism.engines.fields.navier_stokes import (
    VelocityField,
    FlowRegime,
    compute_vorticity,
    compute_vorticity_magnitude,
    compute_strain_rate_tensor,
    compute_rotation_tensor,
    compute_Q_criterion,
    compute_lambda2_criterion,
    compute_turbulent_kinetic_energy,
    compute_dissipation_rate,
    compute_enstrophy,
    compute_helicity,
    compute_reynolds_stress_tensor,
    compute_energy_spectrum,
    compute_kolmogorov_scales,
    compute_taylor_microscale,
    compute_integral_length_scale,
    compute_reynolds_number,
    compute_taylor_reynolds_number,
    classify_flow_regime,
    analyze_velocity_field,
)

__all__ = [
    'VelocityField',
    'FlowRegime',
    'compute_vorticity',
    'compute_vorticity_magnitude',
    'compute_strain_rate_tensor',
    'compute_rotation_tensor',
    'compute_Q_criterion',
    'compute_lambda2_criterion',
    'compute_turbulent_kinetic_energy',
    'compute_dissipation_rate',
    'compute_enstrophy',
    'compute_helicity',
    'compute_reynolds_stress_tensor',
    'compute_energy_spectrum',
    'compute_kolmogorov_scales',
    'compute_taylor_microscale',
    'compute_integral_length_scale',
    'compute_reynolds_number',
    'compute_taylor_reynolds_number',
    'classify_flow_regime',
    'analyze_velocity_field',
]

"""
Membrane Separation Engines

Permeation flux, selectivity, concentration polarization, module design.
"""

import numpy as np
from typing import Dict, Any, Optional, List


def permeation_flux(P: float, delta_p: float, delta: float) -> Dict[str, Any]:
    """
    Permeation flux through membrane.

    J = P × ΔP / δ

    Parameters
    ----------
    P : float
        Permeability [mol/(m·s·Pa)] or [m³(STP)/(m²·s·Pa)]
    delta_p : float
        Pressure difference [Pa]
    delta : float
        Membrane thickness [m]

    Returns
    -------
    dict
        J: Permeation flux [mol/(m²·s)] or volumetric equivalent
        permeance: P/δ [mol/(m²·s·Pa)]
    """
    permeance = P / delta
    J = permeance * delta_p

    return {
        'J': float(J),
        'permeance': float(permeance),
        'P': P,
        'delta_p': delta_p,
        'delta': delta,
        'equation': 'J = P·ΔP/δ',
    }


def membrane_selectivity(P_A: float, P_B: float) -> Dict[str, Any]:
    """
    Membrane selectivity (ideal separation factor).

    α = P_A / P_B

    Parameters
    ----------
    P_A : float
        Permeability of component A (more permeable)
    P_B : float
        Permeability of component B (less permeable)

    Returns
    -------
    dict
        alpha: Ideal selectivity
        separation_quality: Qualitative assessment
    """
    alpha = P_A / P_B if P_B > 0 else float('inf')

    if alpha > 100:
        quality = "Excellent separation"
    elif alpha > 20:
        quality = "Good separation"
    elif alpha > 5:
        quality = "Moderate separation"
    else:
        quality = "Poor separation"

    return {
        'alpha': float(alpha),
        'P_A': P_A,
        'P_B': P_B,
        'separation_quality': quality,
        'equation': 'α = P_A/P_B',
    }


def concentration_polarization(J: float, k: float, C_b: float,
                               C_p: float = 0) -> Dict[str, Any]:
    """
    Concentration polarization at membrane surface.

    C_m/C_b = exp(J/k) + (C_p/C_b)(1 - exp(J/k))

    For complete rejection (C_p = 0):
    C_m/C_b = exp(J/k)

    Parameters
    ----------
    J : float
        Permeate flux [m/s]
    k : float
        Mass transfer coefficient [m/s]
    C_b : float
        Bulk concentration [mol/m³]
    C_p : float
        Permeate concentration [mol/m³] (default 0 for complete rejection)

    Returns
    -------
    dict
        C_m: Membrane surface concentration
        polarization_modulus: C_m/C_b
        flux_reduction: Estimated flux reduction due to polarization
    """
    exp_term = np.exp(J / k)

    if C_p == 0:
        C_m_over_C_b = exp_term
    else:
        C_m_over_C_b = exp_term + (C_p / C_b) * (1 - exp_term)

    C_m = C_b * C_m_over_C_b

    # Flux reduction estimate (osmotic pressure increases)
    flux_reduction = 1 - 1 / C_m_over_C_b if C_m_over_C_b > 1 else 0

    return {
        'C_m': float(C_m),
        'C_b': C_b,
        'polarization_modulus': float(C_m_over_C_b),
        'flux_reduction_estimate': float(flux_reduction),
        'J': J,
        'k': k,
        'equation': 'C_m/C_b = exp(J/k)',
    }


def rejection_coefficient(C_f: float, C_p: float) -> Dict[str, Any]:
    """
    Membrane rejection coefficient.

    R = 1 - C_p/C_f

    Parameters
    ----------
    C_f : float
        Feed concentration
    C_p : float
        Permeate concentration

    Returns
    -------
    dict
        R: Rejection coefficient (0-1)
        passage: 1 - R (fraction passing through)
    """
    R = 1 - C_p / C_f if C_f > 0 else 0
    passage = C_p / C_f if C_f > 0 else 1

    return {
        'R': float(R),
        'R_percent': float(R * 100),
        'passage': float(passage),
        'C_f': C_f,
        'C_p': C_p,
        'equation': 'R = 1 - C_p/C_f',
    }


def stage_cut(F_p: float, F_f: float) -> Dict[str, Any]:
    """
    Stage cut (permeate-to-feed ratio).

    θ = F_p / F_f

    Parameters
    ----------
    F_p : float
        Permeate flow rate
    F_f : float
        Feed flow rate

    Returns
    -------
    dict
        theta: Stage cut (0-1)
        retentate_fraction: 1 - θ
    """
    theta = F_p / F_f if F_f > 0 else 0

    return {
        'theta': float(theta),
        'theta_percent': float(theta * 100),
        'retentate_fraction': float(1 - theta),
        'F_p': F_p,
        'F_f': F_f,
        'equation': 'θ = F_p/F_f',
    }


def gas_separation(x_f: float, alpha: float, theta: float,
                   pressure_ratio: float) -> Dict[str, Any]:
    """
    Gas separation membrane performance.

    For binary mixture with ideal selectivity α:
    y_p/x_f = α × r / (1 + (α-1) × x_f × r)

    where r = pressure ratio (p_l/p_h)

    Parameters
    ----------
    x_f : float
        Feed mole fraction of fast component
    alpha : float
        Membrane selectivity
    theta : float
        Stage cut
    pressure_ratio : float
        Permeate/feed pressure ratio (< 1)

    Returns
    -------
    dict
        y_p: Permeate mole fraction
        x_r: Retentate mole fraction
        enrichment: y_p/x_f
    """
    r = pressure_ratio

    # Simplified model for cross-flow
    # At low stage cut, permeate composition approaches:
    y_p = alpha * x_f * r / (1 + (alpha - 1) * x_f * r)

    # Retentate from mass balance
    # x_f = theta * y_p + (1 - theta) * x_r
    x_r = (x_f - theta * y_p) / (1 - theta) if theta < 1 else 0

    enrichment = y_p / x_f if x_f > 0 else 0

    return {
        'y_p': float(y_p),
        'x_r': float(x_r),
        'enrichment': float(enrichment),
        'separation_factor': float(y_p * (1 - x_r) / (x_r * (1 - y_p))) if x_r > 0 and y_p < 1 else float('inf'),
        'x_f': x_f,
        'alpha': alpha,
        'theta': theta,
        'pressure_ratio': pressure_ratio,
    }


def spiral_wound(A: float, J: float, F_f: float, recovery: float = None) -> Dict[str, Any]:
    """
    Spiral wound module calculations.

    Parameters
    ----------
    A : float
        Membrane area [m²]
    J : float
        Flux [m³/(m²·s) or L/(m²·h)]
    F_f : float
        Feed flow rate [m³/s or L/h]
    recovery : float, optional
        Water recovery (permeate/feed)

    Returns
    -------
    dict
        F_p: Permeate flow rate
        F_r: Retentate flow rate
        recovery: If not provided, calculated
    """
    F_p = A * J

    if recovery is not None:
        # Given recovery, calculate required feed
        F_f_required = F_p / recovery
        F_r = F_f_required - F_p
    else:
        # Calculate recovery from flows
        recovery = F_p / F_f if F_f > 0 else 0
        F_r = F_f - F_p

    return {
        'F_p': float(F_p),
        'F_r': float(F_r),
        'recovery': float(recovery),
        'recovery_percent': float(recovery * 100),
        'A': A,
        'J': J,
        'F_f': float(F_f),
        'equation': 'F_p = A × J',
    }


def hollow_fiber(n_fibers: int, d_inner: float, L: float,
                 J: float) -> Dict[str, Any]:
    """
    Hollow fiber module calculations.

    A = n × π × d × L

    Parameters
    ----------
    n_fibers : int
        Number of fibers
    d_inner : float
        Inner fiber diameter [m]
    L : float
        Fiber length [m]
    J : float
        Flux [m³/(m²·s)]

    Returns
    -------
    dict
        A: Total membrane area [m²]
        F_p: Permeate flow rate [m³/s]
        packing_density: Typical for hollow fiber
    """
    A = n_fibers * np.pi * d_inner * L
    F_p = A * J

    # Typical packing density (membrane area per module volume)
    # Approximate module diameter assuming hexagonal packing
    d_outer = d_inner * 1.5  # Typical wall thickness
    module_area = n_fibers * (d_outer**2) * 0.9  # Packing factor
    module_diameter = np.sqrt(4 * module_area / np.pi)
    module_volume = np.pi * (module_diameter / 2)**2 * L
    packing_density = A / module_volume if module_volume > 0 else 0

    return {
        'A': float(A),
        'F_p': float(F_p),
        'n_fibers': n_fibers,
        'd_inner': d_inner,
        'L': L,
        'module_diameter_approx': float(module_diameter),
        'packing_density': float(packing_density),
        'equation': 'A = n·π·d·L',
    }


def osmotic_pressure(C: float, T: float = 298.15, i: float = 1) -> Dict[str, Any]:
    """
    Osmotic pressure (van't Hoff equation).

    π = i × C × R × T

    Parameters
    ----------
    C : float
        Molar concentration [mol/m³]
    T : float
        Temperature [K]
    i : float
        van't Hoff factor (1 for non-electrolytes)

    Returns
    -------
    dict
        pi: Osmotic pressure [Pa]
        pi_bar: Osmotic pressure [bar]
    """
    R = 8.314  # J/(mol·K)
    pi = i * C * R * T

    return {
        'pi': float(pi),
        'pi_bar': float(pi / 1e5),
        'pi_psi': float(pi / 6895),
        'C': C,
        'T': T,
        'i': i,
        'equation': 'π = iCRT',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for membrane calculations.
    """
    if all(k in kwargs for k in ['P', 'delta_p', 'delta']):
        return permeation_flux(kwargs['P'], kwargs['delta_p'], kwargs['delta'])

    if 'P_A' in kwargs and 'P_B' in kwargs:
        return membrane_selectivity(kwargs['P_A'], kwargs['P_B'])

    if 'C_f' in kwargs and 'C_p' in kwargs:
        return rejection_coefficient(kwargs['C_f'], kwargs['C_p'])

    return {'error': 'Insufficient parameters'}

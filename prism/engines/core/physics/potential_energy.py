"""
Potential Energy Engine — THE REAL EQUATIONS

Harmonic:      V = ½k(x - x₀)²  [J]
Gravitational: V = mgh          [J]
Elastic:       V = ½kx²         [J]

When constants known: Returns energy in Joules
When constants unknown: Returns specific/normalized form
"""

import numpy as np
from typing import Dict, Optional


def compute_potential_energy_harmonic(
    position: np.ndarray,
    equilibrium: float = 0.0,
    spring_constant: Optional[float] = None,
) -> Dict:
    """
    Harmonic oscillator potential: V = ½k(x - x₀)²

    Args:
        position: x [m] or displacement signal
        equilibrium: x₀ equilibrium position [m]
        spring_constant: k [N/m]. If None, returns V/k.

    Returns:
        Dict with potential energy
    """
    x = np.asarray(position, dtype=float)

    if np.all(np.isnan(x)):
        return {
            'potential_energy': None,
            'mean_potential_energy': None,
            'max_potential_energy': None,
            'displacement': None,
            'displacement_squared': None,
            'equilibrium_position': equilibrium,
            'spring_constant': spring_constant,
            'is_specific': spring_constant is None,
            'units': None,
            'equation': 'V = ½k(x-x₀)²' if spring_constant else 'V/k = ½(x-x₀)²',
        }

    displacement = x - equilibrium
    displacement_squared = displacement**2

    if spring_constant is not None:
        V = 0.5 * spring_constant * displacement_squared
        units = 'J'
        is_specific = False
    else:
        V = 0.5 * displacement_squared
        units = 'm²'  # V/k has units of m²
        is_specific = True

    return {
        'potential_energy': V,
        'mean_potential_energy': float(np.nanmean(V)),
        'max_potential_energy': float(np.nanmax(V)),

        'displacement': displacement,
        'displacement_squared': displacement_squared,
        'equilibrium_position': equilibrium,

        'spring_constant': spring_constant,
        'is_specific': is_specific,
        'units': units,
        'equation': 'V = ½k(x-x₀)²' if spring_constant else 'V/k = ½(x-x₀)²',
    }


def compute_potential_energy_gravitational(
    height: np.ndarray,
    mass: Optional[float] = None,
    g: float = 9.81,
    reference_height: float = 0.0,
) -> Dict:
    """
    Gravitational potential energy: V = mg(h - h₀)

    Args:
        height: h [m]
        mass: m [kg]. If None, returns V/m = g(h - h₀).
        g: Gravitational acceleration [m/s²]
        reference_height: h₀ [m]
    """
    h = np.asarray(height, dtype=float)

    if np.all(np.isnan(h)):
        return {
            'potential_energy': None,
            'mean_potential_energy': None,
            'height': None,
            'height_change': None,
            'reference_height': reference_height,
            'mass': mass,
            'g': g,
            'is_specific': mass is None,
            'units': None,
            'equation': 'V = mgh' if mass else 'V/m = gh',
        }

    delta_h = h - reference_height

    if mass is not None:
        V = mass * g * delta_h
        units = 'J'
        is_specific = False
    else:
        V = g * delta_h
        units = 'J/kg'
        is_specific = True

    return {
        'potential_energy': V,
        'mean_potential_energy': float(np.nanmean(V)),
        'max_potential_energy': float(np.nanmax(V)),
        'min_potential_energy': float(np.nanmin(V)),

        'height': h,
        'height_change': delta_h,
        'reference_height': reference_height,

        'mass': mass,
        'g': g,
        'is_specific': is_specific,
        'units': units,
        'equation': 'V = mgh' if mass else 'V/m = gh',
    }


def estimate_spring_constant(
    position: np.ndarray,
    force: np.ndarray,
) -> Dict:
    """
    Estimate spring constant from position-force data: F = -kx

    Uses linear regression on Hooke's law.
    """
    x = np.asarray(position, dtype=float)
    F = np.asarray(force, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(F))
    x = x[valid]
    F = F[valid]

    if len(x) < 3:
        return {
            'spring_constant': None,
            'units': 'N/m',
            'r_squared': None,
            'fit_quality': None,
            'equation': 'F = -kx (Hooke\'s Law)',
        }

    # Linear fit: F = -k*x + b
    # slope = -k
    coeffs = np.polyfit(x, F, 1)
    k = -coeffs[0]

    # R² for fit quality
    F_pred = np.polyval(coeffs, x)
    ss_res = np.sum((F - F_pred)**2)
    ss_tot = np.sum((F - np.mean(F))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r_squared > 0.9:
        fit_quality = 'good'
    elif r_squared > 0.7:
        fit_quality = 'moderate'
    else:
        fit_quality = 'poor'

    return {
        'spring_constant': float(k),
        'units': 'N/m',
        'r_squared': float(r_squared),
        'fit_quality': fit_quality,
        'intercept': float(coeffs[1]),
        'equation': 'F = -kx (Hooke\'s Law)',
    }


def compute(
    values: np.ndarray,
    equilibrium: float = 0.0,
    spring_constant: Optional[float] = None,
    potential_type: str = 'harmonic',
    mass: Optional[float] = None,
    g: float = 9.81,
) -> Dict:
    """
    Main compute function for potential energy.

    Args:
        values: Position array [m] or height array [m]
        equilibrium: Equilibrium position [m] (for harmonic)
        spring_constant: k [N/m] (for harmonic). If None, returns specific.
        potential_type: 'harmonic' or 'gravitational'
        mass: Mass [kg] (for gravitational). If None, returns specific.
        g: Gravitational acceleration [m/s²]

    Returns:
        Dict with potential energy metrics
    """
    values = np.asarray(values, dtype=float)

    if potential_type == 'gravitational':
        return compute_potential_energy_gravitational(
            height=values,
            mass=mass,
            g=g,
            reference_height=equilibrium,
        )
    else:
        return compute_potential_energy_harmonic(
            position=values,
            equilibrium=equilibrium,
            spring_constant=spring_constant,
        )

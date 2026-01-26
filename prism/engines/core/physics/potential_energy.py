"""
Potential Energy Engine — THE REAL EQUATIONS

Harmonic:      V = ½k(x - x₀)²  [J]
Gravitational: V = mgh          [J]
Elastic:       V = ½kx²         [J]

REQUIRES: spring_constant [N/m] for harmonic, mass [kg] for gravitational
"""

import numpy as np
from typing import Dict, Optional, Any

from prism.engines.validation import get_constant


def compute_potential_energy_harmonic(
    position: np.ndarray,
    equilibrium: float = 0.0,
    spring_constant: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Harmonic oscillator potential: V = ½k(x - x₀)²

    REQUIRES: spring_constant [N/m]

    Args:
        position: x [m] or displacement signal
        equilibrium: x₀ equilibrium position [m]
        spring_constant: k [N/m]. REQUIRED.
        config: Optional config dict

    Returns:
        Dict with potential energy
    """
    # Get spring_constant from config if not provided
    if spring_constant is None and config is not None:
        spring_constant = get_constant(config, 'spring_constant')

    # VALIDATION: spring_constant MUST exist
    if spring_constant is None or np.isnan(spring_constant):
        return {
            'potential_energy': float('nan'),
            'mean_potential_energy': float('nan'),
            'max_potential_energy': float('nan'),
            'error': 'Missing required constant: spring_constant [N/m]',
            'equation': 'V = ½k(x-x₀)²',
        }

    x = np.asarray(position, dtype=float)

    if np.all(np.isnan(x)):
        return {
            'potential_energy': float('nan'),
            'mean_potential_energy': float('nan'),
            'max_potential_energy': float('nan'),
            'error': 'Invalid position data (all NaN)',
            'spring_constant': spring_constant,
            'equation': 'V = ½k(x-x₀)²',
        }

    displacement = x - equilibrium
    displacement_squared = displacement**2
    V = 0.5 * spring_constant * displacement_squared

    return {
        'potential_energy': V,
        'mean_potential_energy': float(np.nanmean(V)),
        'max_potential_energy': float(np.nanmax(V)),
        'displacement': displacement,
        'displacement_squared': displacement_squared,
        'equilibrium_position': equilibrium,
        'spring_constant': spring_constant,
        'units': 'J',
        'equation': 'V = ½k(x-x₀)²',
    }


def compute_potential_energy_gravitational(
    height: np.ndarray,
    mass: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    g: float = 9.81,
    reference_height: float = 0.0,
) -> Dict:
    """
    Gravitational potential energy: V = mg(h - h₀)

    REQUIRES: mass [kg]

    Args:
        height: h [m]
        mass: m [kg]. REQUIRED.
        config: Optional config dict
        g: Gravitational acceleration [m/s²]
        reference_height: h₀ [m]
    """
    # Get mass from config if not provided
    if mass is None and config is not None:
        mass = get_constant(config, 'mass')

    # VALIDATION: mass MUST exist
    if mass is None or np.isnan(mass):
        return {
            'potential_energy': float('nan'),
            'mean_potential_energy': float('nan'),
            'max_potential_energy': float('nan'),
            'min_potential_energy': float('nan'),
            'error': 'Missing required constant: mass [kg]',
            'equation': 'V = mgh',
        }

    h = np.asarray(height, dtype=float)

    if np.all(np.isnan(h)):
        return {
            'potential_energy': float('nan'),
            'mean_potential_energy': float('nan'),
            'max_potential_energy': float('nan'),
            'min_potential_energy': float('nan'),
            'error': 'Invalid height data (all NaN)',
            'mass': mass,
            'g': g,
            'equation': 'V = mgh',
        }

    delta_h = h - reference_height
    V = mass * g * delta_h

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
        'units': 'J',
        'equation': 'V = mgh',
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
    config: Optional[Dict[str, Any]] = None,
    g: float = 9.81,
) -> Dict:
    """
    Main compute function for potential energy.

    REQUIRES: spring_constant [N/m] for harmonic, mass [kg] for gravitational

    Args:
        values: Position array [m] or height array [m]
        equilibrium: Equilibrium position [m] (for harmonic)
        spring_constant: k [N/m] (for harmonic). REQUIRED for harmonic.
        potential_type: 'harmonic' or 'gravitational'
        mass: Mass [kg] (for gravitational). REQUIRED for gravitational.
        config: Optional config dict
        g: Gravitational acceleration [m/s²]

    Returns:
        Dict with potential energy metrics
    """
    values = np.asarray(values, dtype=float)

    if potential_type == 'gravitational':
        return compute_potential_energy_gravitational(
            height=values,
            mass=mass,
            config=config,
            g=g,
            reference_height=equilibrium,
        )
    else:
        return compute_potential_energy_harmonic(
            position=values,
            equilibrium=equilibrium,
            spring_constant=spring_constant,
            config=config,
        )

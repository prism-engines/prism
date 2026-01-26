"""
Kinetic Energy Engine — THE REAL EQUATION

T = ½mv²  [J]

REQUIRES: mass [kg]

If mass not provided, returns NaN. No silent fallbacks.
"""

import numpy as np
from typing import Dict, Optional, Any

from prism.engines.validation import validate_or_nan, get_constant


def compute_kinetic_energy(
    velocity: np.ndarray,
    mass: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    velocity_is_derivative: bool = False,
    position: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Dict:
    """
    Compute kinetic energy: T = ½mv²

    REQUIRES: mass [kg]

    Args:
        velocity: Velocity array [m/s] or signal to differentiate
        mass: Mass [kg]. REQUIRED - returns NaN if missing.
        config: Optional config dict with 'constants' or 'global_constants'
        velocity_is_derivative: If True, velocity is already dx/dt
        position: Position array to differentiate if velocity not provided
        dt: Time step [s] for differentiation

    Returns:
        Dict with kinetic energy and metadata
    """
    # Get mass from config if not provided directly
    if mass is None and config is not None:
        mass = get_constant(config, 'mass')

    # VALIDATION: mass MUST exist
    if mass is None or np.isnan(mass):
        return {
            'kinetic_energy': float('nan'),
            'mean_kinetic_energy': float('nan'),
            'max_kinetic_energy': float('nan'),
            'total_kinetic_energy': float('nan'),
            'error': 'Missing required constant: mass [kg]',
            'equation': 'T = ½mv²',
        }

    # Get velocity
    if position is not None and not velocity_is_derivative:
        v = np.gradient(position, dt, axis=0)
    else:
        v = np.asarray(velocity, dtype=float)

    # Handle NaN velocity
    if np.all(np.isnan(v)):
        return {
            'kinetic_energy': float('nan'),
            'mean_kinetic_energy': float('nan'),
            'max_kinetic_energy': float('nan'),
            'total_kinetic_energy': float('nan'),
            'error': 'Invalid velocity data (all NaN)',
            'mass': mass,
            'equation': 'T = ½mv²',
        }

    # Handle multidimensional (v could be vector)
    if v.ndim > 1:
        v_squared = np.sum(v**2, axis=-1)
    else:
        v_squared = v**2

    # Compute REAL kinetic energy with known mass
    T = 0.5 * mass * v_squared

    return {
        'kinetic_energy': T,
        'mean_kinetic_energy': float(np.nanmean(T)),
        'max_kinetic_energy': float(np.nanmax(T)),
        'total_kinetic_energy': float(np.nansum(T) * dt),

        'velocity': v,
        'velocity_squared': v_squared,

        'mass': mass,
        'units': 'J',
        'equation': 'T = ½mv²',
    }


def compute_kinetic_energy_rotational(
    angular_velocity: np.ndarray,
    moment_of_inertia: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Rotational kinetic energy: T_rot = ½Iω²

    REQUIRES: moment_of_inertia [kg·m²]

    Args:
        angular_velocity: ω [rad/s]
        moment_of_inertia: I [kg·m²]. REQUIRED.
        config: Optional config dict
    """
    # Get from config if not provided
    if moment_of_inertia is None and config is not None:
        moment_of_inertia = get_constant(config, 'moment_of_inertia')

    # VALIDATION: moment_of_inertia MUST exist
    if moment_of_inertia is None or np.isnan(moment_of_inertia):
        return {
            'rotational_kinetic_energy': float('nan'),
            'mean': float('nan'),
            'max': float('nan'),
            'error': 'Missing required constant: moment_of_inertia [kg·m²]',
            'equation': 'T = ½Iω²',
        }

    omega = np.asarray(angular_velocity, dtype=float)

    if np.all(np.isnan(omega)):
        return {
            'rotational_kinetic_energy': float('nan'),
            'mean': float('nan'),
            'max': float('nan'),
            'error': 'Invalid angular velocity data (all NaN)',
            'moment_of_inertia': moment_of_inertia,
            'equation': 'T = ½Iω²',
        }

    omega_squared = omega**2
    T_rot = 0.5 * moment_of_inertia * omega_squared

    return {
        'rotational_kinetic_energy': T_rot,
        'mean': float(np.nanmean(T_rot)),
        'max': float(np.nanmax(T_rot)),
        'moment_of_inertia': moment_of_inertia,
        'units': 'J',
        'equation': 'T = ½Iω²',
    }


def compute(
    values: np.ndarray,
    mass: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    dt: float = 1.0,
    mode: str = 'velocity',
) -> Dict:
    """
    Main compute function for kinetic energy.

    REQUIRES: mass [kg]

    Args:
        values: Input array - velocity [m/s] or position [m]
        mass: Mass [kg]. REQUIRED.
        config: Optional config dict with constants
        dt: Time step [s]
        mode: 'velocity' if values are velocities, 'position' if positions

    Returns:
        Dict with kinetic energy metrics
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)] if values.ndim == 1 else values

    if len(values) < 2:
        return {
            'kinetic_energy': float('nan'),
            'mean_kinetic_energy': float('nan'),
            'max_kinetic_energy': float('nan'),
            'error': 'Insufficient data (need at least 2 points)',
            'equation': 'T = ½mv²',
        }

    if mode == 'position':
        return compute_kinetic_energy(
            velocity=None,
            mass=mass,
            config=config,
            position=values,
            dt=dt,
        )
    else:
        return compute_kinetic_energy(
            velocity=values,
            mass=mass,
            config=config,
        )

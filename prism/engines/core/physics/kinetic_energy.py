"""
Kinetic Energy Engine — THE REAL EQUATION

T = ½mv²  [J]

When mass is known: Returns energy in Joules
When mass unknown: Returns specific kinetic energy T/m = ½v² [J/kg]

The computation is always real. The interpretation depends on available data.
"""

import numpy as np
from typing import Dict, Optional


def compute_kinetic_energy(
    velocity: np.ndarray,
    mass: Optional[float] = None,
    velocity_is_derivative: bool = False,
    position: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Dict:
    """
    Compute kinetic energy: T = ½mv²

    Args:
        velocity: Velocity array [m/s] or signal to differentiate
        mass: Mass [kg]. If None, returns specific KE (per unit mass)
        velocity_is_derivative: If True, velocity is already dx/dt
        position: Position array to differentiate if velocity not provided
        dt: Time step [s] for differentiation

    Returns:
        Dict with kinetic energy and metadata
    """
    # Get velocity
    if position is not None and not velocity_is_derivative:
        v = np.gradient(position, dt, axis=0)
    else:
        v = np.asarray(velocity, dtype=float)

    # Handle NaN
    if np.all(np.isnan(v)):
        return {
            'kinetic_energy': None,
            'mean_kinetic_energy': None,
            'max_kinetic_energy': None,
            'total_kinetic_energy': None,
            'velocity': None,
            'velocity_squared': None,
            'mass': mass,
            'is_specific': mass is None,
            'units': None,
            'equation': 'T = ½mv²' if mass else 'T/m = ½v²',
        }

    # Handle multidimensional (v could be vector)
    if v.ndim > 1:
        v_squared = np.sum(v**2, axis=-1)
    else:
        v_squared = v**2

    # Compute kinetic energy
    if mass is not None:
        # REAL kinetic energy with known mass
        T = 0.5 * mass * v_squared
        units = 'J'
        is_specific = False
    else:
        # Specific kinetic energy (per unit mass)
        T = 0.5 * v_squared
        units = 'J/kg'
        is_specific = True

    return {
        'kinetic_energy': T,
        'mean_kinetic_energy': float(np.nanmean(T)),
        'max_kinetic_energy': float(np.nanmax(T)),
        'total_kinetic_energy': float(np.nansum(T) * dt),  # Integrated over time

        'velocity': v,
        'velocity_squared': v_squared,

        'mass': mass,
        'is_specific': is_specific,  # True if per-unit-mass
        'units': units,

        # Honest about what we computed
        'equation': 'T = ½mv²' if mass else 'T/m = ½v²',
    }


def compute_kinetic_energy_rotational(
    angular_velocity: np.ndarray,
    moment_of_inertia: Optional[float] = None,
) -> Dict:
    """
    Rotational kinetic energy: T_rot = ½Iω²

    Args:
        angular_velocity: ω [rad/s]
        moment_of_inertia: I [kg·m²]. If None, returns T/I.
    """
    omega = np.asarray(angular_velocity, dtype=float)

    if np.all(np.isnan(omega)):
        return {
            'rotational_kinetic_energy': None,
            'mean': None,
            'moment_of_inertia': moment_of_inertia,
            'is_specific': moment_of_inertia is None,
            'units': None,
            'equation': 'T = ½Iω²' if moment_of_inertia else 'T/I = ½ω²',
        }

    omega_squared = omega**2

    if moment_of_inertia is not None:
        T_rot = 0.5 * moment_of_inertia * omega_squared
        units = 'J'
        is_specific = False
    else:
        T_rot = 0.5 * omega_squared
        units = 'J/(kg·m²)'
        is_specific = True

    return {
        'rotational_kinetic_energy': T_rot,
        'mean': float(np.nanmean(T_rot)),
        'max': float(np.nanmax(T_rot)),
        'moment_of_inertia': moment_of_inertia,
        'is_specific': is_specific,
        'units': units,
        'equation': 'T = ½Iω²' if moment_of_inertia else 'T/I = ½ω²',
    }


def compute(
    values: np.ndarray,
    mass: Optional[float] = None,
    dt: float = 1.0,
    mode: str = 'velocity',
) -> Dict:
    """
    Main compute function for kinetic energy.

    Args:
        values: Input array - velocity [m/s] or position [m]
        mass: Mass [kg]. If None, returns specific KE
        dt: Time step [s]
        mode: 'velocity' if values are velocities, 'position' if positions

    Returns:
        Dict with kinetic energy metrics
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)] if values.ndim == 1 else values

    if len(values) < 2:
        return {
            'kinetic_energy': None,
            'mean_kinetic_energy': None,
            'max_kinetic_energy': None,
            'mass': mass,
            'is_specific': mass is None,
            'units': None,
            'equation': 'T = ½mv²',
        }

    if mode == 'position':
        return compute_kinetic_energy(
            velocity=None,
            mass=mass,
            position=values,
            dt=dt,
        )
    else:
        return compute_kinetic_energy(
            velocity=values,
            mass=mass,
        )

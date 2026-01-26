"""
Hamiltonian Mechanics Engine — THE REAL EQUATION

H(q, p, t) = T(p) + V(q)  [J]

REQUIRES: mass [kg], spring_constant [N/m] (for harmonic potential)

Total mechanical energy. Conserved in closed systems.
"""

import numpy as np
from typing import Dict, Optional, Callable, Any

from prism.engines.validation import get_constant


def compute_hamiltonian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    spring_constant: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    potential_func: Optional[Callable] = None,
    equilibrium: float = 0.0,
) -> Dict:
    """
    Compute Hamiltonian: H = T + V (total mechanical energy)

    REQUIRES: mass [kg], spring_constant [N/m]

    Args:
        position: q [m]
        velocity: v = dq/dt [m/s]
        mass: m [kg]. REQUIRED.
        spring_constant: k [N/m]. REQUIRED (unless potential_func provided).
        config: Optional config dict
        potential_func: V(q) function returning potential energy
        equilibrium: x₀ for harmonic potential

    Returns:
        Dict with Hamiltonian and components
    """
    # Get constants from config if not provided
    if mass is None and config is not None:
        mass = get_constant(config, 'mass')
    if spring_constant is None and config is not None:
        spring_constant = get_constant(config, 'spring_constant')

    # VALIDATION: mass MUST exist
    if mass is None or np.isnan(mass):
        return {
            'hamiltonian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'momentum': float('nan'),
            'mean_H': float('nan'),
            'energy_conserved': False,
            'error': 'Missing required constant: mass [kg]',
            'equation': 'H = T + V = ½mv² + V(q)',
        }

    # VALIDATION: spring_constant MUST exist (unless potential_func provided)
    if potential_func is None and (spring_constant is None or np.isnan(spring_constant)):
        return {
            'hamiltonian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'momentum': float('nan'),
            'mean_H': float('nan'),
            'energy_conserved': False,
            'error': 'Missing required constant: spring_constant [N/m]',
            'equation': 'H = T + V = ½mv² + V(q)',
        }

    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)

    # Validate inputs
    if np.all(np.isnan(q)) or np.all(np.isnan(v)):
        return {
            'hamiltonian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'momentum': float('nan'),
            'mean_H': float('nan'),
            'energy_conserved': False,
            'error': 'Invalid position/velocity data (all NaN)',
            'mass': mass,
            'spring_constant': spring_constant,
            'equation': 'H = T + V = ½mv² + V(q)',
        }

    # Handle vector quantities
    if v.ndim > 1:
        v_squared = np.sum(v**2, axis=-1)
    else:
        v_squared = v**2

    # Kinetic energy
    T = 0.5 * mass * v_squared
    p = mass * v  # Momentum

    # Potential energy
    if potential_func is not None:
        V = potential_func(q)
    else:
        displacement = q - equilibrium
        if q.ndim > 1:
            disp_sq = np.sum(displacement**2, axis=-1)
        else:
            disp_sq = displacement**2
        V = 0.5 * spring_constant * disp_sq

    # Hamiltonian
    H = T + V

    # Check conservation
    H_mean = np.nanmean(H)
    H_std = np.nanstd(H)
    is_conserved = (H_std / np.abs(H_mean)) < 0.01 if H_mean != 0 else H_std < 0.01

    return {
        'hamiltonian': H,
        'kinetic_energy': T,
        'potential_energy': V,
        'momentum': p,

        'mean_H': float(H_mean),
        'std_H': float(H_std),
        'min_H': float(np.nanmin(H)),
        'max_H': float(np.nanmax(H)),

        'T_fraction': float(np.nanmean(T) / H_mean) if H_mean != 0 else 0.0,
        'V_fraction': float(np.nanmean(V) / H_mean) if H_mean != 0 else 0.0,

        'energy_conserved': bool(is_conserved),
        'conservation_error': float(H_std / np.abs(H_mean)) if H_mean != 0 else float(H_std),

        'mass': mass,
        'spring_constant': spring_constant,
        'units': 'J',
        'equation': 'H = T + V = ½mv² + V(q)',
    }


def compute_hamiltons_equations(
    position: np.ndarray,
    momentum: np.ndarray,
    dt: float,
    mass: float,
    dV_dq: Optional[np.ndarray] = None,
    spring_constant: Optional[float] = None,
) -> Dict:
    """
    Verify Hamilton's equations of motion:

    dq/dt = ∂H/∂p = p/m
    dp/dt = -∂H/∂q = -dV/dq

    Args:
        position: q(t)
        momentum: p(t)
        dt: Time step
        mass: m [kg]
        dV_dq: Potential gradient (computed if not provided)
        spring_constant: k for harmonic potential
    """
    q = np.asarray(position, dtype=float)
    p = np.asarray(momentum, dtype=float)

    if len(q) < 3:
        return {
            'dq_dt_error': None,
            'hamiltons_eq1_satisfied': None,
            'error': 'Insufficient data points',
        }

    # Numerical derivatives
    dq_dt = np.gradient(q, dt, axis=0)
    dp_dt = np.gradient(p, dt, axis=0)

    # Hamilton's equations predictions
    dq_dt_predicted = p / mass  # ∂H/∂p

    if dV_dq is not None:
        dp_dt_predicted = -dV_dq  # -∂H/∂q
    elif spring_constant is not None:
        dp_dt_predicted = -spring_constant * q  # -kq for harmonic
    else:
        dp_dt_predicted = None

    # Check how well equations are satisfied
    dq_error = np.sqrt(np.nanmean((dq_dt - dq_dt_predicted)**2))
    dq_scale = np.nanstd(dq_dt)

    result = {
        'dq_dt_actual': dq_dt,
        'dq_dt_predicted': dq_dt_predicted,
        'dq_dt_error': float(dq_error),
        'hamiltons_eq1_satisfied': bool(dq_error < 0.1 * dq_scale) if dq_scale > 0 else True,
    }

    if dp_dt_predicted is not None:
        dp_error = np.sqrt(np.nanmean((dp_dt - dp_dt_predicted)**2))
        dp_scale = np.nanstd(dp_dt)
        result['dp_dt_actual'] = dp_dt
        result['dp_dt_predicted'] = dp_dt_predicted
        result['dp_dt_error'] = float(dp_error)
        result['hamiltons_eq2_satisfied'] = bool(dp_error < 0.1 * dp_scale) if dp_scale > 0 else True

    return result


def compute(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    spring_constant: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    equilibrium: float = 0.0,
) -> Dict:
    """
    Main compute function for Hamiltonian.

    REQUIRES: mass [kg], spring_constant [N/m]

    Args:
        position: q [m]
        velocity: v [m/s]
        mass: m [kg]. REQUIRED.
        spring_constant: k [N/m]. REQUIRED.
        config: Optional config dict
        equilibrium: x₀ [m]

    Returns:
        Dict with Hamiltonian metrics
    """
    return compute_hamiltonian(
        position=position,
        velocity=velocity,
        mass=mass,
        spring_constant=spring_constant,
        config=config,
        equilibrium=equilibrium,
    )

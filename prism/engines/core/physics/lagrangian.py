"""
Lagrangian Mechanics Engine — THE REAL EQUATION

L(q, q̇, t) = T - V  [J]

REQUIRES: mass [kg], spring_constant [N/m]

The action integral S = ∫L dt is stationary for true paths.
Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0
"""

import numpy as np
from typing import Dict, Optional, Any

from prism.engines.validation import get_constant


def compute_lagrangian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    spring_constant: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    equilibrium: float = 0.0,
) -> Dict:
    """
    Compute Lagrangian: L = T - V

    REQUIRES: mass [kg], spring_constant [N/m]

    Args:
        position: Generalized coordinate q
        velocity: Generalized velocity q̇ = dq/dt
        mass: m [kg]. REQUIRED.
        spring_constant: k [N/m]. REQUIRED.
        config: Optional config dict
        equilibrium: q₀ equilibrium position

    Returns:
        Dict with Lagrangian and components
    """
    # Get constants from config if not provided
    if mass is None and config is not None:
        mass = get_constant(config, 'mass')
    if spring_constant is None and config is not None:
        spring_constant = get_constant(config, 'spring_constant')

    # VALIDATION: mass MUST exist
    if mass is None or np.isnan(mass):
        return {
            'lagrangian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'mean_L': float('nan'),
            'error': 'Missing required constant: mass [kg]',
            'equation': 'L = T - V = ½mq̇² - V(q)',
        }

    # VALIDATION: spring_constant MUST exist
    if spring_constant is None or np.isnan(spring_constant):
        return {
            'lagrangian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'mean_L': float('nan'),
            'error': 'Missing required constant: spring_constant [N/m]',
            'equation': 'L = T - V = ½mq̇² - V(q)',
        }

    q = np.asarray(position, dtype=float)
    q_dot = np.asarray(velocity, dtype=float)

    # Validate inputs
    if np.all(np.isnan(q)) or np.all(np.isnan(q_dot)):
        return {
            'lagrangian': float('nan'),
            'kinetic_energy': float('nan'),
            'potential_energy': float('nan'),
            'mean_L': float('nan'),
            'error': 'Invalid position/velocity data (all NaN)',
            'mass': mass,
            'spring_constant': spring_constant,
            'equation': 'L = T - V = ½mq̇² - V(q)',
        }

    # Kinetic energy
    if q_dot.ndim > 1:
        v_squared = np.sum(q_dot**2, axis=-1)
    else:
        v_squared = q_dot**2

    T = 0.5 * mass * v_squared

    # Potential energy
    displacement = q - equilibrium
    if q.ndim > 1:
        disp_sq = np.sum(displacement**2, axis=-1)
    else:
        disp_sq = displacement**2

    V = 0.5 * spring_constant * disp_sq

    # Lagrangian
    L = T - V

    mean_T = float(np.nanmean(T))
    mean_V = float(np.nanmean(V))

    return {
        'lagrangian': L,
        'kinetic_energy': T,
        'potential_energy': V,

        'mean_L': float(np.nanmean(L)),
        'mean_T': mean_T,
        'mean_V': mean_V,

        # Sign of L indicates energy balance
        'T_dominated': mean_T > mean_V,
        'V_dominated': mean_V > mean_T,

        'mass': mass,
        'spring_constant': spring_constant,
        'units': 'J',
        'equation': 'L = T - V = ½mq̇² - V(q)',
    }


def compute_action(
    lagrangian: np.ndarray,
    dt: float,
) -> Dict:
    """
    Compute action integral: S = ∫L dt

    The principle of least action: δS = 0 for true paths.
    """
    L = np.asarray(lagrangian, dtype=float)

    if np.all(np.isnan(L)):
        return {
            'action': None,
            'mean_lagrangian': None,
            'units': 'J·s',
            'equation': 'S = ∫L dt',
        }

    # np.trapz renamed to np.trapezoid in numpy 2.0
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    S = trapz_fn(L[~np.isnan(L)], dx=dt)

    return {
        'action': float(S),
        'mean_lagrangian': float(np.nanmean(L)),
        'units': 'J·s',
        'equation': 'S = ∫L dt',
    }


def check_euler_lagrange(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    mass: float,
    spring_constant: float,
    dt: float,
) -> Dict:
    """
    Check Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0

    For harmonic oscillator:
        ∂L/∂q̇ = mq̇
        d/dt(∂L/∂q̇) = mq̈
        ∂L/∂q = -kq

    So: mq̈ + kq = 0  (equation of motion)
    """
    q = np.asarray(position, dtype=float)
    q_ddot = np.asarray(acceleration, dtype=float)

    if len(q) < 3:
        return {
            'euler_lagrange_residual': None,
            'residual_rms': None,
            'relative_error': None,
            'equation_satisfied': None,
            'equation': 'mq̈ + kq = 0',
        }

    # LHS of EOM: mq̈
    lhs = mass * q_ddot

    # RHS of EOM: -kq
    rhs = -spring_constant * q

    # Residual (should be ~0)
    residual = lhs - rhs
    residual_rms = np.sqrt(np.nanmean(residual**2))

    # Relative error
    scale = np.sqrt(np.nanmean(lhs**2))
    relative_error = residual_rms / scale if scale > 0 else residual_rms

    return {
        'euler_lagrange_residual': residual,
        'residual_rms': float(residual_rms),
        'relative_error': float(relative_error),
        'equation_satisfied': bool(relative_error < 0.05),
        'equation': 'mq̈ + kq = 0',
    }


def compute(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    spring_constant: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    equilibrium: float = 0.0,
    dt: float = 1.0,
) -> Dict:
    """
    Main compute function for Lagrangian.

    REQUIRES: mass [kg], spring_constant [N/m]

    Args:
        position: q [m]
        velocity: v [m/s]
        mass: m [kg]. REQUIRED.
        spring_constant: k [N/m]. REQUIRED.
        config: Optional config dict
        equilibrium: x₀ [m]
        dt: Time step [s]

    Returns:
        Dict with Lagrangian metrics and action
    """
    result = compute_lagrangian(
        position=position,
        velocity=velocity,
        mass=mass,
        spring_constant=spring_constant,
        config=config,
        equilibrium=equilibrium,
    )

    # Also compute action if we have a valid Lagrangian
    if 'error' not in result and not np.isnan(result.get('mean_L', float('nan'))):
        action_result = compute_action(result['lagrangian'], dt)
        result['action'] = action_result['action']
    else:
        result['action'] = float('nan')

    return result

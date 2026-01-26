"""
Lagrangian Mechanics Engine — THE REAL EQUATION

L(q, q̇, t) = T - V  [J]

The action integral S = ∫L dt is stationary for true paths.
Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0
"""

import numpy as np
from typing import Dict, Optional


def compute_lagrangian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    spring_constant: Optional[float] = None,
    equilibrium: float = 0.0,
) -> Dict:
    """
    Compute Lagrangian: L = T - V

    Args:
        position: Generalized coordinate q
        velocity: Generalized velocity q̇ = dq/dt
        mass: m [kg]
        spring_constant: k [N/m] for harmonic potential
        equilibrium: q₀ equilibrium position

    Returns:
        Dict with Lagrangian and components
    """
    q = np.asarray(position, dtype=float)
    q_dot = np.asarray(velocity, dtype=float)

    # Validate inputs
    if np.all(np.isnan(q)) or np.all(np.isnan(q_dot)):
        return {
            'lagrangian': None,
            'kinetic_energy': None,
            'potential_energy': None,
            'mean_L': None,
            'mass': mass,
            'spring_constant': spring_constant,
            'is_specific': True,
            'equation': 'L = T - V = ½mq̇² - V(q)',
        }

    # Kinetic energy
    if q_dot.ndim > 1:
        v_squared = np.sum(q_dot**2, axis=-1)
    else:
        v_squared = q_dot**2

    if mass is not None:
        T = 0.5 * mass * v_squared
    else:
        T = 0.5 * v_squared

    # Potential energy
    displacement = q - equilibrium
    if q.ndim > 1:
        disp_sq = np.sum(displacement**2, axis=-1)
    else:
        disp_sq = displacement**2

    if spring_constant is not None:
        V = 0.5 * spring_constant * disp_sq
    else:
        V = 0.5 * disp_sq

    # Lagrangian
    L = T - V

    # Determine if specific
    is_specific = mass is None or spring_constant is None

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
        'is_specific': is_specific,
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
    equilibrium: float = 0.0,
    dt: float = 1.0,
) -> Dict:
    """
    Main compute function for Lagrangian.

    Args:
        position: q [m]
        velocity: v [m/s]
        mass: m [kg]
        spring_constant: k [N/m]
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
        equilibrium=equilibrium,
    )

    # Also compute action if we have a valid Lagrangian
    if result['lagrangian'] is not None:
        action_result = compute_action(result['lagrangian'], dt)
        result['action'] = action_result['action']
    else:
        result['action'] = None

    return result

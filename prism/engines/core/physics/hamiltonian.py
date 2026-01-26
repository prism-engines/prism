"""
Hamiltonian Mechanics Engine — THE REAL EQUATION

H(q, p, t) = T(p) + V(q)  [J]

Total mechanical energy. Conserved in closed systems.

When mass/constants known: Real Hamiltonian in Joules
When unknown: Specific Hamiltonian H/m
"""

import numpy as np
from typing import Dict, Optional, Callable


def compute_hamiltonian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    potential_func: Optional[Callable] = None,
    spring_constant: Optional[float] = None,
    equilibrium: float = 0.0,
) -> Dict:
    """
    Compute Hamiltonian: H = T + V (total mechanical energy)

    Args:
        position: q [m]
        velocity: v = dq/dt [m/s]
        mass: m [kg]
        potential_func: V(q) function returning potential energy
        spring_constant: k [N/m] for harmonic potential (if no potential_func)
        equilibrium: x₀ for harmonic potential

    Returns:
        Dict with Hamiltonian and components
    """
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)

    # Validate inputs
    if np.all(np.isnan(q)) or np.all(np.isnan(v)):
        return {
            'hamiltonian': None,
            'kinetic_energy': None,
            'potential_energy': None,
            'momentum': None,
            'mean_H': None,
            'energy_conserved': None,
            'mass': mass,
            'spring_constant': spring_constant,
            'is_specific': True,
            'units': None,
            'equation': 'H = T + V = ½mv² + V(q)',
        }

    # Handle vector quantities
    if v.ndim > 1:
        v_squared = np.sum(v**2, axis=-1)
    else:
        v_squared = v**2

    # Kinetic energy
    if mass is not None:
        T = 0.5 * mass * v_squared
        p = mass * v  # Momentum
    else:
        T = 0.5 * v_squared  # Specific
        p = v  # Specific momentum

    # Potential energy
    if potential_func is not None:
        V = potential_func(q)
    elif spring_constant is not None:
        displacement = q - equilibrium
        if q.ndim > 1:
            disp_sq = np.sum(displacement**2, axis=-1)
        else:
            disp_sq = displacement**2
        V = 0.5 * spring_constant * disp_sq
    else:
        # Specific harmonic potential (k=1)
        displacement = q - equilibrium
        if q.ndim > 1:
            disp_sq = np.sum(displacement**2, axis=-1)
        else:
            disp_sq = displacement**2
        V = 0.5 * disp_sq

    # Hamiltonian
    H = T + V

    # Check conservation
    H_mean = np.nanmean(H)
    H_std = np.nanstd(H)
    is_conserved = (H_std / np.abs(H_mean)) < 0.01 if H_mean != 0 else H_std < 0.01

    # Determine units
    if mass is not None and (potential_func is not None or spring_constant is not None):
        units = 'J'
        is_specific = False
    else:
        units = 'J/kg or m²/s²'
        is_specific = True

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
        'is_specific': is_specific,
        'units': units,
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
    equilibrium: float = 0.0,
) -> Dict:
    """
    Main compute function for Hamiltonian.

    Args:
        position: q [m]
        velocity: v [m/s]
        mass: m [kg]
        spring_constant: k [N/m]
        equilibrium: x₀ [m]

    Returns:
        Dict with Hamiltonian metrics
    """
    return compute_hamiltonian(
        position=position,
        velocity=velocity,
        mass=mass,
        spring_constant=spring_constant,
        equilibrium=equilibrium,
    )

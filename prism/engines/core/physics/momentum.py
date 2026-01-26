"""
Momentum Engine — THE REAL EQUATIONS

Linear:  p = mv     [kg·m/s]
Angular: L = r × p  [kg·m²/s]

Momentum is conserved in closed systems.
"""

import numpy as np
from typing import Dict, Optional


def compute_linear_momentum(
    velocity: np.ndarray,
    mass: Optional[float] = None,
) -> Dict:
    """
    Compute linear momentum: p = mv

    Args:
        velocity: v [m/s], can be 1D or 3D
        mass: m [kg]. If None, returns specific momentum (= velocity)

    Returns:
        Dict with momentum
    """
    v = np.asarray(velocity, dtype=float)

    if np.all(np.isnan(v)):
        return {
            'momentum': None,
            'momentum_magnitude': None,
            'mean_momentum_magnitude': None,
            'mass': mass,
            'is_specific': mass is None,
            'units': None,
            'equation': 'p = mv' if mass else 'p/m = v',
        }

    if mass is not None:
        p = mass * v
        units = 'kg·m/s'
        is_specific = False
    else:
        p = v.copy()
        units = 'm/s'
        is_specific = True

    # Magnitude
    # Handle both time series of scalars and single/multiple 3D vectors
    if v.ndim > 1:
        # Time series of vectors: shape (N, 3)
        p_magnitude = np.sqrt(np.sum(p**2, axis=-1))
    elif v.ndim == 1 and len(v) == 3:
        # Single 3D vector: shape (3,)
        p_magnitude = float(np.sqrt(np.sum(p**2)))
    else:
        # Time series of scalars: shape (N,)
        p_magnitude = np.abs(p)

    # Ensure mean/max work for both scalar and array
    if np.isscalar(p_magnitude):
        mean_mag = p_magnitude
        max_mag = p_magnitude
    else:
        mean_mag = float(np.nanmean(p_magnitude))
        max_mag = float(np.nanmax(p_magnitude))

    return {
        'momentum': p,
        'momentum_magnitude': p_magnitude,
        'mean_momentum_magnitude': mean_mag,
        'max_momentum_magnitude': max_mag,

        'mass': mass,
        'is_specific': is_specific,
        'units': units,
        'equation': 'p = mv' if mass else 'p/m = v',
    }


def compute_angular_momentum(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: Optional[float] = None,
    origin: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute angular momentum: L = r × p = m(r × v)

    THIS IS THE REAL CROSS PRODUCT.

    Args:
        position: r [m], shape (..., 3) for 3D
        velocity: v [m/s], shape (..., 3) for 3D
        mass: m [kg]. If None, returns specific angular momentum r × v
        origin: Origin point [m], default [0,0,0]

    Returns:
        Dict with angular momentum vector and magnitude
    """
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)

    # Validate 3D
    if r.shape[-1] != 3 or v.shape[-1] != 3:
        return {
            'angular_momentum': None,
            'angular_momentum_magnitude': None,
            'mean_magnitude': None,
            'mass': mass,
            'is_specific': mass is None,
            'units': None,
            'error': f'Angular momentum requires 3D vectors. Got position shape {r.shape}, velocity shape {v.shape}',
            'equation': 'L = r × p = m(r × v)',
        }

    # Adjust for origin
    if origin is not None:
        r = r - np.asarray(origin)

    # Cross product: L = r × p = m(r × v)
    r_cross_v = np.cross(r, v)

    if mass is not None:
        L = mass * r_cross_v
        units = 'kg·m²/s'
        is_specific = False
    else:
        L = r_cross_v
        units = 'm²/s'
        is_specific = True

    # Magnitude
    L_magnitude = np.sqrt(np.sum(L**2, axis=-1))

    # Direction (unit vector), with safe division
    L_mag_safe = np.where(L_magnitude[..., np.newaxis] > 1e-10,
                          L_magnitude[..., np.newaxis], 1.0)
    L_direction = L / L_mag_safe

    return {
        'angular_momentum': L,
        'angular_momentum_magnitude': L_magnitude,
        'angular_momentum_direction': L_direction,
        'mean_magnitude': float(np.nanmean(L_magnitude)),
        'max_magnitude': float(np.nanmax(L_magnitude)),

        'mass': mass,
        'origin': origin,
        'is_specific': is_specific,
        'units': units,
        'equation': 'L = r × p = m(r × v)',
    }


def check_momentum_conservation(
    momentum: np.ndarray,
    dt: float,
    tolerance: float = 0.01,
) -> Dict:
    """
    Check if momentum is conserved: dp/dt = F_ext = 0 for closed system.
    """
    p = np.asarray(momentum, dtype=float)

    if np.all(np.isnan(p)) or len(p) < 3:
        return {
            'momentum_mean': None,
            'momentum_std': None,
            'relative_variation': None,
            'is_conserved': None,
            'implied_force': None,
            'mean_force_magnitude': None,
            'equation': 'dp/dt = F_ext (= 0 for closed system)',
        }

    if p.ndim > 1:
        p_magnitude = np.sqrt(np.sum(p**2, axis=-1))
    else:
        p_magnitude = np.abs(p)

    p_mean = np.nanmean(p_magnitude)
    p_std = np.nanstd(p_magnitude)

    dp_dt = np.gradient(p, dt, axis=0)

    if dp_dt.ndim > 1:
        force_magnitude = np.sqrt(np.sum(dp_dt**2, axis=-1))
    else:
        force_magnitude = np.abs(dp_dt)

    mean_force = np.nanmean(force_magnitude)

    # Conservation check
    relative_variation = p_std / p_mean if p_mean > 0 else p_std
    is_conserved = relative_variation < tolerance

    return {
        'momentum_mean': float(p_mean),
        'momentum_std': float(p_std),
        'relative_variation': float(relative_variation),
        'is_conserved': bool(is_conserved),
        'implied_force': dp_dt,
        'mean_force_magnitude': float(mean_force),
        'equation': 'dp/dt = F_ext (= 0 for closed system)',
    }


def compute_impulse(
    force: np.ndarray,
    dt: float,
) -> Dict:
    """
    Compute impulse: J = ∫F dt = Δp
    """
    F = np.asarray(force, dtype=float)

    if np.all(np.isnan(F)):
        return {
            'impulse': None,
            'impulse_magnitude': None,
            'units': 'kg·m/s',
            'equation': 'J = ∫F dt = Δp',
        }

    # Handle NaN values
    F_clean = np.where(np.isnan(F), 0, F)

    # np.trapz renamed to np.trapezoid in numpy 2.0
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    J = trapz_fn(F_clean, dx=dt, axis=0)

    if F.ndim > 1 and F.shape[-1] > 1:
        J_magnitude = np.sqrt(np.sum(J**2))
    else:
        J_magnitude = np.abs(J) if np.isscalar(J) else np.abs(J).item()

    return {
        'impulse': J if not np.isscalar(J) else float(J),
        'impulse_magnitude': float(J_magnitude),
        'units': 'kg·m/s',
        'equation': 'J = ∫F dt = Δp',
    }


def compute(
    velocity: np.ndarray,
    mass: Optional[float] = None,
    position: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Dict:
    """
    Main compute function for momentum.

    Args:
        velocity: v [m/s]
        mass: m [kg]
        position: r [m] (for angular momentum, must be 3D)
        dt: Time step [s]

    Returns:
        Dict with momentum metrics
    """
    # Compute linear momentum
    result = compute_linear_momentum(velocity, mass)

    # Add conservation check
    if result['momentum'] is not None:
        conservation = check_momentum_conservation(result['momentum'], dt)
        result['is_conserved'] = conservation['is_conserved']
        result['relative_variation'] = conservation['relative_variation']
        result['mean_force_magnitude'] = conservation['mean_force_magnitude']

    # Compute angular momentum if 3D position provided
    if position is not None:
        v = np.asarray(velocity, dtype=float)
        r = np.asarray(position, dtype=float)
        if r.shape[-1] == 3 and v.shape[-1] == 3:
            angular = compute_angular_momentum(position, velocity, mass)
            result['angular_momentum'] = angular['angular_momentum']
            result['angular_momentum_magnitude'] = angular['mean_magnitude']

    return result

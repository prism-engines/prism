"""
Work-Energy Engine — THE REAL EQUATIONS

Work: W = ∫F·dx = ΔKE  [J]
Power: P = dW/dt = F·v  [W]

Work-Energy Theorem: W_net = ΔKE
"""

import numpy as np
from typing import Dict, Optional


def compute_work(
    force: np.ndarray,
    displacement: np.ndarray,
    dt: float = 1.0,
) -> Dict:
    """
    Compute work: W = ∫F·dx

    For discrete data: W = Σ F·Δx

    Args:
        force: F [N], can be scalar or vector
        displacement: x [m], must match force dimensions
        dt: Time step (for power calculation)

    Returns:
        Dict with work and related quantities
    """
    F = np.asarray(force, dtype=float)
    x = np.asarray(displacement, dtype=float)

    if len(F) < 2 or len(x) < 2:
        return {
            'work_incremental': None,
            'work_cumulative': None,
            'work_total': None,
            'mean_force': None,
            'total_displacement': None,
            'units': 'J',
            'equation': 'W = ∫F·dx',
        }

    # Compute displacement increments
    dx = np.diff(x, axis=0)

    # Average force over each interval
    if F.ndim > 1:
        F_avg = 0.5 * (F[:-1] + F[1:])
        # Dot product for each interval
        dW = np.sum(F_avg * dx, axis=-1)
    else:
        F_avg = 0.5 * (F[:-1] + F[1:])
        dW = F_avg * dx

    # Cumulative work
    W = np.concatenate([[0], np.cumsum(dW)])

    # Total work
    W_total = W[-1]

    return {
        'work_incremental': dW,
        'work_cumulative': W,
        'work_total': float(W_total),

        'mean_force': float(np.nanmean(np.abs(F))),
        'total_displacement': float(np.nansum(np.abs(dx))),

        'units': 'J',
        'equation': 'W = ∫F·dx',
    }


def compute_power(
    force: np.ndarray,
    velocity: np.ndarray,
) -> Dict:
    """
    Compute power: P = F·v = dW/dt

    Args:
        force: F [N]
        velocity: v [m/s]

    Returns:
        Dict with power
    """
    F = np.asarray(force, dtype=float)
    v = np.asarray(velocity, dtype=float)

    if np.all(np.isnan(F)) or np.all(np.isnan(v)):
        return {
            'power': None,
            'mean_power': None,
            'max_power': None,
            'units': 'W',
            'equation': 'P = F·v',
        }

    if F.ndim > 1 and v.ndim > 1:
        P = np.sum(F * v, axis=-1)
    else:
        P = F * v

    return {
        'power': P,
        'mean_power': float(np.nanmean(P)),
        'max_power': float(np.nanmax(np.abs(P))),
        'min_power': float(np.nanmin(P)),

        'units': 'W',
        'equation': 'P = F·v',
    }


def verify_work_energy_theorem(
    kinetic_energy_initial: float,
    kinetic_energy_final: float,
    work_done: float,
    tolerance: float = 0.05,
) -> Dict:
    """
    Verify Work-Energy Theorem: W_net = ΔKE = KE_f - KE_i
    """
    delta_KE = kinetic_energy_final - kinetic_energy_initial

    error = abs(work_done - delta_KE)
    relative_error = error / abs(delta_KE) if delta_KE != 0 else error

    return {
        'work_done': float(work_done),
        'delta_KE': float(delta_KE),
        'error': float(error),
        'relative_error': float(relative_error),
        'theorem_satisfied': bool(relative_error < tolerance),
        'equation': 'W_net = ΔKE',
    }


def compute_conservative_force_test(
    force: np.ndarray,
    position: np.ndarray,
) -> Dict:
    """
    Test if force field is conservative: ∇ × F = 0

    For 3D force field, compute curl. Zero curl = conservative.
    """
    F = np.asarray(force, dtype=float)
    x = np.asarray(position, dtype=float)

    if F.shape[-1] != 3:
        return {
            'is_conservative': None,
            'error': 'Conservative test requires 3D force vectors',
            'equation': '∇ × F = 0 for conservative forces',
        }

    if len(F) < 3:
        return {
            'is_conservative': None,
            'error': 'Insufficient data points for curl calculation',
            'equation': '∇ × F = 0 for conservative forces',
        }

    # Numerical curl (approximate)
    # For trajectory data, this is an approximation
    dFx_dy = np.gradient(F[..., 0], axis=0)
    dFy_dx = np.gradient(F[..., 1], axis=0)

    curl_z = dFy_dx - dFx_dy

    # Conservative if curl ≈ 0
    curl_magnitude = np.sqrt(np.nanmean(curl_z**2))
    force_magnitude = np.sqrt(np.nanmean(F**2))

    is_conservative = curl_magnitude < 0.1 * force_magnitude if force_magnitude > 0 else True

    return {
        'curl_z_component': curl_z,
        'curl_magnitude': float(curl_magnitude),
        'force_magnitude': float(force_magnitude),
        'is_conservative': bool(is_conservative),
        'equation': '∇ × F = 0 for conservative forces',
    }


def compute_mechanical_energy(
    kinetic_energy: np.ndarray,
    potential_energy: np.ndarray,
) -> Dict:
    """
    Compute total mechanical energy: E = KE + PE

    For conservative forces, E is conserved.
    """
    KE = np.asarray(kinetic_energy, dtype=float)
    PE = np.asarray(potential_energy, dtype=float)

    E = KE + PE

    E_mean = np.nanmean(E)
    E_std = np.nanstd(E)

    # Check conservation
    relative_variation = E_std / np.abs(E_mean) if E_mean != 0 else E_std
    is_conserved = relative_variation < 0.01

    return {
        'mechanical_energy': E,
        'mean_energy': float(E_mean),
        'std_energy': float(E_std),
        'relative_variation': float(relative_variation),
        'is_conserved': bool(is_conserved),

        'KE_fraction': float(np.nanmean(KE) / E_mean) if E_mean != 0 else 0.0,
        'PE_fraction': float(np.nanmean(PE) / E_mean) if E_mean != 0 else 0.0,

        'equation': 'E = KE + PE',
    }


def compute(
    force: np.ndarray,
    displacement: np.ndarray,
    velocity: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Dict:
    """
    Main compute function for work-energy.

    Args:
        force: F [N]
        displacement: x [m]
        velocity: v [m/s] (optional, for power)
        dt: Time step [s]

    Returns:
        Dict with work, power, and energy metrics
    """
    result = compute_work(force, displacement, dt)

    # Add power if velocity provided
    if velocity is not None:
        power_result = compute_power(force, velocity)
        result['power'] = power_result['power']
        result['mean_power'] = power_result['mean_power']
        result['max_power'] = power_result['max_power']

    return result

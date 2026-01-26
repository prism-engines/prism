"""
Flash Calculation Engines

Isothermal flash, adiabatic flash, Rachford-Rice equation.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import brentq, fsolve


def rachford_rice(z: np.ndarray, K: np.ndarray,
                  V_F_guess: float = 0.5) -> Dict[str, Any]:
    """
    Rachford-Rice equation for flash calculations.

    Σ z_i(K_i - 1) / (1 + V/F(K_i - 1)) = 0

    Solves for vapor fraction V/F given feed composition z and K-values.

    Parameters
    ----------
    z : array
        Feed mole fractions (must sum to 1)
    K : array
        K-values (y_i/x_i) for each component
    V_F_guess : float
        Initial guess for V/F (vapor fraction)

    Returns
    -------
    dict
        V_F: Vapor fraction (V/F)
        x: Liquid mole fractions
        y: Vapor mole fractions
        phase_state: 'two-phase', 'all-liquid', 'all-vapor'
    """
    z = np.asarray(z)
    K = np.asarray(K)

    # Check bounds
    # If V/F = 0: Σ z_i * K_i >= 1 for any vapor
    # If V/F = 1: Σ z_i / K_i >= 1 for any liquid
    sum_zK = np.sum(z * K)
    sum_z_over_K = np.sum(z / K)

    if sum_zK <= 1.0:
        # All liquid
        return {
            'V_F': 0.0,
            'x': z.tolist(),
            'y': (K * z).tolist(),  # Hypothetical
            'phase_state': 'all-liquid',
            'equation': 'Σ z_i·K_i < 1 → subcooled liquid',
        }

    if sum_z_over_K <= 1.0:
        # All vapor
        return {
            'V_F': 1.0,
            'x': (z / K).tolist(),  # Hypothetical
            'y': z.tolist(),
            'phase_state': 'all-vapor',
            'equation': 'Σ z_i/K_i < 1 → superheated vapor',
        }

    # Two-phase: solve Rachford-Rice
    def objective(V_F):
        return np.sum(z * (K - 1) / (1 + V_F * (K - 1)))

    # Find valid bounds where objective changes sign
    V_F_min = max(0, (K.max() - 1) / (K.max() - K.min()) * (-0.01))
    V_F_max = min(1, (1 - K.min()) / (K.max() - K.min()) * 1.01)

    try:
        V_F = brentq(objective, 1e-10, 1 - 1e-10)
    except ValueError:
        V_F = fsolve(objective, V_F_guess)[0]
        V_F = np.clip(V_F, 0, 1)

    # Calculate compositions
    x = z / (1 + V_F * (K - 1))
    y = K * x

    # Normalize to ensure sum = 1
    x = x / np.sum(x)
    y = y / np.sum(y)

    return {
        'V_F': float(V_F),
        'L_F': float(1 - V_F),
        'x': x.tolist(),
        'y': y.tolist(),
        'phase_state': 'two-phase',
        'equation': 'Σ z_i(K_i-1)/(1+V/F(K_i-1)) = 0',
    }


def isothermal_flash(z: np.ndarray, T: float, P: float,
                     antoine_params: List[Tuple[float, float, float]],
                     gamma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Isothermal flash calculation at specified T and P.

    Given: T, P, z
    Find: V/F, x, y

    Parameters
    ----------
    z : array
        Feed mole fractions
    T : float
        Temperature [°C for Antoine]
    P : float
        Pressure [Pa]
    antoine_params : list of tuples
        Antoine parameters for each component
    gamma : array, optional
        Activity coefficients (default: 1.0)

    Returns
    -------
    dict
        V_F: Vapor fraction
        x: Liquid compositions
        y: Vapor compositions
        K_values: Equilibrium ratios
        phase_state: Phase determination
    """
    z = np.asarray(z)
    n_comp = len(z)

    if gamma is None:
        gamma = np.ones(n_comp)

    # Calculate K-values from Antoine equation
    K = np.zeros(n_comp)
    P_sat = np.zeros(n_comp)

    for i, (A, B, C) in enumerate(antoine_params):
        log_P = A - B / (T + C)
        P_sat[i] = (10 ** log_P) * 133.322  # mmHg to Pa
        K[i] = gamma[i] * P_sat[i] / P

    # Solve Rachford-Rice
    result = rachford_rice(z, K)

    result['K_values'] = K.tolist()
    result['P_sat'] = P_sat.tolist()
    result['temperature'] = T
    result['pressure'] = P

    return result


def adiabatic_flash(z: np.ndarray, H_feed: float, P: float,
                    antoine_params: List[Tuple[float, float, float]],
                    Cp_liquid: np.ndarray, Cp_vapor: np.ndarray,
                    H_vap: np.ndarray, T_ref: float = 298.15,
                    T_guess: float = 350.0,
                    gamma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Adiabatic flash calculation.

    Given: H_feed, P, z
    Find: T, V/F, x, y (no heat added or removed)

    Parameters
    ----------
    z : array
        Feed mole fractions
    H_feed : float
        Feed enthalpy [J/mol]
    P : float
        Pressure [Pa]
    antoine_params : list
        Antoine parameters
    Cp_liquid : array
        Liquid heat capacities [J/(mol·K)]
    Cp_vapor : array
        Vapor heat capacities [J/(mol·K)]
    H_vap : array
        Heats of vaporization [J/mol]
    T_ref : float
        Reference temperature [K]
    T_guess : float
        Initial temperature guess [K]
    gamma : array, optional
        Activity coefficients

    Returns
    -------
    dict
        T: Flash temperature
        V_F: Vapor fraction
        x: Liquid compositions
        y: Vapor compositions
        H_outlet: Outlet enthalpy (should equal H_feed)
    """
    z = np.asarray(z)
    n_comp = len(z)

    if gamma is None:
        gamma = np.ones(n_comp)

    Cp_liquid = np.asarray(Cp_liquid)
    Cp_vapor = np.asarray(Cp_vapor)
    H_vap = np.asarray(H_vap)

    def enthalpy_outlet(T, V_F, x, y):
        """Calculate outlet enthalpy at temperature T."""
        # Liquid enthalpy: H_L = sum(x_i * Cp_L_i * (T - T_ref))
        H_L = np.sum(x * Cp_liquid * (T - T_ref))

        # Vapor enthalpy: H_V = sum(y_i * (Cp_V_i * (T - T_ref) + H_vap_i))
        H_V = np.sum(y * (Cp_vapor * (T - T_ref) + H_vap))

        # Total outlet enthalpy
        return (1 - V_F) * H_L + V_F * H_V

    def objective(T):
        """Energy balance: H_feed = H_outlet."""
        # Calculate K-values at T
        K = np.zeros(n_comp)
        for i, (A, B, C) in enumerate(antoine_params):
            log_P = A - B / (T + C)
            P_sat = (10 ** log_P) * 133.322
            K[i] = gamma[i] * P_sat / P

        # Solve for V/F
        rr_result = rachford_rice(z, K)
        V_F = rr_result['V_F']
        x = np.asarray(rr_result['x'])
        y = np.asarray(rr_result['y'])

        H_out = enthalpy_outlet(T, V_F, x, y)
        return H_feed - H_out

    # Solve for T
    try:
        T_flash = brentq(objective, 200, 600)
    except ValueError:
        T_flash = fsolve(objective, T_guess)[0]

    # Final flash calculation at T_flash
    K = np.zeros(n_comp)
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        log_P = A - B / (T_flash + C)
        P_sat[i] = (10 ** log_P) * 133.322
        K[i] = gamma[i] * P_sat[i] / P

    rr_result = rachford_rice(z, K)
    V_F = rr_result['V_F']
    x = np.asarray(rr_result['x'])
    y = np.asarray(rr_result['y'])

    H_outlet = enthalpy_outlet(T_flash, V_F, x, y)

    return {
        'T': float(T_flash),
        'V_F': float(V_F),
        'L_F': float(1 - V_F),
        'x': x.tolist(),
        'y': y.tolist(),
        'K_values': K.tolist(),
        'H_feed': H_feed,
        'H_outlet': float(H_outlet),
        'energy_balance_error': float(H_feed - H_outlet),
        'phase_state': rr_result['phase_state'],
    }


def three_phase_flash(z: np.ndarray, T: float, P: float,
                      K_VL: np.ndarray, K_LL: np.ndarray) -> Dict[str, Any]:
    """
    Three-phase flash (VLLE) calculation.

    Simplified three-phase flash assuming two liquid phases + vapor.

    Parameters
    ----------
    z : array
        Feed mole fractions
    T : float
        Temperature
    P : float
        Pressure
    K_VL : array
        Vapor-liquid 1 K-values
    K_LL : array
        Liquid 2 - liquid 1 K-values

    Returns
    -------
    dict
        phase_fractions: V, L1, L2
        compositions: x1, x2, y
        phase_state: 'three-phase', 'two-phase', etc.
    """
    z = np.asarray(z)
    K_VL = np.asarray(K_VL)
    K_LL = np.asarray(K_LL)
    n_comp = len(z)

    # Three-phase Rachford-Rice requires solving 2 equations
    # This is a simplified version assuming small vapor fraction

    def objective(fracs):
        V, L2 = fracs
        L1 = 1 - V - L2

        if L1 <= 0 or L2 <= 0 or V < 0:
            return [1e10, 1e10]

        # Mass balance for each phase
        denom = 1 + V * (K_VL - 1) + L2 * (K_LL - 1)
        x1 = z / denom

        # Rachford-Rice type equations
        eq1 = np.sum(z * (K_VL - 1) / denom)  # Vapor
        eq2 = np.sum(z * (K_LL - 1) / denom)  # Second liquid

        return [eq1, eq2]

    # Initial guess: small amounts of each phase
    try:
        solution = fsolve(objective, [0.1, 0.2], full_output=True)
        V, L2 = solution[0]
        L1 = 1 - V - L2
    except:
        # Fall back to two-phase
        return {
            'phase_state': 'calculation-failed',
            'error': 'Three-phase flash did not converge',
        }

    # Check physical validity
    if V < 0 or L1 < 0 or L2 < 0:
        return {
            'phase_state': 'two-phase-or-single',
            'note': 'Negative phase fraction - system is not three-phase',
        }

    # Calculate compositions
    denom = 1 + V * (K_VL - 1) + L2 * (K_LL - 1)
    x1 = z / denom
    x2 = K_LL * x1
    y = K_VL * x1

    # Normalize
    x1 = x1 / np.sum(x1)
    x2 = x2 / np.sum(x2)
    y = y / np.sum(y)

    return {
        'V': float(V),
        'L1': float(L1),
        'L2': float(L2),
        'x1': x1.tolist(),
        'x2': x2.tolist(),
        'y': y.tolist(),
        'phase_state': 'three-phase',
        'temperature': T,
        'pressure': P,
    }


def compute(signal: np.ndarray = None, z: np.ndarray = None,
            K: np.ndarray = None, T: float = None, P: float = None,
            antoine_params: List = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for flash calculations.
    """
    if z is not None and K is not None:
        return rachford_rice(z, K)

    if z is not None and T is not None and P is not None and antoine_params is not None:
        return isothermal_flash(z, T, P, antoine_params, kwargs.get('gamma'))

    return {'error': 'Insufficient parameters for flash calculation'}

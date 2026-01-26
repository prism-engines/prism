"""
Vapor-Liquid Equilibrium (VLE) Engines

Fundamental VLE calculations: Antoine, Raoult's law, K-values,
bubble point, dew point calculations.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import brentq, fsolve


# Physical constants
R = 8.314  # J/(mol·K)


def antoine(T: float, A: float, B: float, C: float,
            units: str = "mmHg") -> Dict[str, Any]:
    """
    Antoine equation for vapor pressure.

    log10(P*) = A - B/(T + C)

    Parameters
    ----------
    T : float
        Temperature [°C or K depending on constants]
    A, B, C : float
        Antoine constants (component-specific)
    units : str
        'mmHg' (default), 'bar', 'Pa', 'kPa'

    Returns
    -------
    dict
        vapor_pressure: Saturated vapor pressure
        equation: str
        units: str
    """
    log_P = A - B / (T + C)
    P_sat = 10 ** log_P

    # Convert to standard units if needed
    P_Pa = P_sat
    if units == "mmHg":
        P_Pa = P_sat * 133.322
    elif units == "bar":
        P_Pa = P_sat * 1e5
    elif units == "kPa":
        P_Pa = P_sat * 1e3

    return {
        'vapor_pressure': P_sat,
        'vapor_pressure_Pa': P_Pa,
        'equation': f'log10(P*) = {A} - {B}/(T + {C})',
        'units': units,
    }


def raoults_law(x: np.ndarray, P_sat: np.ndarray,
                P_total: Optional[float] = None) -> Dict[str, Any]:
    """
    Raoult's Law for ideal mixtures.

    y_i * P = x_i * P_i*

    Parameters
    ----------
    x : array
        Liquid mole fractions
    P_sat : array
        Vapor pressures of pure components [Pa]
    P_total : float, optional
        Total pressure [Pa]. If None, calculated as sum(x_i * P_i*)

    Returns
    -------
    dict
        y: Vapor mole fractions
        P_total: Total pressure [Pa]
        K_values: y_i/x_i for each component
    """
    x = np.asarray(x)
    P_sat = np.asarray(P_sat)

    # Partial pressures
    p_i = x * P_sat

    if P_total is None:
        P_total = np.sum(p_i)

    # Vapor compositions
    y = p_i / P_total

    # K-values
    K = P_sat / P_total

    return {
        'y': y.tolist(),
        'P_total': float(P_total),
        'K_values': K.tolist(),
        'partial_pressures': p_i.tolist(),
        'equation': 'y_i·P = x_i·P_i*',
    }


def modified_raoults(x: np.ndarray, P_sat: np.ndarray, gamma: np.ndarray,
                     P_total: Optional[float] = None) -> Dict[str, Any]:
    """
    Modified Raoult's Law with activity coefficients.

    y_i * P = x_i * gamma_i * P_i*

    Parameters
    ----------
    x : array
        Liquid mole fractions
    P_sat : array
        Vapor pressures [Pa]
    gamma : array
        Activity coefficients
    P_total : float, optional
        Total pressure [Pa]

    Returns
    -------
    dict
        y: Vapor mole fractions
        P_total: Total pressure [Pa]
        K_values: Apparent K values (y_i/x_i)
    """
    x = np.asarray(x)
    P_sat = np.asarray(P_sat)
    gamma = np.asarray(gamma)

    # Partial pressures
    p_i = x * gamma * P_sat

    if P_total is None:
        P_total = np.sum(p_i)

    y = p_i / P_total
    K = gamma * P_sat / P_total

    return {
        'y': y.tolist(),
        'P_total': float(P_total),
        'K_values': K.tolist(),
        'activity_coefficients': gamma.tolist(),
        'equation': 'y_i·P = x_i·γ_i·P_i*',
    }


def k_value(T: float, P: float, antoine_params: List[Tuple[float, float, float]],
            gamma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate K-values (equilibrium ratios) for multicomponent mixture.

    K_i = y_i/x_i = gamma_i * P_i*(T) / P

    Parameters
    ----------
    T : float
        Temperature [°C for Antoine]
    P : float
        Total pressure [Pa]
    antoine_params : list of tuples
        [(A1, B1, C1), (A2, B2, C2), ...] for each component
    gamma : array, optional
        Activity coefficients (default: 1.0 for all)

    Returns
    -------
    dict
        K_values: Equilibrium ratios
        P_sat: Vapor pressures
        relative_volatilities: K_i/K_j matrix
    """
    n_comp = len(antoine_params)

    # Calculate vapor pressures
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        P_sat[i] = antoine(T, A, B, C, units='Pa')['vapor_pressure_Pa']

    if gamma is None:
        gamma = np.ones(n_comp)

    # K-values
    K = gamma * P_sat / P

    # Relative volatilities
    alpha = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            alpha[i, j] = K[i] / K[j] if K[j] != 0 else np.inf

    return {
        'K_values': K.tolist(),
        'P_sat': P_sat.tolist(),
        'relative_volatilities': alpha.tolist(),
        'temperature': T,
        'pressure': P,
    }


def relative_volatility(K1: float, K2: float) -> Dict[str, Any]:
    """
    Relative volatility between two components.

    alpha_12 = K_1 / K_2 = (y_1/x_1) / (y_2/x_2)

    Parameters
    ----------
    K1 : float
        K-value of more volatile component
    K2 : float
        K-value of less volatile component

    Returns
    -------
    dict
        alpha: Relative volatility
        separation_feasibility: Qualitative assessment
    """
    alpha = K1 / K2

    if alpha > 2.0:
        feasibility = "Easy separation"
    elif alpha > 1.2:
        feasibility = "Moderate separation"
    elif alpha > 1.05:
        feasibility = "Difficult separation"
    else:
        feasibility = "Near-azeotropic, consider extractive distillation"

    return {
        'alpha': alpha,
        'separation_feasibility': feasibility,
        'equation': 'α = K_1/K_2',
    }


def bubble_point_pressure(x: np.ndarray, T: float,
                          antoine_params: List[Tuple[float, float, float]],
                          gamma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Bubble point pressure calculation at given T and liquid composition.

    At bubble point: sum(y_i) = sum(x_i * gamma_i * P_i* / P) = 1
    Therefore: P_bubble = sum(x_i * gamma_i * P_i*)

    Parameters
    ----------
    x : array
        Liquid mole fractions
    T : float
        Temperature [°C for Antoine]
    antoine_params : list of tuples
        Antoine parameters for each component
    gamma : array, optional
        Activity coefficients

    Returns
    -------
    dict
        P_bubble: Bubble point pressure [Pa]
        y: Vapor composition at bubble point
        K_values: Equilibrium ratios
    """
    x = np.asarray(x)
    n_comp = len(x)

    if gamma is None:
        gamma = np.ones(n_comp)

    # Vapor pressures at T
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        P_sat[i] = antoine(T, A, B, C, units='Pa')['vapor_pressure_Pa']

    # Bubble pressure
    P_bubble = np.sum(x * gamma * P_sat)

    # Vapor composition
    K = gamma * P_sat / P_bubble
    y = K * x

    return {
        'P_bubble': float(P_bubble),
        'y': y.tolist(),
        'K_values': K.tolist(),
        'P_sat': P_sat.tolist(),
        'temperature': T,
    }


def bubble_point_temperature(x: np.ndarray, P: float,
                             antoine_params: List[Tuple[float, float, float]],
                             gamma: Optional[np.ndarray] = None,
                             T_guess: float = 100.0) -> Dict[str, Any]:
    """
    Bubble point temperature calculation at given P and liquid composition.

    Find T such that sum(x_i * gamma_i * P_i*(T) / P) = 1

    Parameters
    ----------
    x : array
        Liquid mole fractions
    P : float
        Total pressure [Pa]
    antoine_params : list of tuples
        Antoine parameters
    gamma : array, optional
        Activity coefficients
    T_guess : float
        Initial temperature guess [°C]

    Returns
    -------
    dict
        T_bubble: Bubble point temperature [°C]
        y: Vapor composition
        K_values: Equilibrium ratios
    """
    x = np.asarray(x)
    n_comp = len(x)

    if gamma is None:
        gamma = np.ones(n_comp)

    def objective(T):
        P_sat = np.zeros(n_comp)
        for i, (A, B, C) in enumerate(antoine_params):
            P_sat[i] = antoine(T, A, B, C, units='Pa')['vapor_pressure_Pa']
        return np.sum(x * gamma * P_sat / P) - 1.0

    # Solve for bubble point temperature
    try:
        T_bubble = brentq(objective, -50, 500)
    except ValueError:
        # Fall back to fsolve
        T_bubble = fsolve(objective, T_guess)[0]

    # Calculate compositions at bubble point
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        P_sat[i] = antoine(T_bubble, A, B, C, units='Pa')['vapor_pressure_Pa']

    K = gamma * P_sat / P
    y = K * x

    return {
        'T_bubble': float(T_bubble),
        'y': y.tolist(),
        'K_values': K.tolist(),
        'P_sat': P_sat.tolist(),
        'pressure': P,
    }


def dew_point_pressure(y: np.ndarray, T: float,
                       antoine_params: List[Tuple[float, float, float]],
                       gamma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Dew point pressure calculation at given T and vapor composition.

    At dew point: sum(x_i) = sum(y_i * P / (gamma_i * P_i*)) = 1
    Therefore: P_dew = 1 / sum(y_i / (gamma_i * P_i*))

    Parameters
    ----------
    y : array
        Vapor mole fractions
    T : float
        Temperature [°C]
    antoine_params : list of tuples
        Antoine parameters
    gamma : array, optional
        Activity coefficients (assumed constant for simplicity)

    Returns
    -------
    dict
        P_dew: Dew point pressure [Pa]
        x: Liquid composition at dew point
        K_values: Equilibrium ratios
    """
    y = np.asarray(y)
    n_comp = len(y)

    if gamma is None:
        gamma = np.ones(n_comp)

    # Vapor pressures
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        P_sat[i] = antoine(T, A, B, C, units='Pa')['vapor_pressure_Pa']

    # Dew pressure (ideal case)
    P_dew = 1.0 / np.sum(y / (gamma * P_sat))

    # Liquid composition
    K = gamma * P_sat / P_dew
    x = y / K

    return {
        'P_dew': float(P_dew),
        'x': x.tolist(),
        'K_values': K.tolist(),
        'P_sat': P_sat.tolist(),
        'temperature': T,
    }


def dew_point_temperature(y: np.ndarray, P: float,
                          antoine_params: List[Tuple[float, float, float]],
                          gamma: Optional[np.ndarray] = None,
                          T_guess: float = 100.0) -> Dict[str, Any]:
    """
    Dew point temperature calculation at given P and vapor composition.

    Find T such that sum(y_i * P / (gamma_i * P_i*(T))) = 1

    Parameters
    ----------
    y : array
        Vapor mole fractions
    P : float
        Total pressure [Pa]
    antoine_params : list of tuples
        Antoine parameters
    gamma : array, optional
        Activity coefficients
    T_guess : float
        Initial temperature guess [°C]

    Returns
    -------
    dict
        T_dew: Dew point temperature [°C]
        x: Liquid composition
        K_values: Equilibrium ratios
    """
    y = np.asarray(y)
    n_comp = len(y)

    if gamma is None:
        gamma = np.ones(n_comp)

    def objective(T):
        P_sat = np.zeros(n_comp)
        for i, (A, B, C) in enumerate(antoine_params):
            P_sat[i] = antoine(T, A, B, C, units='Pa')['vapor_pressure_Pa']
        return np.sum(y * P / (gamma * P_sat)) - 1.0

    try:
        T_dew = brentq(objective, -50, 500)
    except ValueError:
        T_dew = fsolve(objective, T_guess)[0]

    # Compositions at dew point
    P_sat = np.zeros(n_comp)
    for i, (A, B, C) in enumerate(antoine_params):
        P_sat[i] = antoine(T_dew, A, B, C, units='Pa')['vapor_pressure_Pa']

    K = gamma * P_sat / P
    x = y / K

    return {
        'T_dew': float(T_dew),
        'x': x.tolist(),
        'K_values': K.tolist(),
        'P_sat': P_sat.tolist(),
        'pressure': P,
    }


def txy_diagram(x_range: np.ndarray, P: float,
                antoine_params: List[Tuple[float, float, float]],
                gamma_func=None) -> Dict[str, Any]:
    """
    Generate T-x-y diagram data at constant pressure (binary system).

    Parameters
    ----------
    x_range : array
        Range of x_1 values (0 to 1)
    P : float
        Total pressure [Pa]
    antoine_params : list
        Antoine parameters for components 1 and 2
    gamma_func : callable, optional
        Function(x1, T) -> (gamma1, gamma2)

    Returns
    -------
    dict
        x1: Liquid compositions
        y1: Vapor compositions
        T_bubble: Bubble point curve
        T_dew: Dew point curve
    """
    x_range = np.asarray(x_range)
    n_points = len(x_range)

    T_bubble = np.zeros(n_points)
    y1 = np.zeros(n_points)

    for i, x1 in enumerate(x_range):
        x = np.array([x1, 1 - x1])

        if gamma_func is not None:
            # Need iterative solution with gamma
            def solve_T(T):
                gamma = gamma_func(x1, T)
                P_sat = np.array([
                    antoine(T, *antoine_params[0], units='Pa')['vapor_pressure_Pa'],
                    antoine(T, *antoine_params[1], units='Pa')['vapor_pressure_Pa']
                ])
                return np.sum(x * gamma * P_sat / P) - 1.0

            try:
                T_bubble[i] = brentq(solve_T, -50, 500)
            except:
                T_bubble[i] = fsolve(solve_T, 100)[0]

            gamma = gamma_func(x1, T_bubble[i])
        else:
            gamma = np.array([1.0, 1.0])
            result = bubble_point_temperature(x, P, antoine_params, gamma)
            T_bubble[i] = result['T_bubble']

        # Calculate y1
        P_sat = np.array([
            antoine(T_bubble[i], *antoine_params[0], units='Pa')['vapor_pressure_Pa'],
            antoine(T_bubble[i], *antoine_params[1], units='Pa')['vapor_pressure_Pa']
        ])
        K = gamma * P_sat / P
        y1[i] = K[0] * x1

    return {
        'x1': x_range.tolist(),
        'y1': y1.tolist(),
        'T_bubble': T_bubble.tolist(),
        'pressure': P,
        'diagram_type': 'T-x-y',
    }


def pxy_diagram(x_range: np.ndarray, T: float,
                antoine_params: List[Tuple[float, float, float]],
                gamma_func=None) -> Dict[str, Any]:
    """
    Generate P-x-y diagram data at constant temperature (binary system).

    Parameters
    ----------
    x_range : array
        Range of x_1 values (0 to 1)
    T : float
        Temperature [°C]
    antoine_params : list
        Antoine parameters for components 1 and 2
    gamma_func : callable, optional
        Function(x1) -> (gamma1, gamma2)

    Returns
    -------
    dict
        x1: Liquid compositions
        y1: Vapor compositions
        P_bubble: Bubble point pressure curve
        P_dew: Dew point pressure curve
    """
    x_range = np.asarray(x_range)
    n_points = len(x_range)

    # Vapor pressures at T
    P_sat = np.array([
        antoine(T, *antoine_params[0], units='Pa')['vapor_pressure_Pa'],
        antoine(T, *antoine_params[1], units='Pa')['vapor_pressure_Pa']
    ])

    P_bubble = np.zeros(n_points)
    y1 = np.zeros(n_points)

    for i, x1 in enumerate(x_range):
        x = np.array([x1, 1 - x1])

        if gamma_func is not None:
            gamma = gamma_func(x1)
        else:
            gamma = np.array([1.0, 1.0])

        P_bubble[i] = np.sum(x * gamma * P_sat)
        K = gamma * P_sat / P_bubble[i]
        y1[i] = K[0] * x1

    return {
        'x1': x_range.tolist(),
        'y1': y1.tolist(),
        'P_bubble': P_bubble.tolist(),
        'P_sat': P_sat.tolist(),
        'temperature': T,
        'diagram_type': 'P-x-y',
    }


# Convenience function for single call
def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for VLE calculations.

    Automatically selects appropriate calculation based on provided inputs.
    """
    if 'x' in kwargs and 'T' in kwargs and 'antoine_params' in kwargs:
        if 'P' not in kwargs:
            return bubble_point_pressure(kwargs['x'], kwargs['T'],
                                         kwargs['antoine_params'],
                                         kwargs.get('gamma'))
        else:
            return raoults_law(kwargs['x'],
                              [antoine(kwargs['T'], *p, units='Pa')['vapor_pressure_Pa']
                               for p in kwargs['antoine_params']],
                              kwargs.get('P'))

    if 'y' in kwargs and 'T' in kwargs and 'antoine_params' in kwargs:
        return dew_point_pressure(kwargs['y'], kwargs['T'],
                                  kwargs['antoine_params'],
                                  kwargs.get('gamma'))

    return {'error': 'Insufficient parameters for VLE calculation'}

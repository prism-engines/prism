"""
Absorption and Stripping Engines

NTU, HTU, Kremser equation, operating lines.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy.integrate import quad


def ntu(y_in: float, y_out: float, y_eq_func, L_over_G: float = None,
        m: float = None, A: float = None) -> Dict[str, Any]:
    """
    Number of Transfer Units for absorption/stripping.

    NTU = ∫(dy / (y - y*))  from y_out to y_in

    For dilute systems with linear equilibrium (y* = mx):
    NTU = ln[(y_in - mx_in)/(y_out - mx_out)] / (1 - 1/A)  for A ≠ 1

    Parameters
    ----------
    y_in : float
        Gas phase mole fraction at inlet
    y_out : float
        Gas phase mole fraction at outlet
    y_eq_func : callable or float
        Function y*(x) or slope m for linear equilibrium
    L_over_G : float, optional
        Liquid to gas molar flow ratio
    m : float, optional
        Equilibrium line slope (y* = mx)
    A : float, optional
        Absorption factor A = L/(mG)

    Returns
    -------
    dict
        NTU: Number of transfer units
        removal_efficiency: (y_in - y_out) / y_in
    """
    removal_eff = (y_in - y_out) / y_in if y_in > 0 else 0

    # If absorption factor provided directly
    if A is not None:
        if abs(A - 1.0) < 1e-6:
            # A = 1 case
            NTU_val = (y_in - y_out) / y_out if y_out > 0 else float('inf')
        else:
            # General case using Colburn equation
            NTU_val = np.log((1 - 1/A) * (y_in/y_out) + 1/A) / (1 - 1/A)
    elif m is not None and L_over_G is not None:
        A = L_over_G / m
        if abs(A - 1.0) < 1e-6:
            NTU_val = (y_in - y_out) / y_out if y_out > 0 else float('inf')
        else:
            NTU_val = np.log((1 - 1/A) * (y_in/y_out) + 1/A) / (1 - 1/A)
    elif callable(y_eq_func):
        # Numerical integration
        def integrand(y):
            # Need to relate y to x via operating line
            # Simplified: assume linear operating line
            x = (y_in - y) / L_over_G if L_over_G else 0
            y_star = y_eq_func(x)
            driving_force = y - y_star
            return 1.0 / driving_force if abs(driving_force) > 1e-10 else 0

        NTU_val, _ = quad(integrand, y_out, y_in)
    else:
        return {'error': 'Insufficient parameters for NTU calculation'}

    return {
        'NTU': float(NTU_val),
        'y_in': y_in,
        'y_out': y_out,
        'removal_efficiency': float(removal_eff),
        'removal_percent': float(removal_eff * 100),
        'A': float(A) if A is not None else None,
        'equation': 'NTU = ∫dy/(y-y*)',
    }


def htu(G: float, k_y_a: float, P: float = 101325) -> Dict[str, Any]:
    """
    Height of a Transfer Unit.

    HTU = G / (k_y · a · P)

    Parameters
    ----------
    G : float
        Gas molar flux [mol/(m²·s)]
    k_y_a : float
        Volumetric mass transfer coefficient [mol/(m³·s)]
    P : float
        Total pressure [Pa] (default 101325)

    Returns
    -------
    dict
        HTU: Height of transfer unit [m]
    """
    HTU_val = G / k_y_a

    return {
        'HTU': float(HTU_val),
        'G': G,
        'k_y_a': k_y_a,
        'P': P,
        'equation': 'HTU = G/(k_y·a)',
    }


def packed_height(NTU: float, HTU: float) -> Dict[str, Any]:
    """
    Required packing height.

    Z = NTU × HTU

    Parameters
    ----------
    NTU : float
        Number of transfer units
    HTU : float
        Height of transfer unit [m]

    Returns
    -------
    dict
        Z: Packing height [m]
    """
    Z = NTU * HTU

    return {
        'Z': float(Z),
        'NTU': NTU,
        'HTU': HTU,
        'equation': 'Z = NTU × HTU',
    }


def kremser(y_in: float, y_out: float, x_in: float, m: float,
            L_over_G: float) -> Dict[str, Any]:
    """
    Kremser equation for theoretical stages in absorption.

    For absorption:
    N = ln[(y_in - mx_in)/(y_out - mx_in) × (1 - 1/A) + 1/A] / ln(A)

    Parameters
    ----------
    y_in : float
        Gas inlet mole fraction
    y_out : float
        Gas outlet mole fraction (desired)
    x_in : float
        Liquid inlet mole fraction (usually 0 for fresh solvent)
    m : float
        Equilibrium line slope (y* = mx)
    L_over_G : float
        Liquid to gas molar flow ratio

    Returns
    -------
    dict
        N: Number of theoretical stages
        A: Absorption factor
        removal_efficiency: Fraction removed
    """
    A = L_over_G / m

    y_star_in = m * x_in  # Equilibrium with inlet liquid

    if abs(A - 1.0) < 1e-6:
        # A = 1 special case
        N = (y_in - y_out) / (y_out - y_star_in) if (y_out - y_star_in) > 0 else float('inf')
    else:
        numerator = (y_in - y_star_in) / (y_out - y_star_in) * (1 - 1/A) + 1/A
        N = np.log(numerator) / np.log(A)

    removal_eff = (y_in - y_out) / (y_in - y_star_in) if (y_in - y_star_in) > 0 else 0

    return {
        'N': float(N),
        'N_rounded': int(np.ceil(N)),
        'A': float(A),
        'y_in': y_in,
        'y_out': y_out,
        'x_in': x_in,
        'm': m,
        'L_over_G': L_over_G,
        'removal_efficiency': float(removal_eff),
        'equation': 'Kremser equation',
    }


def operating_line(y_in: float, y_out: float, x_in: float, x_out: float,
                   G: float = None, L: float = None) -> Dict[str, Any]:
    """
    Operating line for absorption/stripping.

    Mass balance: G(y_in - y) = L(x - x_in)
    Slope: L/G = (y_in - y_out)/(x_out - x_in)

    Parameters
    ----------
    y_in : float
        Gas inlet mole fraction
    y_out : float
        Gas outlet mole fraction
    x_in : float
        Liquid inlet mole fraction
    x_out : float
        Liquid outlet mole fraction
    G : float, optional
        Gas molar flow rate [mol/s]
    L : float, optional
        Liquid molar flow rate [mol/s]

    Returns
    -------
    dict
        slope: L/G ratio
        intercept: y-intercept of operating line
        L_over_G: Liquid to gas ratio
    """
    if (x_out - x_in) != 0:
        slope = (y_in - y_out) / (x_out - x_in)
    else:
        slope = float('inf')

    # Operating line: y = (L/G)x + (y_in - (L/G)x_out)
    intercept = y_in - slope * x_out if slope != float('inf') else None

    # Calculate L/G from flows if provided
    if G is not None and L is not None:
        L_over_G = L / G
    else:
        L_over_G = slope

    return {
        'slope': float(slope) if slope != float('inf') else None,
        'intercept': float(intercept) if intercept is not None else None,
        'L_over_G': float(L_over_G),
        'y_in': y_in,
        'y_out': y_out,
        'x_in': x_in,
        'x_out': x_out,
        'equation': 'y = (L/G)x + b',
    }


def minimum_liquid_rate(y_in: float, y_out: float, x_in: float,
                        m: float, G: float) -> Dict[str, Any]:
    """
    Minimum liquid rate for absorption.

    At minimum L, operating line touches equilibrium at gas inlet.
    L_min/G = m × (y_in - y_out)/(y_in/m - x_in)

    Parameters
    ----------
    y_in : float
        Gas inlet mole fraction
    y_out : float
        Gas outlet mole fraction
    x_in : float
        Liquid inlet mole fraction
    m : float
        Equilibrium slope
    G : float
        Gas flow rate [mol/s]

    Returns
    -------
    dict
        L_min: Minimum liquid rate [mol/s]
        L_over_G_min: Minimum L/G ratio
        A_min: Minimum absorption factor
    """
    x_out_max = y_in / m  # Maximum possible liquid outlet (equilibrium with gas inlet)

    L_over_G_min = (y_in - y_out) / (x_out_max - x_in) if (x_out_max - x_in) > 0 else float('inf')
    L_min = L_over_G_min * G
    A_min = L_over_G_min / m

    return {
        'L_min': float(L_min),
        'L_over_G_min': float(L_over_G_min),
        'A_min': float(A_min),
        'x_out_max': float(x_out_max),
        'typical_L': float(1.5 * L_min),  # Typical: 1.3-1.5 × L_min
        'equation': 'L_min from pinch at gas inlet',
    }


def overall_mass_transfer(k_G: float, k_L: float, m: float, H: float = None,
                          P: float = 101325) -> Dict[str, Any]:
    """
    Overall mass transfer coefficients.

    Gas-phase resistance: 1/K_G = 1/k_G + m/k_L
    Liquid-phase resistance: 1/K_L = 1/(m·k_G) + 1/k_L

    Parameters
    ----------
    k_G : float
        Gas-side mass transfer coefficient [mol/(m²·s·Pa)]
    k_L : float
        Liquid-side mass transfer coefficient [m/s]
    m : float
        Equilibrium slope (dimensionless)
    H : float, optional
        Henry's law constant [Pa] (alternative to m)
    P : float
        Total pressure [Pa]

    Returns
    -------
    dict
        K_G: Overall gas-phase coefficient
        K_L: Overall liquid-phase coefficient
        gas_resistance_fraction: Fraction of resistance in gas phase
    """
    if H is not None:
        m = H / P  # Convert Henry's constant to slope

    # Resistances
    R_G = 1 / k_G if k_G > 0 else float('inf')
    R_L = m / k_L if k_L > 0 else float('inf')

    # Overall coefficients
    K_G = 1 / (R_G + R_L) if (R_G + R_L) > 0 else 0
    K_L = 1 / (R_G / m + 1 / k_L) if k_L > 0 and m > 0 else 0

    gas_frac = R_G / (R_G + R_L) if (R_G + R_L) > 0 else 0

    return {
        'K_G': float(K_G),
        'K_L': float(K_L),
        'k_G': k_G,
        'k_L': k_L,
        'm': m,
        'gas_resistance_fraction': float(gas_frac),
        'liquid_resistance_fraction': float(1 - gas_frac),
        'controlling_resistance': 'gas' if gas_frac > 0.5 else 'liquid',
        'equation': '1/K_G = 1/k_G + m/k_L',
    }


def stripping_factor(m: float, G_over_L: float) -> Dict[str, Any]:
    """
    Stripping factor (inverse of absorption factor).

    S = m × G/L = 1/A

    Parameters
    ----------
    m : float
        Equilibrium slope
    G_over_L : float
        Gas to liquid molar ratio

    Returns
    -------
    dict
        S: Stripping factor
        A: Absorption factor
        favors: 'absorption' if A > 1, 'stripping' if S > 1
    """
    S = m * G_over_L
    A = 1 / S

    return {
        'S': float(S),
        'A': float(A),
        'm': m,
        'G_over_L': G_over_L,
        'favors': 'absorption' if A > 1 else 'stripping',
        'equation': 'S = mG/L = 1/A',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for absorption calculations.
    """
    if all(k in kwargs for k in ['y_in', 'y_out', 'x_in', 'm', 'L_over_G']):
        return kremser(kwargs['y_in'], kwargs['y_out'], kwargs['x_in'],
                      kwargs['m'], kwargs['L_over_G'])

    if 'NTU' in kwargs and 'HTU' in kwargs:
        return packed_height(kwargs['NTU'], kwargs['HTU'])

    return {'error': 'Insufficient parameters'}

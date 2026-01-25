"""
Liquid-Liquid Extraction Engines

Single stage, cross-current, counter-current extraction.
"""

import numpy as np
from typing import Dict, Any, Optional, List


def single_stage_extraction(F: float, x_F: float, S: float, x_S: float,
                            K_D: float) -> Dict[str, Any]:
    """
    Single-stage liquid-liquid extraction.

    Mass balance: F·x_F + S·x_S = R·x_R + E·x_E
    Equilibrium: x_E = K_D · x_R

    Where:
    - F = Feed flow rate
    - S = Solvent flow rate
    - R = Raffinate (feed phase leaving)
    - E = Extract (solvent phase leaving)

    Parameters
    ----------
    F : float
        Feed flow rate [mol/s or kg/s]
    x_F : float
        Feed solute concentration (mass or mole fraction)
    S : float
        Solvent flow rate [mol/s or kg/s]
    x_S : float
        Solvent inlet concentration (usually 0)
    K_D : float
        Distribution coefficient (x_E / x_R)

    Returns
    -------
    dict
        x_R: Raffinate concentration
        x_E: Extract concentration
        extraction_efficiency: Fraction of solute removed
        E_factor: Extraction factor (K_D × S/F)
    """
    # Extraction factor
    E_factor = K_D * S / F

    # Mass balance on solute: F·x_F + S·x_S = F·x_R + S·x_E
    # Equilibrium: x_E = K_D · x_R
    # Combining: F·x_F + S·x_S = F·x_R + S·K_D·x_R = x_R(F + S·K_D)

    x_R = (F * x_F + S * x_S) / (F + S * K_D)
    x_E = K_D * x_R

    # Solute in each phase
    solute_in = F * x_F + S * x_S
    solute_raffinate = F * x_R
    solute_extract = S * x_E

    extraction_eff = (x_F - x_R) / x_F if x_F > 0 else 0

    return {
        'x_R': float(x_R),
        'x_E': float(x_E),
        'extraction_efficiency': float(extraction_eff),
        'extraction_percent': float(extraction_eff * 100),
        'E_factor': float(E_factor),
        'solute_in_raffinate': float(solute_raffinate),
        'solute_in_extract': float(solute_extract),
        'F': F,
        'S': S,
        'K_D': K_D,
        'equation': 'Single equilibrium stage',
    }


def cross_current(F: float, x_F: float, S_per_stage: float, K_D: float,
                  n_stages: int) -> Dict[str, Any]:
    """
    Cross-current (multiple contact) extraction.

    Fresh solvent added at each stage. Raffinates connected in series.

    Parameters
    ----------
    F : float
        Feed flow rate [mol/s]
    x_F : float
        Feed concentration
    S_per_stage : float
        Fresh solvent per stage [mol/s]
    K_D : float
        Distribution coefficient
    n_stages : int
        Number of stages

    Returns
    -------
    dict
        x_R_final: Final raffinate concentration
        x_R_stages: Concentration after each stage
        overall_efficiency: Total fraction extracted
        total_solvent: Total solvent used
    """
    x_R = x_F
    x_R_stages = []
    x_E_stages = []

    E_factor = K_D * S_per_stage / F

    for i in range(n_stages):
        # Each stage with fresh solvent (x_S = 0)
        x_R_new = x_R / (1 + E_factor)
        x_E = K_D * x_R_new

        x_R_stages.append(float(x_R_new))
        x_E_stages.append(float(x_E))
        x_R = x_R_new

    overall_eff = (x_F - x_R) / x_F if x_F > 0 else 0
    total_solvent = S_per_stage * n_stages

    # Analytical solution: x_R/x_F = 1/(1+E)^n
    x_R_analytical = x_F / (1 + E_factor)**n_stages

    return {
        'x_R_final': float(x_R),
        'x_R_analytical': float(x_R_analytical),
        'x_R_stages': x_R_stages,
        'x_E_stages': x_E_stages,
        'overall_efficiency': float(overall_eff),
        'overall_efficiency_percent': float(overall_eff * 100),
        'total_solvent': float(total_solvent),
        'E_factor_per_stage': float(E_factor),
        'n_stages': n_stages,
        'equation': 'x_R/x_F = 1/(1+E)^n',
    }


def counter_current(F: float, x_F: float, S: float, x_S: float,
                    K_D: float, n_stages: int) -> Dict[str, Any]:
    """
    Counter-current extraction (most efficient).

    Feed and solvent flow in opposite directions.
    Uses Kremser-type equation.

    Parameters
    ----------
    F : float
        Feed flow rate [mol/s]
    x_F : float
        Feed concentration
    S : float
        Solvent flow rate [mol/s]
    x_S : float
        Solvent inlet concentration (usually 0)
    K_D : float
        Distribution coefficient
    n_stages : int
        Number of stages

    Returns
    -------
    dict
        x_R: Final raffinate concentration
        x_E: Final extract concentration
        extraction_efficiency: Fraction extracted
    """
    E = K_D * S / F  # Extraction factor

    if abs(E - 1.0) < 1e-6:
        # E = 1 special case
        x_R = x_F * (1 - n_stages / (n_stages + 1))
    else:
        # Kremser equation for extraction
        # (x_F - x_R)/(x_F - x_S/K_D) = (E^(n+1) - E)/(E^(n+1) - 1)
        x_eq_with_solvent = x_S / K_D
        kremser_factor = (E**(n_stages + 1) - E) / (E**(n_stages + 1) - 1)
        x_R = x_F - kremser_factor * (x_F - x_eq_with_solvent)

    # Extract concentration from mass balance
    # F·x_F + S·x_S = F·x_R + S·x_E
    x_E = (F * (x_F - x_R) + S * x_S) / S if S > 0 else 0

    extraction_eff = (x_F - x_R) / x_F if x_F > 0 else 0

    return {
        'x_R': float(x_R),
        'x_E': float(x_E),
        'extraction_efficiency': float(extraction_eff),
        'extraction_percent': float(extraction_eff * 100),
        'E_factor': float(E),
        'n_stages': n_stages,
        'F': F,
        'S': S,
        'K_D': K_D,
        'equation': 'Kremser equation for counter-current',
    }


def extraction_efficiency(x_in: float, x_out: float, x_eq: float = 0) -> Dict[str, Any]:
    """
    Stage efficiency for extraction.

    Murphree efficiency: E_M = (x_in - x_out) / (x_in - x_eq)

    Parameters
    ----------
    x_in : float
        Inlet concentration
    x_out : float
        Outlet concentration
    x_eq : float
        Equilibrium concentration (default 0)

    Returns
    -------
    dict
        efficiency: Murphree efficiency (0-1)
        actual_change: x_in - x_out
        max_possible_change: x_in - x_eq
    """
    actual = x_in - x_out
    maximum = x_in - x_eq

    efficiency = actual / maximum if maximum > 0 else 0

    return {
        'efficiency': float(efficiency),
        'efficiency_percent': float(efficiency * 100),
        'actual_change': float(actual),
        'max_possible_change': float(maximum),
        'x_in': x_in,
        'x_out': x_out,
        'x_eq': x_eq,
        'equation': 'E_M = (x_in - x_out)/(x_in - x_eq)',
    }


def minimum_solvent_ratio(x_F: float, x_R_desired: float, K_D: float,
                          n_stages: int = None) -> Dict[str, Any]:
    """
    Minimum solvent-to-feed ratio.

    For infinite stages: (S/F)_min = (x_F - x_R)/(K_D × x_F)

    Parameters
    ----------
    x_F : float
        Feed concentration
    x_R_desired : float
        Desired raffinate concentration
    K_D : float
        Distribution coefficient
    n_stages : int, optional
        Number of stages (if None, assumes infinite)

    Returns
    -------
    dict
        S_over_F_min: Minimum solvent ratio
        E_min: Minimum extraction factor
    """
    # At minimum S/F, extract is in equilibrium with feed
    x_E_max = K_D * x_F

    # Mass balance: F·x_F = F·x_R + S·x_E
    # S/F = (x_F - x_R) / x_E
    S_over_F_min = (x_F - x_R_desired) / x_E_max if x_E_max > 0 else float('inf')
    E_min = K_D * S_over_F_min

    return {
        'S_over_F_min': float(S_over_F_min),
        'E_min': float(E_min),
        'x_E_max': float(x_E_max),
        'typical_S_over_F': float(1.5 * S_over_F_min),
        'x_F': x_F,
        'x_R_desired': x_R_desired,
        'K_D': K_D,
        'equation': '(S/F)_min at infinite stages',
    }


def stages_required(x_F: float, x_R_desired: float, K_D: float,
                    S_over_F: float, mode: str = 'counter_current') -> Dict[str, Any]:
    """
    Calculate stages required for given separation.

    Parameters
    ----------
    x_F : float
        Feed concentration
    x_R_desired : float
        Desired raffinate concentration
    K_D : float
        Distribution coefficient
    S_over_F : float
        Solvent to feed ratio
    mode : str
        'counter_current' or 'cross_current'

    Returns
    -------
    dict
        n_stages: Required number of stages
    """
    E = K_D * S_over_F

    if mode == 'cross_current':
        # x_R/x_F = 1/(1+E)^n
        # n = ln(x_F/x_R) / ln(1+E)
        n = np.log(x_F / x_R_desired) / np.log(1 + E) if x_R_desired > 0 else float('inf')
    else:  # counter_current
        # Kremser: solve for n
        if abs(E - 1) < 1e-6:
            # E = 1 case
            recovery = (x_F - x_R_desired) / x_F
            n = recovery / (1 - recovery) if recovery < 1 else float('inf')
        else:
            # (x_F - x_R)/(x_F) = (E^(n+1) - E)/(E^(n+1) - 1)
            recovery = (x_F - x_R_desired) / x_F
            # Solve numerically or use log form
            # Approximation:
            if E > 1:
                n = np.log((1 - recovery) * (E - 1) + E) / np.log(E) - 1
            else:
                n = np.log(1 / (1 - recovery * (1 - 1/E))) / np.log(E)

    return {
        'n_stages': float(n),
        'n_stages_rounded': int(np.ceil(n)),
        'E_factor': float(E),
        'mode': mode,
        'x_F': x_F,
        'x_R_desired': x_R_desired,
        'recovery': float((x_F - x_R_desired) / x_F),
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for extraction calculations.
    """
    if all(k in kwargs for k in ['F', 'x_F', 'S', 'K_D']):
        if 'n_stages' in kwargs:
            return counter_current(kwargs['F'], kwargs['x_F'], kwargs['S'],
                                   kwargs.get('x_S', 0), kwargs['K_D'],
                                   kwargs['n_stages'])
        else:
            return single_stage_extraction(kwargs['F'], kwargs['x_F'],
                                          kwargs['S'], kwargs.get('x_S', 0),
                                          kwargs['K_D'])

    return {'error': 'Insufficient parameters'}

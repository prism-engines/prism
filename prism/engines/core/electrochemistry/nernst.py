"""
Electrochemical Thermodynamics Engines

Nernst equation, cell potential, Gibbs free energy relationships.
"""

import numpy as np
from typing import Dict, Any, Optional, List


# Physical constants
F = 96485.33212  # Faraday constant [C/mol]
R = 8.314462618  # Gas constant [J/(mol·K)]


def nernst(E0: float, n: int, Q: float, T: float = 298.15) -> Dict[str, Any]:
    """
    Nernst equation for electrode potential.

    E = E° - (RT/nF) ln(Q)

    At 25°C (298.15 K):
    E = E° - (0.05916/n) log₁₀(Q)

    Parameters
    ----------
    E0 : float
        Standard electrode potential [V]
    n : int
        Number of electrons transferred
    Q : float
        Reaction quotient (products/reactants)
    T : float
        Temperature [K] (default 298.15)

    Returns
    -------
    dict
        E: Electrode potential [V]
        E0: Standard potential [V]
        RT_nF: Nernst slope [V]
        overpotential: E - E0 [V]
    """
    RT_nF = R * T / (n * F)
    E = E0 - RT_nF * np.log(Q)

    # At 25°C coefficient for log10
    coeff_25C = 0.05916 / n

    return {
        'E': float(E),
        'E0': E0,
        'RT_nF': float(RT_nF),
        'n': n,
        'Q': Q,
        'temperature': T,
        'overpotential': float(E - E0),
        'nernst_slope_log10': float(coeff_25C) if abs(T - 298.15) < 1 else float(2.303 * RT_nF),
        'equation': 'E = E° - (RT/nF)ln(Q)',
    }


def cell_potential(E_cathode: float, E_anode: float) -> Dict[str, Any]:
    """
    Calculate cell potential from half-cell potentials.

    E_cell = E_cathode - E_anode

    Parameters
    ----------
    E_cathode : float
        Cathode (reduction) potential [V]
    E_anode : float
        Anode (reduction) potential [V]

    Returns
    -------
    dict
        E_cell: Cell potential [V]
        spontaneous: Whether reaction is spontaneous (E_cell > 0)
    """
    E_cell = E_cathode - E_anode

    return {
        'E_cell': float(E_cell),
        'E_cathode': E_cathode,
        'E_anode': E_anode,
        'spontaneous': E_cell > 0,
        'equation': 'E_cell = E_cathode - E_anode',
    }


def gibbs_electrochemical(E: float, n: int, T: float = 298.15) -> Dict[str, Any]:
    """
    Gibbs free energy from cell potential.

    ΔG = -nFE

    Parameters
    ----------
    E : float
        Cell potential [V]
    n : int
        Number of electrons transferred
    T : float
        Temperature [K] (for entropy/enthalpy if needed)

    Returns
    -------
    dict
        delta_G: Gibbs free energy change [J/mol]
        delta_G_kJ: Gibbs free energy change [kJ/mol]
        spontaneous: ΔG < 0
        max_work: Maximum electrical work [J/mol]
    """
    delta_G = -n * F * E
    delta_G_kJ = delta_G / 1000

    return {
        'delta_G': float(delta_G),
        'delta_G_kJ': float(delta_G_kJ),
        'E': E,
        'n': n,
        'spontaneous': delta_G < 0,
        'max_work': float(-delta_G),  # W_max = -ΔG
        'equation': 'ΔG = -nFE',
    }


def equilibrium_constant_echem(E0: float, n: int, T: float = 298.15) -> Dict[str, Any]:
    """
    Equilibrium constant from standard cell potential.

    K = exp(nFE°/RT)

    At equilibrium, E = 0, so:
    E° = (RT/nF) ln(K)

    Parameters
    ----------
    E0 : float
        Standard cell potential [V]
    n : int
        Number of electrons transferred
    T : float
        Temperature [K]

    Returns
    -------
    dict
        K: Equilibrium constant
        log_K: log₁₀(K)
        delta_G0: Standard Gibbs energy [J/mol]
    """
    nFE0_RT = n * F * E0 / (R * T)
    K = np.exp(nFE0_RT)
    log_K = nFE0_RT / np.log(10)
    delta_G0 = -n * F * E0

    return {
        'K': float(K),
        'log_K': float(log_K),
        'delta_G0': float(delta_G0),
        'E0': E0,
        'n': n,
        'temperature': T,
        'equation': 'K = exp(nFE°/RT)',
    }


def pourbaix(E: float, pH: float, E0: float, n_H: int, n_e: int) -> Dict[str, Any]:
    """
    Pourbaix diagram point calculation.

    For reactions involving H⁺:
    E = E° - (0.05916 * n_H / n_e) * pH  at 25°C

    Parameters
    ----------
    E : float
        Electrode potential [V]
    pH : float
        Solution pH
    E0 : float
        Standard potential at pH = 0 [V]
    n_H : int
        Number of H⁺ in reaction
    n_e : int
        Number of electrons transferred

    Returns
    -------
    dict
        E_equilibrium: Equilibrium potential at given pH
        slope: dE/dpH [V/pH unit]
        stability: Above/below equilibrium line
    """
    slope = -0.05916 * n_H / n_e  # V per pH unit at 25°C
    E_equilibrium = E0 + slope * pH

    return {
        'E_equilibrium': float(E_equilibrium),
        'E_measured': E,
        'pH': pH,
        'slope': float(slope),
        'above_line': E > E_equilibrium,
        'stability': 'oxidized' if E > E_equilibrium else 'reduced',
        'equation': 'E = E° - (0.05916·n_H/n_e)·pH',
    }


def concentration_cell(C1: float, C2: float, n: int, T: float = 298.15) -> Dict[str, Any]:
    """
    Concentration cell potential.

    E = (RT/nF) ln(C1/C2)

    Parameters
    ----------
    C1 : float
        Concentration at cathode [mol/L]
    C2 : float
        Concentration at anode [mol/L]
    n : int
        Number of electrons transferred
    T : float
        Temperature [K]

    Returns
    -------
    dict
        E: Cell potential [V]
        direction: Current flow direction
    """
    RT_nF = R * T / (n * F)
    E = RT_nF * np.log(C1 / C2)

    return {
        'E': float(E),
        'C_high': max(C1, C2),
        'C_low': min(C1, C2),
        'concentration_ratio': C1 / C2,
        'current_direction': 'cathode to anode' if E > 0 else 'anode to cathode',
        'equation': 'E = (RT/nF)ln(C₁/C₂)',
    }


def temperature_coefficient(E1: float, T1: float, E2: float, T2: float,
                           n: int) -> Dict[str, Any]:
    """
    Temperature coefficient of cell potential.

    dE/dT = ΔS / (nF)

    Parameters
    ----------
    E1, E2 : float
        Cell potentials at T1, T2 [V]
    T1, T2 : float
        Temperatures [K]
    n : int
        Number of electrons

    Returns
    -------
    dict
        dE_dT: Temperature coefficient [V/K]
        delta_S: Entropy change [J/(mol·K)]
        delta_H: Enthalpy change [J/mol]
    """
    dE_dT = (E2 - E1) / (T2 - T1)
    delta_S = n * F * dE_dT

    # Gibbs-Helmholtz: ΔH = ΔG + TΔS = -nFE + nFT(dE/dT)
    T_avg = (T1 + T2) / 2
    E_avg = (E1 + E2) / 2
    delta_H = -n * F * E_avg + n * F * T_avg * dE_dT

    return {
        'dE_dT': float(dE_dT),
        'delta_S': float(delta_S),
        'delta_H': float(delta_H),
        'delta_G_at_T1': float(-n * F * E1),
        'reversible': abs(dE_dT) < 1e-4,  # Small temp coefficient
        'equation': 'dE/dT = ΔS/(nF)',
    }


def compute(signal: np.ndarray = None, E0: float = None, n: int = None,
            Q: float = None, T: float = 298.15, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for electrochemical thermodynamics.
    """
    if E0 is not None and n is not None and Q is not None:
        return nernst(E0, n, Q, T)

    if 'E_cathode' in kwargs and 'E_anode' in kwargs:
        return cell_potential(kwargs['E_cathode'], kwargs['E_anode'])

    if 'E' in kwargs and n is not None:
        return gibbs_electrochemical(kwargs['E'], n, T)

    return {'error': 'Insufficient parameters'}

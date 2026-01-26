"""
Energy Balance Engines

Enthalpy balance, heat of reaction, LMTD, effectiveness-NTU.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


# Gas constant
R = 8.314  # J/(mol·K)


def sensible_heat(n: float, Cp: float, T1: float, T2: float) -> Dict[str, Any]:
    """
    Sensible heat for temperature change.

    Q = n · Cp · (T2 - T1)

    Parameters
    ----------
    n : float
        Moles [mol] or mass [kg]
    Cp : float
        Heat capacity [J/(mol·K) or J/(kg·K)]
    T1 : float
        Initial temperature [K]
    T2 : float
        Final temperature [K]

    Returns
    -------
    dict
        Q: Heat required [J]
        Q_kJ: Heat required [kJ]
        direction: 'heating' or 'cooling'
    """
    Q = n * Cp * (T2 - T1)

    return {
        'Q': float(Q),
        'Q_kJ': float(Q / 1000),
        'n': n,
        'Cp': Cp,
        'T1': T1,
        'T2': T2,
        'delta_T': float(T2 - T1),
        'direction': 'heating' if T2 > T1 else 'cooling',
        'equation': 'Q = n·Cp·ΔT',
    }


def sensible_heat_integral(n: float, Cp_coeffs: Tuple[float, ...],
                           T1: float, T2: float) -> Dict[str, Any]:
    """
    Sensible heat with temperature-dependent Cp.

    Cp = a + bT + cT² + dT³
    Q = n · ∫Cp dT = n · [aT + bT²/2 + cT³/3 + dT⁴/4]

    Parameters
    ----------
    n : float
        Moles [mol]
    Cp_coeffs : tuple
        Coefficients (a, b, c, d) for Cp polynomial
    T1 : float
        Initial temperature [K]
    T2 : float
        Final temperature [K]

    Returns
    -------
    dict
        Q: Heat required [J]
        Cp_avg: Average heat capacity over range
    """
    a = Cp_coeffs[0] if len(Cp_coeffs) > 0 else 0
    b = Cp_coeffs[1] if len(Cp_coeffs) > 1 else 0
    c = Cp_coeffs[2] if len(Cp_coeffs) > 2 else 0
    d = Cp_coeffs[3] if len(Cp_coeffs) > 3 else 0

    def H(T):
        return a * T + b * T**2 / 2 + c * T**3 / 3 + d * T**4 / 4

    delta_H = H(T2) - H(T1)
    Q = n * delta_H

    # Average Cp
    Cp_avg = delta_H / (T2 - T1) if T2 != T1 else a

    return {
        'Q': float(Q),
        'Q_kJ': float(Q / 1000),
        'Cp_avg': float(Cp_avg),
        'Cp_at_T1': float(a + b * T1 + c * T1**2 + d * T1**3),
        'Cp_at_T2': float(a + b * T2 + c * T2**2 + d * T2**3),
        'equation': 'Q = n·∫Cp(T)dT',
    }


def latent_heat(n: float, delta_H_vap: float) -> Dict[str, Any]:
    """
    Latent heat for phase change.

    Q = n · ΔH_vap  (or ΔH_fus)

    Parameters
    ----------
    n : float
        Moles [mol] or mass [kg]
    delta_H_vap : float
        Heat of vaporization [J/mol or J/kg]

    Returns
    -------
    dict
        Q: Heat required for vaporization [J]
        Q_kJ: Heat required [kJ]
    """
    Q = n * delta_H_vap

    return {
        'Q': float(Q),
        'Q_kJ': float(Q / 1000),
        'n': n,
        'delta_H_vap': delta_H_vap,
        'equation': 'Q = n·ΔH_vap',
    }


def enthalpy_balance(H_in: List[float], H_out: List[float],
                     Q: float = 0.0, W: float = 0.0) -> Dict[str, Any]:
    """
    Overall enthalpy balance.

    Σ(n·H)_in + Q = Σ(n·H)_out + W

    Parameters
    ----------
    H_in : list of float
        Enthalpy of each inlet stream [J]
    H_out : list of float
        Enthalpy of each outlet stream [J]
    Q : float
        Heat added to system [J] (positive = heating)
    W : float
        Work done by system [J] (positive = work out)

    Returns
    -------
    dict
        balance_error: Residual in energy balance
        total_in: Total enthalpy in + Q
        total_out: Total enthalpy out + W
    """
    total_H_in = sum(H_in)
    total_H_out = sum(H_out)

    balance_error = (total_H_in + Q) - (total_H_out + W)

    return {
        'total_H_in': float(total_H_in),
        'total_H_out': float(total_H_out),
        'Q': Q,
        'W': W,
        'balance_error': float(balance_error),
        'balanced': abs(balance_error) < 1e-6 * max(abs(total_H_in), abs(total_H_out), 1),
        'equation': 'ΣH_in + Q = ΣH_out + W',
    }


def heat_of_reaction_calc(delta_Hf_products: List[float], nu_products: List[int],
                          delta_Hf_reactants: List[float], nu_reactants: List[int]) -> Dict[str, Any]:
    """
    Heat of reaction from heats of formation.

    ΔH_rxn = Σ(ν_products · ΔH_f,products) - Σ(ν_reactants · ΔH_f,reactants)

    Parameters
    ----------
    delta_Hf_products : list of float
        Heats of formation of products [J/mol]
    nu_products : list of int
        Stoichiometric coefficients of products
    delta_Hf_reactants : list of float
        Heats of formation of reactants [J/mol]
    nu_reactants : list of int
        Stoichiometric coefficients of reactants

    Returns
    -------
    dict
        delta_H_rxn: Heat of reaction [J/mol]
        exothermic: Whether reaction releases heat
    """
    H_products = sum(nu * Hf for nu, Hf in zip(nu_products, delta_Hf_products))
    H_reactants = sum(nu * Hf for nu, Hf in zip(nu_reactants, delta_Hf_reactants))

    delta_H_rxn = H_products - H_reactants

    return {
        'delta_H_rxn': float(delta_H_rxn),
        'delta_H_rxn_kJ': float(delta_H_rxn / 1000),
        'H_products': float(H_products),
        'H_reactants': float(H_reactants),
        'exothermic': delta_H_rxn < 0,
        'endothermic': delta_H_rxn > 0,
        'equation': 'ΔH_rxn = Σ(ν·ΔH_f)_products - Σ(ν·ΔH_f)_reactants',
    }


def adiabatic_flame_temp(T_reactants: float, Cp_products: float,
                         delta_H_rxn: float, n_products: float) -> Dict[str, Any]:
    """
    Adiabatic flame temperature.

    At adiabatic conditions (Q = 0):
    n_products · Cp · (T_flame - T_ref) = -ΔH_rxn

    Parameters
    ----------
    T_reactants : float
        Temperature of reactants [K]
    Cp_products : float
        Average heat capacity of products [J/(mol·K)]
    delta_H_rxn : float
        Heat of reaction (negative for exothermic) [J/mol]
    n_products : float
        Total moles of products per mole of reaction

    Returns
    -------
    dict
        T_flame: Adiabatic flame temperature [K]
        delta_T: Temperature rise [K]
    """
    delta_T = -delta_H_rxn / (n_products * Cp_products)
    T_flame = T_reactants + delta_T

    return {
        'T_flame': float(T_flame),
        'T_flame_C': float(T_flame - 273.15),
        'T_reactants': T_reactants,
        'delta_T': float(delta_T),
        'equation': 'T_flame = T_in - ΔH_rxn/(n·Cp)',
    }


def adiabatic_reaction_temp(T_in: float, delta_H_rxn: float, conversion: float,
                            Cp: float, n_total: float) -> Dict[str, Any]:
    """
    Outlet temperature for adiabatic reactor.

    Energy balance: n · Cp · (T_out - T_in) = -X · ΔH_rxn

    Parameters
    ----------
    T_in : float
        Inlet temperature [K]
    delta_H_rxn : float
        Heat of reaction [J/mol]
    conversion : float
        Conversion (0-1)
    Cp : float
        Heat capacity of mixture [J/(mol·K)]
    n_total : float
        Total molar flow rate [mol/s]

    Returns
    -------
    dict
        T_out: Outlet temperature [K]
        Q_released: Heat released by reaction [J/s]
    """
    Q_released = -conversion * delta_H_rxn
    delta_T = Q_released / (n_total * Cp) if n_total * Cp > 0 else 0
    T_out = T_in + delta_T

    return {
        'T_out': float(T_out),
        'T_out_C': float(T_out - 273.15),
        'T_in': T_in,
        'delta_T': float(delta_T),
        'Q_released': float(Q_released),
        'conversion': conversion,
        'equation': 'T_out = T_in - X·ΔH_rxn/(n·Cp)',
    }


def heat_exchanger_duty(m_dot: float, Cp: float, T_in: float, T_out: float) -> Dict[str, Any]:
    """
    Heat exchanger duty calculation.

    Q = ṁ · Cp · (T_out - T_in)

    Parameters
    ----------
    m_dot : float
        Mass flow rate [kg/s]
    Cp : float
        Heat capacity [J/(kg·K)]
    T_in : float
        Inlet temperature [K]
    T_out : float
        Outlet temperature [K]

    Returns
    -------
    dict
        Q: Heat duty [W]
        Q_kW: Heat duty [kW]
    """
    Q = m_dot * Cp * (T_out - T_in)

    return {
        'Q_W': float(Q),
        'Q_kW': float(Q / 1000),
        'm_dot': m_dot,
        'Cp': Cp,
        'T_in': T_in,
        'T_out': T_out,
        'delta_T': float(T_out - T_in),
        'equation': 'Q = ṁ·Cp·ΔT',
    }


def lmtd(T_h_in: float, T_h_out: float, T_c_in: float, T_c_out: float,
         flow_type: str = 'countercurrent') -> Dict[str, Any]:
    """
    Log Mean Temperature Difference.

    LMTD = (ΔT₁ - ΔT₂) / ln(ΔT₁/ΔT₂)

    Parameters
    ----------
    T_h_in : float
        Hot fluid inlet temperature [K]
    T_h_out : float
        Hot fluid outlet temperature [K]
    T_c_in : float
        Cold fluid inlet temperature [K]
    T_c_out : float
        Cold fluid outlet temperature [K]
    flow_type : str
        'countercurrent' (default) or 'cocurrent'

    Returns
    -------
    dict
        LMTD: Log mean temperature difference [K]
        delta_T1: Temperature difference at one end
        delta_T2: Temperature difference at other end
    """
    if flow_type == 'countercurrent':
        delta_T1 = T_h_in - T_c_out
        delta_T2 = T_h_out - T_c_in
    else:  # cocurrent
        delta_T1 = T_h_in - T_c_in
        delta_T2 = T_h_out - T_c_out

    # Handle case where delta_T1 ≈ delta_T2
    if abs(delta_T1 - delta_T2) < 1e-6:
        LMTD_val = (delta_T1 + delta_T2) / 2
    else:
        LMTD_val = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)

    return {
        'LMTD': float(LMTD_val),
        'delta_T1': float(delta_T1),
        'delta_T2': float(delta_T2),
        'T_h_in': T_h_in,
        'T_h_out': T_h_out,
        'T_c_in': T_c_in,
        'T_c_out': T_c_out,
        'flow_type': flow_type,
        'equation': 'LMTD = (ΔT₁-ΔT₂)/ln(ΔT₁/ΔT₂)',
    }


def effectiveness_ntu(NTU: float, C_ratio: float,
                      flow_type: str = 'countercurrent') -> Dict[str, Any]:
    """
    ε-NTU method for heat exchanger analysis.

    NTU = U·A / C_min
    C_ratio = C_min / C_max

    Parameters
    ----------
    NTU : float
        Number of transfer units
    C_ratio : float
        Heat capacity rate ratio (0 ≤ C_ratio ≤ 1)
    flow_type : str
        'countercurrent', 'cocurrent', 'crossflow', 'shell_and_tube'

    Returns
    -------
    dict
        effectiveness: Heat exchanger effectiveness ε
        Q_actual_ratio: Q_actual / Q_max
    """
    C_ratio = min(C_ratio, 1.0)

    if flow_type == 'countercurrent':
        if C_ratio < 1:
            epsilon = (1 - np.exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * np.exp(-NTU * (1 - C_ratio)))
        else:
            epsilon = NTU / (1 + NTU)

    elif flow_type == 'cocurrent':
        epsilon = (1 - np.exp(-NTU * (1 + C_ratio))) / (1 + C_ratio)

    elif flow_type == 'crossflow':
        # Approximation for unmixed fluids
        epsilon = 1 - np.exp((NTU**0.22 / C_ratio) * (np.exp(-C_ratio * NTU**0.78) - 1))

    else:  # shell_and_tube (1 shell pass, 2 tube passes)
        E = np.exp(-NTU * np.sqrt(1 + C_ratio**2))
        epsilon = 2 / (1 + C_ratio + np.sqrt(1 + C_ratio**2) * (1 + E) / (1 - E))

    return {
        'effectiveness': float(epsilon),
        'NTU': NTU,
        'C_ratio': C_ratio,
        'flow_type': flow_type,
        'equation': f'ε = f(NTU, C_ratio) for {flow_type}',
    }


def ntu_from_effectiveness(epsilon: float, C_ratio: float,
                           flow_type: str = 'countercurrent') -> Dict[str, Any]:
    """
    Calculate NTU from effectiveness.

    Inverse of ε-NTU relations.

    Parameters
    ----------
    epsilon : float
        Heat exchanger effectiveness
    C_ratio : float
        Heat capacity rate ratio
    flow_type : str
        'countercurrent', 'cocurrent'

    Returns
    -------
    dict
        NTU: Number of transfer units
    """
    if flow_type == 'countercurrent':
        if C_ratio < 1:
            NTU = (1 / (1 - C_ratio)) * np.log((1 - C_ratio * epsilon) / (1 - epsilon))
        else:
            NTU = epsilon / (1 - epsilon)

    elif flow_type == 'cocurrent':
        NTU = -np.log(1 - epsilon * (1 + C_ratio)) / (1 + C_ratio)

    else:
        NTU = None  # Iterative solution needed for other types

    return {
        'NTU': float(NTU) if NTU is not None else None,
        'effectiveness': epsilon,
        'C_ratio': C_ratio,
        'flow_type': flow_type,
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for energy balance calculations.
    """
    if all(k in kwargs for k in ['T_h_in', 'T_h_out', 'T_c_in', 'T_c_out']):
        return lmtd(kwargs['T_h_in'], kwargs['T_h_out'],
                    kwargs['T_c_in'], kwargs['T_c_out'],
                    kwargs.get('flow_type', 'countercurrent'))

    if 'NTU' in kwargs and 'C_ratio' in kwargs:
        return effectiveness_ntu(kwargs['NTU'], kwargs['C_ratio'],
                                kwargs.get('flow_type', 'countercurrent'))

    if 'n' in kwargs and 'Cp' in kwargs and 'T1' in kwargs and 'T2' in kwargs:
        return sensible_heat(kwargs['n'], kwargs['Cp'], kwargs['T1'], kwargs['T2'])

    return {'error': 'Insufficient parameters'}

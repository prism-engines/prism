"""
Faraday's Law and Electrochemical Applications

REQUIRES: n_electrons, molar_mass [kg/mol]

Electrolysis, electroplating, corrosion rates.
"""

import numpy as np
from typing import Dict, Any, Optional


# Physical constants
F = 96485.33212  # Faraday constant [C/mol]


def _nan_result(reason: str, keys: list) -> Dict[str, Any]:
    """Return NaN result with error reason."""
    result = {k: float('nan') for k in keys}
    result['error'] = reason
    return result


def faraday(I: float = None, t: float = None, M: float = None, n: int = None,
            efficiency: float = 1.0, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Faraday's Law of Electrolysis.

    REQUIRES: n (electrons), M (molar_mass)

    m = (I·t·M) / (n·F)

    Parameters
    ----------
    I : float
        Current [A]
    t : float
        Time [s]
    M : float
        Molar mass [kg/mol]. REQUIRED.
    n : int
        Number of electrons transferred. REQUIRED.
    efficiency : float
        Current efficiency (0-1)
    config : dict, optional
        Config with global_constants

    Returns
    -------
    dict
        mass: Mass deposited/dissolved [same units as M]
        moles: Moles reacted
        charge: Total charge passed [C]
    """
    # Get from config if not provided
    if config is not None:
        gc = config.get('global_constants', {})
        if M is None:
            M = gc.get('molar_mass')
        if n is None:
            n = gc.get('n_electrons')

    result_keys = ['mass', 'moles', 'charge']

    # VALIDATION
    if n is None:
        return _nan_result('Missing required constant: n_electrons', result_keys)
    if M is None or np.isnan(M):
        return _nan_result('Missing required constant: molar_mass [kg/mol]', result_keys)
    if I is None or np.isnan(I):
        return _nan_result('Missing current I [A]', result_keys)
    if t is None or np.isnan(t):
        return _nan_result('Missing time t [s]', result_keys)

    Q = I * t  # Charge [C]
    moles = Q / (n * F)
    mass = moles * M * efficiency

    return {
        'mass': float(mass),
        'moles': float(moles),
        'charge': float(Q),
        'I': I,
        't': t,
        'M': M,
        'n': n,
        'efficiency': efficiency,
        'equation': 'm = (I·t·M)/(n·F)',
    }


def faraday_constant() -> Dict[str, Any]:
    """
    Return Faraday constant and related values.

    F = N_A · e = 96485.33212 C/mol

    Returns
    -------
    dict
        F: Faraday constant [C/mol]
        e: Elementary charge [C]
        N_A: Avogadro constant [1/mol]
    """
    e = 1.602176634e-19  # Elementary charge [C]
    N_A = 6.02214076e23  # Avogadro [1/mol]

    return {
        'F': F,
        'e': e,
        'N_A': N_A,
        'F_calculated': e * N_A,
        'unit': 'C/mol',
    }


def coulombic_efficiency(Q_actual: float, Q_theoretical: float) -> Dict[str, Any]:
    """
    Coulombic (current) efficiency.

    η_C = Q_actual / Q_theoretical × 100%

    Parameters
    ----------
    Q_actual : float
        Actual charge used for desired reaction [C or Ah]
    Q_theoretical : float
        Theoretical charge required [C or Ah]

    Returns
    -------
    dict
        efficiency: Coulombic efficiency (0-1)
        efficiency_percent: Efficiency in %
        charge_lost: Q lost to side reactions
    """
    efficiency = Q_actual / Q_theoretical
    charge_lost = Q_theoretical - Q_actual

    return {
        'efficiency': float(efficiency),
        'efficiency_percent': float(efficiency * 100),
        'Q_actual': Q_actual,
        'Q_theoretical': Q_theoretical,
        'charge_lost': float(charge_lost),
        'equation': 'η_C = Q_actual/Q_theoretical',
    }


def energy_efficiency(E_actual: float, E_theoretical: float,
                      Q_actual: float = None, Q_theoretical: float = None) -> Dict[str, Any]:
    """
    Energy efficiency of electrochemical process.

    η_E = (E_theoretical / E_actual) × η_C

    For batteries:
    η_E = (V_discharge · Q_discharge) / (V_charge · Q_charge)

    Parameters
    ----------
    E_actual : float
        Actual cell voltage [V]
    E_theoretical : float
        Theoretical (reversible) voltage [V]
    Q_actual : float, optional
        Actual charge [C or Ah]
    Q_theoretical : float, optional
        Theoretical charge [C or Ah]

    Returns
    -------
    dict
        voltage_efficiency: E_theoretical / E_actual
        coulombic_efficiency: Q_actual / Q_theoretical (if provided)
        total_efficiency: Combined efficiency
    """
    voltage_eff = E_theoretical / E_actual if E_actual != 0 else 0

    if Q_actual is not None and Q_theoretical is not None:
        coulombic_eff = Q_actual / Q_theoretical
        total_eff = voltage_eff * coulombic_eff
    else:
        coulombic_eff = 1.0
        total_eff = voltage_eff

    return {
        'voltage_efficiency': float(voltage_eff),
        'coulombic_efficiency': float(coulombic_eff),
        'total_efficiency': float(total_eff),
        'efficiency_percent': float(total_eff * 100),
        'overpotential': float(E_actual - E_theoretical),
        'equation': 'η_E = (E_th/E_act) × η_C',
    }


def corrosion_rate(i_corr: float, M: float, n: int, rho: float) -> Dict[str, Any]:
    """
    Corrosion rate from corrosion current density.

    CR = (i_corr · M) / (n · F · ρ)

    Parameters
    ----------
    i_corr : float
        Corrosion current density [A/m²]
    M : float
        Molar mass [g/mol]
    n : int
        Number of electrons (valence)
    rho : float
        Density [g/cm³]

    Returns
    -------
    dict
        CR_m_s: Corrosion rate [m/s]
        CR_mm_year: Corrosion rate [mm/year]
        CR_mpy: Corrosion rate [mils per year]
        mass_loss_rate: Mass loss rate [g/(m²·s)]
    """
    # Corrosion rate in m/s
    CR_m_s = (i_corr * M) / (n * F * rho * 1e6)  # Convert g/cm³ to kg/m³

    # Convert to mm/year
    seconds_per_year = 365.25 * 24 * 3600
    CR_mm_year = CR_m_s * 1000 * seconds_per_year

    # mils per year (1 mil = 0.001 inch = 0.0254 mm)
    CR_mpy = CR_mm_year / 0.0254

    # Mass loss rate
    mass_loss = (i_corr * M) / (n * F)  # g/(m²·s)

    return {
        'CR_m_s': float(CR_m_s),
        'CR_mm_year': float(CR_mm_year),
        'CR_mpy': float(CR_mpy),
        'mass_loss_rate': float(mass_loss),
        'i_corr': i_corr,
        'equation': 'CR = (i_corr·M)/(n·F·ρ)',
    }


def electroplating_thickness(I: float, t: float, A: float, M: float,
                             n: int, rho: float, efficiency: float = 1.0) -> Dict[str, Any]:
    """
    Electroplating thickness from Faraday's law.

    δ = (I·t·M) / (n·F·A·ρ)

    Parameters
    ----------
    I : float
        Current [A]
    t : float
        Time [s]
    A : float
        Electrode area [m²]
    M : float
        Molar mass [g/mol]
    n : int
        Electrons per metal ion
    rho : float
        Deposit density [g/cm³]
    efficiency : float
        Current efficiency (0-1)

    Returns
    -------
    dict
        thickness_m: Deposit thickness [m]
        thickness_um: Deposit thickness [μm]
        mass: Total mass deposited [g]
    """
    mass = faraday(I, t, M, n, efficiency)['mass']

    # Volume = mass / density
    volume = mass / (rho * 1e6)  # Convert g/cm³ to g/m³

    # Thickness = volume / area
    thickness = volume / A

    return {
        'thickness_m': float(thickness),
        'thickness_um': float(thickness * 1e6),
        'thickness_mil': float(thickness * 1e6 / 25.4),
        'mass_g': float(mass),
        'volume_m3': float(volume),
        'I': I,
        't': t,
        'A': A,
        'equation': 'δ = (I·t·M)/(n·F·A·ρ)',
    }


def electrolysis_power(I: float, E_cell: float, efficiency: float = 1.0) -> Dict[str, Any]:
    """
    Power consumption in electrolysis.

    P = I · E_cell / η

    Parameters
    ----------
    I : float
        Current [A]
    E_cell : float
        Cell voltage [V]
    efficiency : float
        Overall efficiency (0-1)

    Returns
    -------
    dict
        P_W: Power [W]
        P_kW: Power [kW]
        energy_per_hour_kWh: Energy consumption [kWh/h]
    """
    P = I * E_cell / efficiency

    return {
        'P_W': float(P),
        'P_kW': float(P / 1000),
        'energy_per_hour_kWh': float(P / 1000),
        'I': I,
        'E_cell': E_cell,
        'efficiency': efficiency,
        'equation': 'P = I·E_cell/η',
    }


def specific_energy_consumption(E_cell: float, M: float, n: int,
                                efficiency: float = 1.0) -> Dict[str, Any]:
    """
    Specific energy consumption for electrolysis.

    SEC = (n · F · E_cell) / (M · 3600 · η)  [kWh/kg]

    Parameters
    ----------
    E_cell : float
        Cell voltage [V]
    M : float
        Molar mass [g/mol]
    n : int
        Number of electrons
    efficiency : float
        Current efficiency (0-1)

    Returns
    -------
    dict
        SEC_kWh_kg: Specific energy consumption [kWh/kg]
        SEC_MJ_kg: Specific energy consumption [MJ/kg]
    """
    SEC_kWh_kg = (n * F * E_cell) / (M * 3600 * efficiency)
    SEC_MJ_kg = SEC_kWh_kg * 3.6

    return {
        'SEC_kWh_kg': float(SEC_kWh_kg),
        'SEC_MJ_kg': float(SEC_MJ_kg),
        'E_cell': E_cell,
        'M': M,
        'n': n,
        'efficiency': efficiency,
        'equation': 'SEC = (n·F·E)/(M·3600·η)',
    }


def compute(signal: np.ndarray = None, I: float = None, t: float = None,
            M: float = None, n: int = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for Faraday's law calculations.
    """
    if I is not None and t is not None and M is not None and n is not None:
        return faraday(I, t, M, n, kwargs.get('efficiency', 1.0))

    if 'i_corr' in kwargs and M is not None and n is not None:
        return corrosion_rate(kwargs['i_corr'], M, n, kwargs.get('rho', 7.87))

    return {'error': 'Insufficient parameters'}

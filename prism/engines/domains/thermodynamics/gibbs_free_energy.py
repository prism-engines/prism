"""
Gibbs Free Energy Engine — THE REAL EQUATION

G = H - TS  [J] or [J/mol]

Requires REAL thermodynamic data:
    - Temperature T [K]
    - Enthalpy H [J]
    - Entropy S [J/K]

Or computable from T, P, V for ideal gas.

ΔG < 0: Spontaneous process
ΔG = 0: Equilibrium
ΔG > 0: Non-spontaneous
"""

import numpy as np
from typing import Dict, Optional

# Physical constants
R = 8.314462618  # J/(mol·K) Gas constant


def compute_gibbs_free_energy(
    enthalpy: np.ndarray,
    entropy: np.ndarray,
    temperature: np.ndarray,
) -> Dict:
    """
    Compute Gibbs free energy: G = H - TS

    THIS IS THE REAL EQUATION.

    Args:
        enthalpy: H [J] or [J/mol]
        entropy: S [J/K] or [J/(mol·K)]
        temperature: T [K]

    Returns:
        Dict with Gibbs free energy and analysis
    """
    H = np.asarray(enthalpy, dtype=float)
    S = np.asarray(entropy, dtype=float)
    T = np.asarray(temperature, dtype=float)

    # Validate temperature
    if np.any(T <= 0):
        return {
            'gibbs_free_energy': None,
            'error': 'Temperature must be positive [K]',
            'equation': 'G = H - TS',
        }

    # Gibbs free energy
    TS = T * S
    G = H - TS

    # Spontaneity analysis
    G_mean = np.nanmean(G)
    H_mean = np.nanmean(H)
    TS_mean = np.nanmean(TS)
    is_spontaneous = G_mean < 0

    return {
        'gibbs_free_energy': G,
        'enthalpy': H,
        'entropy': S,
        'temperature': T,
        'TS_term': TS,

        'mean_G': float(G_mean),
        'mean_H': float(H_mean),
        'mean_S': float(np.nanmean(S)),
        'mean_T': float(np.nanmean(T)),
        'mean_TS': float(TS_mean),

        'is_spontaneous': bool(is_spontaneous),
        'enthalpy_driven': bool(H_mean < 0 and TS_mean > 0),
        'entropy_driven': bool(H_mean > 0 and TS_mean > H_mean),

        'equation': 'G = H - TS',
        'units': 'J or J/mol',
    }


def compute_gibbs_ideal_gas(
    temperature: np.ndarray,
    pressure: np.ndarray,
    n_moles: float = 1.0,
    Cp: float = 29.1,  # J/(mol·K) for diatomic gas
    T_ref: float = 298.15,
    P_ref: float = 101325.0,
    H_ref: float = 0.0,
    S_ref: float = 0.0,
) -> Dict:
    """
    Compute Gibbs free energy for ideal gas from T and P.

    H = H_ref + Cp*(T - T_ref)
    S = S_ref + Cp*ln(T/T_ref) - R*ln(P/P_ref)
    G = H - TS

    Args:
        temperature: T [K]
        pressure: P [Pa]
        n_moles: Amount of substance [mol]
        Cp: Molar heat capacity at constant pressure [J/(mol·K)]
        T_ref: Reference temperature [K]
        P_ref: Reference pressure [Pa]
        H_ref: Reference enthalpy [J/mol]
        S_ref: Reference entropy [J/(mol·K)]
    """
    T = np.asarray(temperature, dtype=float)
    P = np.asarray(pressure, dtype=float)

    # Validate
    if np.any(T <= 0):
        return {
            'gibbs_free_energy': None,
            'error': 'Temperature must be positive [K]',
            'equation': 'G = H - TS (ideal gas)',
        }
    if np.any(P <= 0):
        return {
            'gibbs_free_energy': None,
            'error': 'Pressure must be positive [Pa]',
            'equation': 'G = H - TS (ideal gas)',
        }

    # Enthalpy: H = H_ref + Cp*(T - T_ref)
    H = n_moles * (H_ref + Cp * (T - T_ref))

    # Entropy: S = S_ref + Cp*ln(T/T_ref) - R*ln(P/P_ref)
    S = n_moles * (S_ref + Cp * np.log(T / T_ref) - R * np.log(P / P_ref))

    # Gibbs: G = H - TS
    G = H - T * S

    return {
        'gibbs_free_energy': G,
        'enthalpy': H,
        'entropy': S,
        'temperature': T,
        'pressure': P,

        'mean_G': float(np.nanmean(G)),
        'mean_H': float(np.nanmean(H)),
        'mean_S': float(np.nanmean(S)),
        'is_spontaneous': bool(np.nanmean(G) < 0),

        'n_moles': n_moles,
        'Cp': Cp,
        'R': R,
        'T_ref': T_ref,
        'P_ref': P_ref,
        'equation': 'G = H - TS (ideal gas)',
    }


def compute_gibbs_change(
    G_initial: np.ndarray,
    G_final: np.ndarray,
    temperature: float = 298.15,
) -> Dict:
    """
    Compute change in Gibbs free energy: ΔG = G_final - G_initial

    ΔG < 0: Spontaneous (favorable)
    ΔG = 0: Equilibrium
    ΔG > 0: Non-spontaneous (unfavorable)
    """
    G_i = np.asarray(G_initial, dtype=float)
    G_f = np.asarray(G_final, dtype=float)

    delta_G = G_f - G_i
    mean_delta_G = np.nanmean(delta_G)

    # Equilibrium constant from ΔG = -RT ln(K)
    # ln(K) = -ΔG / RT
    K_ln = -mean_delta_G / (R * temperature) if temperature > 0 else None

    return {
        'delta_G': delta_G,
        'mean_delta_G': float(mean_delta_G),

        'is_spontaneous': bool(mean_delta_G < 0),
        'is_equilibrium': bool(np.abs(mean_delta_G) < 1e-6),
        'is_non_spontaneous': bool(mean_delta_G > 0),

        'equilibrium_constant_ln': float(K_ln) if K_ln is not None else None,
        'equilibrium_constant': float(np.exp(K_ln)) if K_ln is not None and np.abs(K_ln) < 700 else None,

        'equation': 'ΔG = G_final - G_initial',
    }


def compute_chemical_potential(
    temperature: np.ndarray,
    pressure: np.ndarray,
    mu_ref: float,
    P_ref: float = 101325.0,
) -> Dict:
    """
    Chemical potential for ideal gas: μ = μ° + RT*ln(P/P°)
    """
    T = np.asarray(temperature, dtype=float)
    P = np.asarray(pressure, dtype=float)

    if np.any(T <= 0) or np.any(P <= 0):
        return {
            'chemical_potential': None,
            'error': 'Temperature and pressure must be positive',
            'equation': 'μ = μ° + RT·ln(P/P°)',
        }

    mu = mu_ref + R * T * np.log(P / P_ref)

    return {
        'chemical_potential': mu,
        'mean_chemical_potential': float(np.nanmean(mu)),
        'mu_ref': mu_ref,
        'P_ref': P_ref,
        'units': 'J/mol',
        'equation': 'μ = μ° + RT·ln(P/P°)',
    }


def compute(
    temperature: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    enthalpy: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    n_moles: float = 1.0,
    Cp: float = 29.1,
) -> Dict:
    """
    Main compute function for Gibbs free energy.

    If enthalpy and entropy provided: G = H - TS
    If only T and P provided: Use ideal gas approximation

    Args:
        temperature: T [K]
        pressure: P [Pa] (optional, for ideal gas)
        enthalpy: H [J] (optional)
        entropy: S [J/K] (optional)
        n_moles: Amount of substance [mol]
        Cp: Molar heat capacity [J/(mol·K)]

    Returns:
        Dict with Gibbs free energy metrics
    """
    T = np.asarray(temperature, dtype=float)

    # If we have real enthalpy and entropy, use them
    if enthalpy is not None and entropy is not None:
        return compute_gibbs_free_energy(enthalpy, entropy, temperature)

    # Otherwise, use ideal gas approximation with T and P
    if pressure is not None:
        return compute_gibbs_ideal_gas(
            temperature=temperature,
            pressure=pressure,
            n_moles=n_moles,
            Cp=Cp,
        )

    return {
        'gibbs_free_energy': None,
        'error': 'Need either (enthalpy, entropy) or (pressure) with temperature',
        'equation': 'G = H - TS',
    }

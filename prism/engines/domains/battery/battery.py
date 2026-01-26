"""
Battery Analysis Engines

State of charge, state of health, C-rate, capacity fade, impedance.
"""

import numpy as np
from typing import Dict, Any, Optional, List


def state_of_charge(Q_remaining: float, Q_total: float,
                    method: str = 'coulomb_counting') -> Dict[str, Any]:
    """
    State of Charge (SOC) calculation.

    SOC = Q_remaining / Q_total × 100%

    Parameters
    ----------
    Q_remaining : float
        Remaining capacity [Ah]
    Q_total : float
        Total (rated) capacity [Ah]
    method : str
        Calculation method: 'coulomb_counting', 'ocv', 'model'

    Returns
    -------
    dict
        SOC: State of charge (0-1)
        SOC_percent: State of charge (0-100%)
        Q_remaining: Remaining capacity [Ah]
        Q_discharged: Discharged capacity [Ah]
    """
    SOC = Q_remaining / Q_total
    SOC = np.clip(SOC, 0, 1)

    return {
        'SOC': float(SOC),
        'SOC_percent': float(SOC * 100),
        'Q_remaining': Q_remaining,
        'Q_total': Q_total,
        'Q_discharged': float(Q_total - Q_remaining),
        'method': method,
        'equation': 'SOC = Q_remaining/Q_total',
    }


def state_of_health(Q_actual: float, Q_nominal: float,
                    R_actual: float = None, R_initial: float = None) -> Dict[str, Any]:
    """
    State of Health (SOH) calculation.

    SOH_capacity = Q_actual / Q_nominal × 100%
    SOH_resistance = R_initial / R_actual × 100%

    Parameters
    ----------
    Q_actual : float
        Actual measured capacity [Ah]
    Q_nominal : float
        Nominal (rated) capacity [Ah]
    R_actual : float, optional
        Current internal resistance [Ω]
    R_initial : float, optional
        Initial internal resistance [Ω]

    Returns
    -------
    dict
        SOH_capacity: Capacity-based SOH (0-1)
        SOH_resistance: Resistance-based SOH (0-1)
        capacity_fade: Capacity loss percentage
        end_of_life: Whether SOH < 80% (typical EOL criterion)
    """
    SOH_capacity = Q_actual / Q_nominal
    capacity_fade = (1 - SOH_capacity) * 100

    result = {
        'SOH_capacity': float(SOH_capacity),
        'SOH_capacity_percent': float(SOH_capacity * 100),
        'Q_actual': Q_actual,
        'Q_nominal': Q_nominal,
        'capacity_fade_percent': float(capacity_fade),
        'end_of_life': SOH_capacity < 0.8,
        'equation': 'SOH = Q_actual/Q_nominal',
    }

    if R_actual is not None and R_initial is not None:
        SOH_resistance = R_initial / R_actual
        result['SOH_resistance'] = float(SOH_resistance)
        result['SOH_resistance_percent'] = float(SOH_resistance * 100)
        result['resistance_increase_percent'] = float((R_actual / R_initial - 1) * 100)

    return result


def c_rate(I: float, Q_nominal: float) -> Dict[str, Any]:
    """
    C-rate calculation.

    C-rate = I / Q_nominal

    1C = fully discharge in 1 hour
    2C = fully discharge in 0.5 hours
    C/2 = fully discharge in 2 hours

    Parameters
    ----------
    I : float
        Current [A] (positive for discharge)
    Q_nominal : float
        Nominal capacity [Ah]

    Returns
    -------
    dict
        C_rate: C-rate value
        time_to_empty: Time to fully discharge [h]
        discharge_time_min: Time to fully discharge [min]
    """
    C_rate = abs(I) / Q_nominal
    time_to_empty = Q_nominal / abs(I) if I != 0 else float('inf')

    return {
        'C_rate': float(C_rate),
        'C_rate_notation': f'{C_rate:.1f}C' if C_rate >= 1 else f'C/{1/C_rate:.1f}',
        'I': I,
        'Q_nominal': Q_nominal,
        'time_to_empty_h': float(time_to_empty),
        'time_to_empty_min': float(time_to_empty * 60),
        'equation': 'C-rate = I/Q_nominal',
    }


def peukert(I: float, Q_nominal: float, I_nominal: float,
            k: float = 1.1) -> Dict[str, Any]:
    """
    Peukert's Law for battery capacity vs discharge rate.

    t = (Q_nominal / I) × (I_nominal / I)^(k-1)

    Effective capacity at current I:
    Q_eff = Q_nominal × (I_nominal / I)^(k-1)

    Parameters
    ----------
    I : float
        Discharge current [A]
    Q_nominal : float
        Nominal capacity at I_nominal [Ah]
    I_nominal : float
        Nominal discharge current [A] (typically C/20 or C/10)
    k : float
        Peukert exponent (1.0-1.4, typically ~1.1-1.3 for lead-acid)

    Returns
    -------
    dict
        Q_effective: Effective capacity at current I [Ah]
        t_discharge: Discharge time [h]
        capacity_ratio: Q_effective / Q_nominal
    """
    capacity_ratio = (I_nominal / I) ** (k - 1)
    Q_effective = Q_nominal * capacity_ratio
    t_discharge = Q_effective / I

    return {
        'Q_effective': float(Q_effective),
        'Q_nominal': Q_nominal,
        'capacity_ratio': float(capacity_ratio),
        't_discharge_h': float(t_discharge),
        'I': I,
        'I_nominal': I_nominal,
        'k': k,
        'equation': 'Q_eff = Q_nom × (I_nom/I)^(k-1)',
    }


def internal_resistance_calc(V_ocv: float, V_load: float, I: float) -> Dict[str, Any]:
    """
    Internal resistance from voltage drop under load.

    R_i = (V_OCV - V_load) / I

    Parameters
    ----------
    V_ocv : float
        Open circuit voltage [V]
    V_load : float
        Voltage under load [V]
    I : float
        Load current [A]

    Returns
    -------
    dict
        R_internal: Internal resistance [Ω]
        R_internal_mOhm: Internal resistance [mΩ]
        voltage_drop: V_OCV - V_load [V]
        power_loss: I² × R [W]
    """
    R_internal = (V_ocv - V_load) / abs(I) if I != 0 else 0
    power_loss = I**2 * R_internal

    return {
        'R_internal': float(R_internal),
        'R_internal_mOhm': float(R_internal * 1000),
        'V_ocv': V_ocv,
        'V_load': V_load,
        'I': I,
        'voltage_drop': float(V_ocv - V_load),
        'power_loss_W': float(power_loss),
        'equation': 'R_i = (V_OCV - V_load)/I',
    }


def capacity_fade(cycles: np.ndarray, capacity: np.ndarray,
                  Q_initial: float = None) -> Dict[str, Any]:
    """
    Analyze capacity fade over cycles.

    Common models:
    - Linear: Q = Q_0 - a×n
    - Square root: Q = Q_0 - a×√n
    - Power: Q = Q_0 × n^(-b)

    Parameters
    ----------
    cycles : array
        Cycle numbers
    capacity : array
        Measured capacity at each cycle [Ah]
    Q_initial : float, optional
        Initial capacity (if not provided, uses first data point)

    Returns
    -------
    dict
        fade_rate_per_cycle: Average capacity loss per cycle
        fade_rate_percent: Percentage loss per cycle
        cycles_to_eol: Estimated cycles to 80% capacity
        current_soh: Current SOH
    """
    cycles = np.asarray(cycles)
    capacity = np.asarray(capacity)

    if Q_initial is None:
        Q_initial = capacity[0]

    # Linear fit for fade rate
    if len(cycles) >= 2:
        fade_rate = (capacity[0] - capacity[-1]) / (cycles[-1] - cycles[0])
        fade_rate_percent = fade_rate / Q_initial * 100
    else:
        fade_rate = 0
        fade_rate_percent = 0

    # Current SOH
    current_soh = capacity[-1] / Q_initial

    # Estimate cycles to 80% capacity (EOL)
    Q_eol = 0.8 * Q_initial
    if fade_rate > 0:
        cycles_to_eol = (Q_initial - Q_eol) / fade_rate + cycles[0]
    else:
        cycles_to_eol = float('inf')

    return {
        'fade_rate_per_cycle': float(fade_rate),
        'fade_rate_percent_per_cycle': float(fade_rate_percent),
        'total_fade_percent': float((1 - current_soh) * 100),
        'current_soh': float(current_soh),
        'Q_initial': Q_initial,
        'Q_current': float(capacity[-1]),
        'cycles_to_eol': float(cycles_to_eol),
        'cycles_completed': int(cycles[-1]),
    }


def cycle_life(fade_rate_per_cycle: float, Q_initial: float,
               eol_criterion: float = 0.8) -> Dict[str, Any]:
    """
    Estimate cycle life to end-of-life criterion.

    Cycles to EOL = (Q_initial × (1 - EOL)) / fade_rate

    Parameters
    ----------
    fade_rate_per_cycle : float
        Capacity fade per cycle [Ah/cycle]
    Q_initial : float
        Initial capacity [Ah]
    eol_criterion : float
        End-of-life SOH criterion (default 0.8 = 80%)

    Returns
    -------
    dict
        estimated_cycles: Cycles to reach EOL
        fade_per_100_cycles: Capacity fade per 100 cycles [%]
    """
    capacity_loss_to_eol = Q_initial * (1 - eol_criterion)

    if fade_rate_per_cycle > 0:
        estimated_cycles = capacity_loss_to_eol / fade_rate_per_cycle
    else:
        estimated_cycles = float('inf')

    fade_per_100 = (fade_rate_per_cycle * 100 / Q_initial) * 100

    return {
        'estimated_cycles': float(estimated_cycles),
        'Q_initial': Q_initial,
        'Q_at_eol': float(Q_initial * eol_criterion),
        'fade_per_100_cycles_percent': float(fade_per_100),
        'eol_criterion': eol_criterion,
        'equation': 'N_EOL = Q_init×(1-EOL)/fade_rate',
    }


def impedance_spectrum(frequencies: np.ndarray, Z_real: np.ndarray,
                       Z_imag: np.ndarray) -> Dict[str, Any]:
    """
    Analyze electrochemical impedance spectrum (EIS).

    Extracts key parameters from Nyquist plot:
    - R_ohmic: High-frequency intercept (ohmic resistance)
    - R_ct: Charge transfer resistance (semicircle diameter)
    - Z_Warburg: Low-frequency diffusion impedance

    Parameters
    ----------
    frequencies : array
        Frequencies [Hz]
    Z_real : array
        Real part of impedance [Ω]
    Z_imag : array
        Imaginary part of impedance [Ω] (typically negative)

    Returns
    -------
    dict
        R_ohmic: Ohmic resistance [Ω]
        R_ct: Charge transfer resistance [Ω]
        R_total: Total DC resistance [Ω]
        characteristic_frequency: Frequency at semicircle peak [Hz]
    """
    frequencies = np.asarray(frequencies)
    Z_real = np.asarray(Z_real)
    Z_imag = np.asarray(Z_imag)

    # Ohmic resistance: high-frequency intercept
    hf_idx = np.argmax(frequencies)
    R_ohmic = Z_real[hf_idx]

    # Find semicircle peak (maximum -Z_imag)
    peak_idx = np.argmin(Z_imag)  # Most negative
    f_peak = frequencies[peak_idx]

    # Charge transfer resistance estimate (simplified)
    # In a full semicircle, R_ct ≈ 2 × (Z_real at peak - R_ohmic)
    R_ct_estimate = 2 * (Z_real[peak_idx] - R_ohmic)

    # Low-frequency intercept (DC resistance)
    lf_idx = np.argmin(frequencies)
    R_total = Z_real[lf_idx]

    # Warburg coefficient (45° line slope at low frequencies)
    # Z_W = σ × (1-j) / √ω
    lf_mask = frequencies < f_peak / 10
    if np.sum(lf_mask) > 1:
        sigma = np.mean(Z_real[lf_mask] - Z_imag[lf_mask]) / np.sqrt(2)
    else:
        sigma = None

    return {
        'R_ohmic': float(R_ohmic),
        'R_ohmic_mOhm': float(R_ohmic * 1000),
        'R_ct': float(R_ct_estimate),
        'R_ct_mOhm': float(R_ct_estimate * 1000),
        'R_total': float(R_total),
        'characteristic_frequency': float(f_peak),
        'time_constant': float(1 / (2 * np.pi * f_peak)) if f_peak > 0 else None,
        'warburg_coefficient': float(sigma) if sigma else None,
    }


def ocv_soc_curve(V: np.ndarray, SOC: np.ndarray) -> Dict[str, Any]:
    """
    Characterize OCV-SOC relationship.

    Parameters
    ----------
    V : array
        Open circuit voltage [V]
    SOC : array
        State of charge (0-1)

    Returns
    -------
    dict
        V_min: Minimum voltage (at SOC=0)
        V_max: Maximum voltage (at SOC=1)
        V_nominal: Voltage at SOC=0.5
        dV_dSOC: Average slope [V per 1% SOC]
    """
    V = np.asarray(V)
    SOC = np.asarray(SOC)

    # Sort by SOC
    sort_idx = np.argsort(SOC)
    V = V[sort_idx]
    SOC = SOC[sort_idx]

    # Interpolate key values
    V_min = V[0]
    V_max = V[-1]

    # Voltage at 50% SOC
    V_nominal = np.interp(0.5, SOC, V)

    # Average slope
    dV_dSOC = (V_max - V_min) / (SOC[-1] - SOC[0]) / 100  # per 1% SOC

    return {
        'V_min': float(V_min),
        'V_max': float(V_max),
        'V_nominal': float(V_nominal),
        'V_range': float(V_max - V_min),
        'dV_dSOC_percent': float(dV_dSOC),
        'equation': 'V = f(SOC)',
    }


def compute(signal: np.ndarray = None, Q_remaining: float = None,
            Q_total: float = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for battery analysis.
    """
    if Q_remaining is not None and Q_total is not None:
        return state_of_charge(Q_remaining, Q_total)

    if 'Q_actual' in kwargs and 'Q_nominal' in kwargs:
        return state_of_health(kwargs['Q_actual'], kwargs['Q_nominal'],
                              kwargs.get('R_actual'), kwargs.get('R_initial'))

    if 'I' in kwargs and 'Q_nominal' in kwargs:
        return c_rate(kwargs['I'], kwargs['Q_nominal'])

    return {'error': 'Insufficient parameters'}

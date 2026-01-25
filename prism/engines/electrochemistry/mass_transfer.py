"""
Electrochemical Mass Transfer Engines

Limiting current density, rotating disk electrode, concentration overpotential.
"""

import numpy as np
from typing import Dict, Any, Optional


# Physical constants
F = 96485.33212  # Faraday constant [C/mol]
R = 8.314462618  # Gas constant [J/(mol·K)]


def limiting_current_density(n: int, D: float, C_b: float,
                             delta: float) -> Dict[str, Any]:
    """
    Limiting current density from mass transfer.

    i_L = n·F·D·C_b / δ

    At limiting current, surface concentration → 0.

    Parameters
    ----------
    n : int
        Number of electrons transferred
    D : float
        Diffusion coefficient [m²/s]
    C_b : float
        Bulk concentration [mol/m³]
    delta : float
        Diffusion layer thickness [m]

    Returns
    -------
    dict
        i_L: Limiting current density [A/m²]
        mass_transfer_coeff: k_m = D/δ [m/s]
    """
    k_m = D / delta
    i_L = n * F * D * C_b / delta

    return {
        'i_L': float(i_L),
        'i_L_mA_cm2': float(i_L / 10),  # Convert to mA/cm²
        'k_m': float(k_m),
        'n': n,
        'D': D,
        'C_b': C_b,
        'delta': delta,
        'equation': 'i_L = nFDC_b/δ',
    }


def rotating_disk(n: int, D: float, C_b: float, omega: float,
                  nu: float) -> Dict[str, Any]:
    """
    Levich equation for rotating disk electrode.

    i_L = 0.62 · n · F · D^(2/3) · ω^(1/2) · ν^(-1/6) · C_b

    Parameters
    ----------
    n : int
        Number of electrons transferred
    D : float
        Diffusion coefficient [m²/s]
    C_b : float
        Bulk concentration [mol/m³]
    omega : float
        Angular velocity [rad/s]
    nu : float
        Kinematic viscosity [m²/s]

    Returns
    -------
    dict
        i_L: Levich limiting current density [A/m²]
        delta: Diffusion layer thickness [m]
        rotation_rpm: Rotation rate in RPM
    """
    # Levich equation
    i_L = 0.62 * n * F * D**(2/3) * omega**(1/2) * nu**(-1/6) * C_b

    # Diffusion layer thickness
    delta = 1.61 * D**(1/3) * nu**(1/6) * omega**(-1/2)

    # Convert omega to RPM
    rpm = omega * 60 / (2 * np.pi)

    return {
        'i_L': float(i_L),
        'i_L_mA_cm2': float(i_L / 10),
        'delta': float(delta),
        'delta_um': float(delta * 1e6),
        'omega': omega,
        'rotation_rpm': float(rpm),
        'n': n,
        'D': D,
        'C_b': C_b,
        'nu': nu,
        'equation': 'i_L = 0.62·n·F·D^(2/3)·ω^(1/2)·ν^(-1/6)·C_b',
    }


def levich_plot(n: int, D: float, C_b: float, nu: float,
                omega_range: np.ndarray = None) -> Dict[str, Any]:
    """
    Generate Levich plot data (i_L vs ω^0.5).

    Used to verify mass-transfer-limited kinetics.

    Parameters
    ----------
    n : int
        Number of electrons
    D : float
        Diffusion coefficient [m²/s]
    C_b : float
        Bulk concentration [mol/m³]
    nu : float
        Kinematic viscosity [m²/s]
    omega_range : array, optional
        Angular velocities to calculate [rad/s]

    Returns
    -------
    dict
        omega_sqrt: sqrt(ω) values
        i_L: Corresponding limiting currents
        slope: Levich slope
    """
    if omega_range is None:
        # Default: 100 to 10000 RPM
        rpm_range = np.array([100, 500, 1000, 2000, 4000, 6000, 10000])
        omega_range = rpm_range * 2 * np.pi / 60

    omega_sqrt = np.sqrt(omega_range)
    i_L = 0.62 * n * F * D**(2/3) * omega_sqrt * nu**(-1/6) * C_b

    # Levich slope
    slope = 0.62 * n * F * D**(2/3) * nu**(-1/6) * C_b

    return {
        'omega_sqrt': omega_sqrt.tolist(),
        'i_L': i_L.tolist(),
        'slope': float(slope),
        'D_from_slope': 'D = (slope/(0.62·n·F·ν^(-1/6)·C_b))^(3/2)',
    }


def koutecky_levich(i: float, i_k: float, i_L: float) -> Dict[str, Any]:
    """
    Koutecky-Levich equation for mixed kinetic-diffusion control.

    1/i = 1/i_k + 1/i_L

    Parameters
    ----------
    i : float
        Measured current density [A/m²]
    i_k : float
        Kinetic current density [A/m²]
    i_L : float
        Limiting (mass transfer) current density [A/m²]

    Returns
    -------
    dict
        i_predicted: Predicted current from mixed control
        kinetic_fraction: Fraction of control from kinetics
        diffusion_fraction: Fraction from mass transfer
    """
    # From Koutecky-Levich
    i_predicted = 1 / (1/i_k + 1/i_L)

    kinetic_frac = (1/i_L) / (1/i_k + 1/i_L)
    diffusion_frac = (1/i_k) / (1/i_k + 1/i_L)

    return {
        'i_predicted': float(i_predicted),
        'i_measured': i,
        'i_k': i_k,
        'i_L': i_L,
        'kinetic_fraction': float(kinetic_frac),
        'diffusion_fraction': float(diffusion_frac),
        'controlling': 'kinetic' if kinetic_frac > 0.5 else 'diffusion',
        'equation': '1/i = 1/i_k + 1/i_L',
    }


def concentration_overpotential(i: float, i_L: float, n: int,
                                T: float = 298.15) -> Dict[str, Any]:
    """
    Concentration overpotential due to mass transfer.

    η_conc = (RT/nF) · ln(1 - i/i_L)

    Parameters
    ----------
    i : float
        Current density [A/m²]
    i_L : float
        Limiting current density [A/m²]
    n : int
        Number of electrons
    T : float
        Temperature [K]

    Returns
    -------
    dict
        eta_conc: Concentration overpotential [V]
        C_s_over_C_b: Surface to bulk concentration ratio
    """
    if abs(i) >= abs(i_L):
        # At or beyond limiting current
        return {
            'eta_conc': float('-inf') if i > 0 else float('inf'),
            'C_s_over_C_b': 0.0,
            'i': i,
            'i_L': i_L,
            'note': 'At or beyond limiting current',
        }

    i_ratio = abs(i) / abs(i_L)
    C_s_over_C_b = 1 - i_ratio

    eta_conc = (R * T / (n * F)) * np.log(C_s_over_C_b)

    return {
        'eta_conc': float(eta_conc),
        'eta_conc_mV': float(eta_conc * 1000),
        'C_s_over_C_b': float(C_s_over_C_b),
        'i_over_i_L': float(i_ratio),
        'i': i,
        'i_L': i_L,
        'n': n,
        'T': T,
        'equation': 'η_conc = (RT/nF)·ln(1 - i/i_L)',
    }


def diffusion_layer_thickness(D: float, t: float = None, omega: float = None,
                              nu: float = None, mode: str = 'time') -> Dict[str, Any]:
    """
    Diffusion layer thickness estimates.

    Modes:
    - time: δ = √(πDt) (semi-infinite diffusion)
    - steady: δ = D/k_m (steady state with known k_m)
    - rotating: δ = 1.61·D^(1/3)·ν^(1/6)·ω^(-1/2) (RDE)

    Parameters
    ----------
    D : float
        Diffusion coefficient [m²/s]
    t : float, optional
        Time [s] (for 'time' mode)
    omega : float, optional
        Angular velocity [rad/s] (for 'rotating' mode)
    nu : float, optional
        Kinematic viscosity [m²/s] (for 'rotating' mode)
    mode : str
        'time', 'steady', or 'rotating'

    Returns
    -------
    dict
        delta: Diffusion layer thickness [m]
        delta_um: Thickness in micrometers
    """
    if mode == 'time' and t is not None:
        delta = np.sqrt(np.pi * D * t)
        equation = 'δ = √(πDt)'
    elif mode == 'rotating' and omega is not None and nu is not None:
        delta = 1.61 * D**(1/3) * nu**(1/6) * omega**(-1/2)
        equation = 'δ = 1.61·D^(1/3)·ν^(1/6)·ω^(-1/2)'
    else:
        delta = None
        equation = 'Insufficient parameters'

    result = {
        'D': D,
        'mode': mode,
        'equation': equation,
    }

    if delta is not None:
        result['delta'] = float(delta)
        result['delta_um'] = float(delta * 1e6)

    return result


def sherwood_electrode(Sh: float, L: float, D: float) -> Dict[str, Any]:
    """
    Mass transfer coefficient from Sherwood number.

    k_m = Sh · D / L

    Parameters
    ----------
    Sh : float
        Sherwood number
    L : float
        Characteristic length [m]
    D : float
        Diffusion coefficient [m²/s]

    Returns
    -------
    dict
        k_m: Mass transfer coefficient [m/s]
    """
    k_m = Sh * D / L

    return {
        'k_m': float(k_m),
        'Sh': Sh,
        'D': D,
        'L': L,
        'equation': 'k_m = Sh·D/L',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for electrochemical mass transfer calculations.
    """
    if all(k in kwargs for k in ['n', 'D', 'C_b', 'omega', 'nu']):
        return rotating_disk(kwargs['n'], kwargs['D'], kwargs['C_b'],
                            kwargs['omega'], kwargs['nu'])

    if all(k in kwargs for k in ['n', 'D', 'C_b', 'delta']):
        return limiting_current_density(kwargs['n'], kwargs['D'],
                                        kwargs['C_b'], kwargs['delta'])

    if 'i' in kwargs and 'i_L' in kwargs and 'n' in kwargs:
        return concentration_overpotential(kwargs['i'], kwargs['i_L'],
                                          kwargs['n'], kwargs.get('T', 298.15))

    return {'error': 'Insufficient parameters'}

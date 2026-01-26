"""
Distillation Engines

McCabe-Thiele, Fenske, Underwood, Gilliland methods.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy.optimize import brentq, fsolve


def mccabe_thiele(alpha: float, x_F: float, x_D: float, x_B: float,
                  R: float, q: float = 1.0, n_points: int = 100) -> Dict[str, Any]:
    """
    McCabe-Thiele graphical method for binary distillation.

    Operating lines:
    - Rectifying: y = (R/(R+1))x + x_D/(R+1)
    - Stripping: y = (L'/V')x - (B/V')x_B

    Parameters
    ----------
    alpha : float
        Relative volatility (constant)
    x_F : float
        Feed composition (mole fraction of light key)
    x_D : float
        Distillate composition
    x_B : float
        Bottoms composition
    R : float
        Reflux ratio (L/D)
    q : float
        Feed quality (q=1 saturated liquid, q=0 saturated vapor)
    n_points : int
        Number of points for equilibrium curve

    Returns
    -------
    dict
        N_stages: Number of theoretical stages
        N_rectifying: Stages above feed
        N_stripping: Stages below feed
        x_feed: Feed stage liquid composition
        operating_lines: Line equations
    """
    # Equilibrium curve: y = αx / (1 + (α-1)x)
    x_eq = np.linspace(0, 1, n_points)
    y_eq = alpha * x_eq / (1 + (alpha - 1) * x_eq)

    # Operating line equations
    # Rectifying section: y = (R/(R+1))x + x_D/(R+1)
    slope_rect = R / (R + 1)
    intercept_rect = x_D / (R + 1)

    # q-line: y = (q/(q-1))x - x_F/(q-1)
    if abs(q - 1) > 1e-10:
        slope_q = q / (q - 1)
        intercept_q = -x_F / (q - 1)
    else:
        # Saturated liquid: vertical line at x = x_F
        slope_q = float('inf')
        intercept_q = x_F

    # Find intersection of rectifying line and q-line
    if slope_q != float('inf'):
        x_intersect = (intercept_rect - intercept_q) / (slope_q - slope_rect)
        y_intersect = slope_rect * x_intersect + intercept_rect
    else:
        x_intersect = x_F
        y_intersect = slope_rect * x_F + intercept_rect

    # Stripping section operating line passes through (x_B, x_B) and intersection
    slope_strip = (y_intersect - x_B) / (x_intersect - x_B) if x_intersect != x_B else 1

    # Step off stages from x_D down to x_B
    stages = []
    x_curr = x_D
    y_curr = x_D

    while x_curr > x_B:
        # Move horizontally to equilibrium curve
        # Solve: y_curr = αx / (1 + (α-1)x) for x
        x_new = y_curr / (alpha - (alpha - 1) * y_curr)

        # Move vertically to operating line
        if x_new > x_intersect:
            # Rectifying section
            y_new = slope_rect * x_new + intercept_rect
        else:
            # Stripping section
            y_new = slope_strip * (x_new - x_B) + x_B

        stages.append({'x': float(x_new), 'y': float(y_curr)})

        x_curr = x_new
        y_curr = y_new

        if len(stages) > 200:  # Safety limit
            break

    # Count stages
    N_total = len(stages)

    # Find feed stage
    N_rectifying = sum(1 for s in stages if s['x'] > x_intersect)
    N_stripping = N_total - N_rectifying

    return {
        'N_stages': N_total,
        'N_rectifying': N_rectifying,
        'N_stripping': N_stripping,
        'feed_stage': N_rectifying,
        'x_D': x_D,
        'x_B': x_B,
        'x_F': x_F,
        'R': R,
        'q': q,
        'alpha': alpha,
        'slope_rectifying': float(slope_rect),
        'slope_stripping': float(slope_strip),
        'x_equilibrium': x_eq.tolist(),
        'y_equilibrium': y_eq.tolist(),
        'stages': stages,
        'method': 'mccabe_thiele',
    }


def fenske(alpha: float, x_D: float, x_B: float) -> Dict[str, Any]:
    """
    Fenske equation for minimum stages at total reflux.

    N_min = ln[(x_D/(1-x_D)) × ((1-x_B)/x_B)] / ln(α)

    Parameters
    ----------
    alpha : float
        Relative volatility
    x_D : float
        Distillate composition
    x_B : float
        Bottoms composition

    Returns
    -------
    dict
        N_min: Minimum number of stages
        separation_achieved: (x_D/(1-x_D)) / (x_B/(1-x_B))
    """
    # Separation factor
    separation = (x_D / (1 - x_D)) * ((1 - x_B) / x_B)

    N_min = np.log(separation) / np.log(alpha)

    return {
        'N_min': float(N_min),
        'alpha': alpha,
        'x_D': x_D,
        'x_B': x_B,
        'separation_factor': float(separation),
        'equation': 'N_min = ln[(x_D/(1-x_D))×((1-x_B)/x_B)]/ln(α)',
    }


def underwood(alpha: float, x_F: float, x_D: float, q: float) -> Dict[str, Any]:
    """
    Underwood equations for minimum reflux ratio.

    For binary mixture:
    R_min = (1/(α-1)) × [(x_D/x_F) - α(1-x_D)/(1-x_F)]

    Parameters
    ----------
    alpha : float
        Relative volatility
    x_F : float
        Feed composition
    x_D : float
        Distillate composition
    q : float
        Feed quality

    Returns
    -------
    dict
        R_min: Minimum reflux ratio
        theta: Underwood root
    """
    # Find Underwood root θ (between 1 and α)
    def underwood_eq(theta):
        return alpha * x_F / (alpha - theta) + (1 - x_F) / (1 - theta) - 1 + q

    try:
        theta = brentq(underwood_eq, 1.001, alpha - 0.001)
    except:
        theta = (1 + alpha) / 2  # Approximate

    # Minimum reflux
    V_min = alpha * x_D / (alpha - theta) + (1 - x_D) / (1 - theta)
    R_min = V_min - 1

    return {
        'R_min': float(R_min),
        'theta': float(theta),
        'V_min_over_D': float(V_min),
        'alpha': alpha,
        'x_F': x_F,
        'x_D': x_D,
        'q': q,
        'equation': 'Σ(α_i·x_F,i)/(α_i-θ) = 1-q',
    }


def gilliland(N_min: float, R_min: float, R: float) -> Dict[str, Any]:
    """
    Gilliland correlation for actual stages.

    Y = 1 - exp[(1+54.4X)(X-1) / (11+117.2X)X^0.5]

    where X = (R - R_min)/(R + 1)
          Y = (N - N_min)/(N + 1)

    Parameters
    ----------
    N_min : float
        Minimum stages (Fenske)
    R_min : float
        Minimum reflux ratio (Underwood)
    R : float
        Actual reflux ratio

    Returns
    -------
    dict
        N: Actual number of stages
        Y: Gilliland Y parameter
        X: Gilliland X parameter
    """
    X = (R - R_min) / (R + 1)

    # Gilliland correlation (Eduljee form)
    Y = 0.75 * (1 - X**0.5668)

    # Alternative: Molokanov correlation
    # Y = 1 - np.exp((1 + 54.4*X) * (X - 1) / (11 + 117.2*X) / (X**0.5 + 1e-10))

    N = (Y + N_min) / (1 - Y)

    return {
        'N': float(N),
        'N_rounded': int(np.ceil(N)),
        'N_min': N_min,
        'R_min': R_min,
        'R': R,
        'X': float(X),
        'Y': float(Y),
        'R_over_R_min': float(R / R_min) if R_min > 0 else float('inf'),
        'equation': 'Y = f(X) Gilliland correlation',
    }


def kirkbride(x_F: float, x_D: float, x_B: float, D: float, B: float) -> Dict[str, Any]:
    """
    Kirkbride equation for feed stage location.

    log(N_R/N_S) = 0.206 × log[(B/D) × (x_F/(1-x_F)) × ((1-x_D)/x_B)²]

    Parameters
    ----------
    x_F : float
        Feed composition
    x_D : float
        Distillate composition
    x_B : float
        Bottoms composition
    D : float
        Distillate flow rate
    B : float
        Bottoms flow rate

    Returns
    -------
    dict
        N_R_over_N_S: Ratio of rectifying to stripping stages
        feed_fraction: Feed stage location as fraction from top
    """
    term = (B / D) * (x_F / (1 - x_F)) * ((1 - x_D) / x_B)**2
    log_ratio = 0.206 * np.log10(term)
    ratio = 10**log_ratio

    # If N_total stages, N_R/N_S = ratio
    # N_R + N_S = N_total
    # N_R = ratio × N_S
    # (1 + ratio) × N_S = N_total
    # Feed stage from top = N_R = ratio/(1+ratio) × N_total

    feed_fraction = ratio / (1 + ratio)

    return {
        'N_R_over_N_S': float(ratio),
        'feed_fraction_from_top': float(feed_fraction),
        'x_F': x_F,
        'x_D': x_D,
        'x_B': x_B,
        'D': D,
        'B': B,
        'equation': 'log(N_R/N_S) = 0.206×log[...]',
    }


def stage_efficiency(E_M: float, N_theoretical: float) -> Dict[str, Any]:
    """
    Murphree efficiency correction.

    N_actual = N_theoretical / E_M

    Parameters
    ----------
    E_M : float
        Murphree efficiency (0-1, typically 0.5-0.8)
    N_theoretical : float
        Number of theoretical stages

    Returns
    -------
    dict
        N_actual: Actual number of trays needed
        N_actual_rounded: Rounded up to integer
    """
    N_actual = N_theoretical / E_M

    return {
        'N_actual': float(N_actual),
        'N_actual_rounded': int(np.ceil(N_actual)),
        'N_theoretical': N_theoretical,
        'E_M': E_M,
        'equation': 'N_actual = N_theoretical/E_M',
    }


def flooding_velocity(sigma: float, rho_L: float, rho_V: float,
                      tray_spacing: float = 0.6) -> Dict[str, Any]:
    """
    Fair flooding correlation for column hydraulics.

    u_flood = C × [(σ/20)^0.2 × (ρ_L - ρ_V)/ρ_V]^0.5

    Parameters
    ----------
    sigma : float
        Surface tension [mN/m]
    rho_L : float
        Liquid density [kg/m³]
    rho_V : float
        Vapor density [kg/m³]
    tray_spacing : float
        Tray spacing [m] (default 0.6)

    Returns
    -------
    dict
        u_flood: Flooding velocity [m/s]
        u_operating: Recommended operating velocity (70-80% of flood)
    """
    # Capacity parameter C (depends on tray spacing, typically 0.03-0.12)
    # Simplified: C ~ 0.1 for 0.6m tray spacing
    C = 0.1 * (tray_spacing / 0.6)**0.5

    # Flooding velocity
    u_flood = C * ((sigma / 20)**0.2 * (rho_L - rho_V) / rho_V)**0.5

    return {
        'u_flood': float(u_flood),
        'u_operating_70': float(0.70 * u_flood),
        'u_operating_80': float(0.80 * u_flood),
        'sigma': sigma,
        'rho_L': rho_L,
        'rho_V': rho_V,
        'tray_spacing': tray_spacing,
        'C_parameter': float(C),
        'equation': 'u_flood = C×[(σ/20)^0.2×(ρ_L-ρ_V)/ρ_V]^0.5',
    }


def column_diameter(V_dot: float, u_operating: float, fraction_active: float = 0.88) -> Dict[str, Any]:
    """
    Column diameter from vapor flow rate.

    A = V_dot / u
    D = sqrt(4A / (π × f_active))

    Parameters
    ----------
    V_dot : float
        Volumetric vapor flow rate [m³/s]
    u_operating : float
        Operating velocity [m/s]
    fraction_active : float
        Active area fraction (default 0.88 for sieve tray)

    Returns
    -------
    dict
        diameter: Column diameter [m]
        area_total: Total column area [m²]
        area_active: Active (bubbling) area [m²]
    """
    area_total = V_dot / u_operating
    diameter = np.sqrt(4 * area_total / (np.pi * fraction_active))
    area_active = area_total * fraction_active

    return {
        'diameter': float(diameter),
        'diameter_ft': float(diameter * 3.281),
        'area_total': float(area_total),
        'area_active': float(area_active),
        'V_dot': V_dot,
        'u_operating': u_operating,
        'equation': 'D = √(4V/(π·u·f))',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for distillation calculations.
    """
    if all(k in kwargs for k in ['alpha', 'x_F', 'x_D', 'x_B', 'R']):
        return mccabe_thiele(kwargs['alpha'], kwargs['x_F'], kwargs['x_D'],
                             kwargs['x_B'], kwargs['R'], kwargs.get('q', 1.0))

    if all(k in kwargs for k in ['alpha', 'x_D', 'x_B']) and 'x_F' not in kwargs:
        return fenske(kwargs['alpha'], kwargs['x_D'], kwargs['x_B'])

    if all(k in kwargs for k in ['N_min', 'R_min', 'R']):
        return gilliland(kwargs['N_min'], kwargs['R_min'], kwargs['R'])

    return {'error': 'Insufficient parameters'}

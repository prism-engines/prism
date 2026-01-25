"""
Liquid-Liquid Equilibrium (LLE) Engines

Tie lines, binodal curves, lever rule, ternary diagrams.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import fsolve


def tie_line(x_alpha: Tuple[float, float, float],
             x_beta: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Tie line connecting two equilibrium liquid phases.

    A tie line connects compositions of two phases in equilibrium.

    Parameters
    ----------
    x_alpha : tuple
        Composition of phase α (x1, x2, x3) - must sum to 1
    x_beta : tuple
        Composition of phase β (x1, x2, x3) - must sum to 1

    Returns
    -------
    dict
        slope: Slope of tie line in x1-x2 coordinates
        length: Length of tie line
        midpoint: Midpoint composition
        K_values: Distribution coefficients x_beta/x_alpha for each component
    """
    x_alpha = np.array(x_alpha)
    x_beta = np.array(x_beta)

    # Normalize if needed
    x_alpha = x_alpha / np.sum(x_alpha)
    x_beta = x_beta / np.sum(x_beta)

    # Tie line properties in x1-x2 coordinates
    if abs(x_alpha[0] - x_beta[0]) > 1e-10:
        slope = (x_beta[1] - x_alpha[1]) / (x_beta[0] - x_alpha[0])
    else:
        slope = float('inf')

    length = np.sqrt(np.sum((x_beta - x_alpha)**2))
    midpoint = (x_alpha + x_beta) / 2

    # Distribution coefficients
    K_values = x_beta / x_alpha
    K_values = np.where(x_alpha > 1e-10, K_values, np.inf)

    return {
        'x_alpha': x_alpha.tolist(),
        'x_beta': x_beta.tolist(),
        'slope': float(slope) if slope != float('inf') else None,
        'length': float(length),
        'midpoint': midpoint.tolist(),
        'K_values': K_values.tolist(),
    }


def lever_rule(x_overall: Tuple[float, float, float],
               x_alpha: Tuple[float, float, float],
               x_beta: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Lever rule for phase amounts.

    n_alpha/n_beta = |x_overall - x_beta| / |x_alpha - x_overall|

    Parameters
    ----------
    x_overall : tuple
        Overall mixture composition
    x_alpha : tuple
        Composition of phase α
    x_beta : tuple
        Composition of phase β

    Returns
    -------
    dict
        fraction_alpha: Mole fraction of phase α
        fraction_beta: Mole fraction of phase β
    """
    x_overall = np.array(x_overall)
    x_alpha = np.array(x_alpha)
    x_beta = np.array(x_beta)

    # Normalize
    x_overall = x_overall / np.sum(x_overall)
    x_alpha = x_alpha / np.sum(x_alpha)
    x_beta = x_beta / np.sum(x_beta)

    # Distances along tie line
    dist_to_beta = np.linalg.norm(x_overall - x_beta)
    dist_to_alpha = np.linalg.norm(x_alpha - x_overall)
    total_dist = np.linalg.norm(x_alpha - x_beta)

    if total_dist > 1e-10:
        fraction_alpha = dist_to_beta / total_dist
        fraction_beta = dist_to_alpha / total_dist
    else:
        fraction_alpha = 0.5
        fraction_beta = 0.5

    return {
        'fraction_alpha': float(fraction_alpha),
        'fraction_beta': float(fraction_beta),
        'x_overall': x_overall.tolist(),
        'x_alpha': x_alpha.tolist(),
        'x_beta': x_beta.tolist(),
        'ratio_alpha_to_beta': float(fraction_alpha / fraction_beta) if fraction_beta > 1e-10 else float('inf'),
        'equation': 'n_α/n_β = |x-x_β|/|x_α-x|',
    }


def binodal_curve_margules(A12: float, A21: float, n_points: int = 50) -> Dict[str, Any]:
    """
    Generate binodal curve using Margules equation.

    At the binodal: γ1·x1_α = γ1·x1_β (equal activities)

    Parameters
    ----------
    A12 : float
        Margules parameter A12
    A21 : float
        Margules parameter A21
    n_points : int
        Number of points on curve

    Returns
    -------
    dict
        x1_alpha: Component 1 mole fractions in phase α
        x1_beta: Component 1 mole fractions in phase β
        critical_point: Plait point composition
    """
    def margules_gamma(x1, A12, A21):
        x2 = 1 - x1
        ln_gamma1 = x2**2 * (A12 + 2 * (A21 - A12) * x1)
        ln_gamma2 = x1**2 * (A21 + 2 * (A12 - A21) * x2)
        return np.exp(ln_gamma1), np.exp(ln_gamma2)

    def equal_activity(x1_beta, x1_alpha):
        """Find x1_beta where activities are equal to those at x1_alpha."""
        gamma1_alpha, gamma2_alpha = margules_gamma(x1_alpha, A12, A21)
        gamma1_beta, gamma2_beta = margules_gamma(x1_beta, A12, A21)

        # Equal activities: gamma1*x1 equal, gamma2*x2 equal
        eq1 = gamma1_alpha * x1_alpha - gamma1_beta * x1_beta
        eq2 = gamma2_alpha * (1 - x1_alpha) - gamma2_beta * (1 - x1_beta)

        return eq1 + eq2  # Combined error

    # Find critical point (where phases merge)
    # At critical: d²G/dx² = 0 and d³G/dx³ = 0
    # Simplified: search for where tie line length → 0
    x1_critical = 0.5  # Initial guess

    # Generate binodal
    x1_alpha_vals = []
    x1_beta_vals = []

    for x1_alpha in np.linspace(0.05, 0.45, n_points // 2):
        try:
            x1_beta = fsolve(equal_activity, 1 - x1_alpha, args=(x1_alpha,))[0]
            if 0 < x1_beta < 1 and abs(x1_beta - x1_alpha) > 0.01:
                x1_alpha_vals.append(float(x1_alpha))
                x1_beta_vals.append(float(x1_beta))
        except:
            pass

    return {
        'x1_alpha': x1_alpha_vals,
        'x1_beta': x1_beta_vals,
        'A12': A12,
        'A21': A21,
        'n_points': len(x1_alpha_vals),
        'note': 'Simplified calculation; full solution requires iterative methods',
    }


def plait_point(T: float, T_c: float, x_c: float, beta: float = 0.325) -> Dict[str, Any]:
    """
    Estimate plait point (critical solution point) properties.

    Near critical: (x_alpha - x_beta) ∝ |T - T_c|^β

    Parameters
    ----------
    T : float
        Temperature [K]
    T_c : float
        Critical solution temperature [K]
    x_c : float
        Critical composition
    beta : float
        Critical exponent (default 0.325, Ising universality)

    Returns
    -------
    dict
        x_alpha: Composition of dilute phase
        x_beta: Composition of concentrated phase
        two_phase: Whether system is in two-phase region
    """
    if T >= T_c:
        # Single phase (above UCST or below LCST depending on system)
        return {
            'x_alpha': x_c,
            'x_beta': x_c,
            'two_phase': False,
            'T': T,
            'T_c': T_c,
            'note': 'Above critical temperature - single phase',
        }

    # Two-phase region
    delta_x = abs(T - T_c)**beta
    x_alpha = x_c - delta_x / 2
    x_beta = x_c + delta_x / 2

    return {
        'x_alpha': float(max(0, x_alpha)),
        'x_beta': float(min(1, x_beta)),
        'two_phase': True,
        'T': T,
        'T_c': T_c,
        'x_c': x_c,
        'delta_x': float(delta_x),
        'equation': '(x_β - x_α) ∝ |T - T_c|^β',
    }


def ternary_coordinates(x1: float, x2: float, x3: float = None) -> Dict[str, Any]:
    """
    Convert mole fractions to ternary diagram coordinates.

    Equilateral triangle with vertices at components.

    Parameters
    ----------
    x1, x2, x3 : float
        Mole fractions (x3 calculated if not provided)

    Returns
    -------
    dict
        X, Y: Cartesian coordinates for ternary plot
        x1, x2, x3: Normalized mole fractions
    """
    if x3 is None:
        x3 = 1 - x1 - x2

    # Normalize
    total = x1 + x2 + x3
    x1, x2, x3 = x1/total, x2/total, x3/total

    # Ternary to Cartesian (equilateral triangle)
    # x1 at bottom left (0, 0)
    # x2 at bottom right (1, 0)
    # x3 at top (0.5, sqrt(3)/2)
    X = x2 + x3 * 0.5
    Y = x3 * np.sqrt(3) / 2

    return {
        'X': float(X),
        'Y': float(Y),
        'x1': float(x1),
        'x2': float(x2),
        'x3': float(x3),
    }


def ternary_diagram(tie_lines: List[Tuple[Tuple, Tuple]],
                    binodal: List[Tuple] = None) -> Dict[str, Any]:
    """
    Generate ternary diagram data.

    Parameters
    ----------
    tie_lines : list of tuples
        Each element is ((x1_α, x2_α, x3_α), (x1_β, x2_β, x3_β))
    binodal : list of tuples, optional
        Points on binodal curve (x1, x2, x3)

    Returns
    -------
    dict
        tie_lines_xy: Tie lines in Cartesian coordinates
        binodal_xy: Binodal curve in Cartesian coordinates
    """
    tie_lines_xy = []
    for (alpha, beta) in tie_lines:
        alpha_xy = ternary_coordinates(*alpha)
        beta_xy = ternary_coordinates(*beta)
        tie_lines_xy.append({
            'alpha': (alpha_xy['X'], alpha_xy['Y']),
            'beta': (beta_xy['X'], beta_xy['Y']),
        })

    binodal_xy = []
    if binodal is not None:
        for point in binodal:
            xy = ternary_coordinates(*point)
            binodal_xy.append((xy['X'], xy['Y']))

    return {
        'tie_lines_xy': tie_lines_xy,
        'binodal_xy': binodal_xy,
        'n_tie_lines': len(tie_lines),
        'n_binodal_points': len(binodal_xy),
    }


def distribution_coefficient(C_extract: float, C_raffinate: float) -> Dict[str, Any]:
    """
    Distribution coefficient for LLE.

    K_D = C_extract / C_raffinate

    Parameters
    ----------
    C_extract : float
        Concentration in extract phase
    C_raffinate : float
        Concentration in raffinate phase

    Returns
    -------
    dict
        K_D: Distribution coefficient
        selectivity_indicator: ln(K_D)
    """
    K_D = C_extract / C_raffinate if C_raffinate > 0 else float('inf')

    return {
        'K_D': float(K_D),
        'log_K_D': float(np.log10(K_D)) if K_D > 0 and K_D != float('inf') else None,
        'favors': 'extract' if K_D > 1 else 'raffinate',
        'C_extract': C_extract,
        'C_raffinate': C_raffinate,
        'equation': 'K_D = C_extract/C_raffinate',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for LLE calculations.
    """
    if 'x_overall' in kwargs and 'x_alpha' in kwargs and 'x_beta' in kwargs:
        return lever_rule(kwargs['x_overall'], kwargs['x_alpha'], kwargs['x_beta'])

    if 'x_alpha' in kwargs and 'x_beta' in kwargs:
        return tie_line(kwargs['x_alpha'], kwargs['x_beta'])

    if 'C_extract' in kwargs and 'C_raffinate' in kwargs:
        return distribution_coefficient(kwargs['C_extract'], kwargs['C_raffinate'])

    return {'error': 'Insufficient parameters'}

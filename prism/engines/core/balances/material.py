"""
Material Balance Engines

Mass balance, component balance, extent of reaction, recycle calculations.
"""

import numpy as np
from typing import Dict, Any, Optional, List


def total_mass_balance(m_in: float, m_out: float,
                       accumulation: float = 0.0) -> Dict[str, Any]:
    """
    Overall mass balance.

    ṁ_in = ṁ_out + d(m)/dt

    Steady state: ṁ_in = ṁ_out

    Parameters
    ----------
    m_in : float
        Mass flow rate in [kg/s]
    m_out : float
        Mass flow rate out [kg/s]
    accumulation : float
        Rate of accumulation [kg/s] (0 for steady state)

    Returns
    -------
    dict
        balance_error: m_in - m_out - accumulation
        steady_state: Whether accumulation is zero
        closure: Percent closure
    """
    balance_error = m_in - m_out - accumulation
    closure = (m_out + accumulation) / m_in * 100 if m_in > 0 else 100

    return {
        'm_in': m_in,
        'm_out': m_out,
        'accumulation': accumulation,
        'balance_error': float(balance_error),
        'closure_percent': float(closure),
        'steady_state': abs(accumulation) < 1e-10,
        'balanced': abs(balance_error) < 1e-6 * max(m_in, m_out),
        'equation': 'ṁ_in = ṁ_out + d(m)/dt',
    }


def component_balance(F: float, x_in: float, x_out: float,
                      r: float = 0.0, V: float = 1.0) -> Dict[str, Any]:
    """
    Component material balance.

    F_in · x_in = F_out · x_out + r · V + accumulation

    Parameters
    ----------
    F : float
        Flow rate [mol/s or kg/s]
    x_in : float
        Inlet composition (mass or mole fraction)
    x_out : float
        Outlet composition (mass or mole fraction)
    r : float
        Reaction rate [mol/(m³·s) or kg/(m³·s)]
    V : float
        Volume [m³]

    Returns
    -------
    dict
        component_in: F · x_in
        component_out: F · x_out
        generation: r · V
    """
    comp_in = F * x_in
    comp_out = F * x_out
    generation = r * V

    balance_error = comp_in - comp_out - generation

    return {
        'component_in': float(comp_in),
        'component_out': float(comp_out),
        'generation': float(generation),
        'balance_error': float(balance_error),
        'conversion': float((x_in - x_out) / x_in * 100) if x_in > 0 else 0,
        'equation': 'F·x_in = F·x_out + r·V',
    }


def extent_of_reaction(n_initial: float, n_final: float, nu: int) -> Dict[str, Any]:
    """
    Extent of reaction (degree of advancement).

    ξ = (n_i - n_i,0) / ν_i

    Parameters
    ----------
    n_initial : float
        Initial moles of species i
    n_final : float
        Final moles of species i
    nu : int
        Stoichiometric coefficient of species i
        (negative for reactants, positive for products)

    Returns
    -------
    dict
        xi: Extent of reaction [mol]
        moles_reacted: Moles of species consumed/produced
    """
    xi = (n_final - n_initial) / nu

    return {
        'xi': float(xi),
        'n_initial': n_initial,
        'n_final': n_final,
        'nu': nu,
        'moles_changed': float(n_final - n_initial),
        'equation': 'ξ = (n - n₀)/ν',
    }


def limiting_reactant(n: List[float], nu: List[int]) -> Dict[str, Any]:
    """
    Identify limiting reactant.

    The limiting reactant has the smallest n_i / |ν_i| ratio.

    Parameters
    ----------
    n : list of float
        Initial moles of each reactant
    nu : list of int
        Stoichiometric coefficients (negative for reactants)

    Returns
    -------
    dict
        limiting_index: Index of limiting reactant
        max_extent: Maximum extent of reaction [mol]
        remaining: Moles remaining of each reactant at completion
    """
    n = np.asarray(n)
    nu = np.asarray(nu)

    # Only consider reactants (negative nu)
    reactant_mask = nu < 0

    ratios = np.where(reactant_mask, n / np.abs(nu), np.inf)
    limiting_idx = np.argmin(ratios)
    max_extent = ratios[limiting_idx]

    # Calculate remaining moles
    remaining = n + nu * max_extent

    return {
        'limiting_index': int(limiting_idx),
        'max_extent': float(max_extent),
        'remaining': remaining.tolist(),
        'limiting_ratio': float(ratios[limiting_idx]),
        'all_ratios': ratios.tolist(),
        'equation': 'limiting: min(n_i/|ν_i|)',
    }


def excess_reactant(n_A: float, n_B: float, nu_A: int, nu_B: int) -> Dict[str, Any]:
    """
    Calculate percent excess of reactant.

    % Excess = (n_excess - n_stoichiometric) / n_stoichiometric × 100

    Parameters
    ----------
    n_A : float
        Moles of reactant A
    n_B : float
        Moles of reactant B
    nu_A : int
        Stoichiometric coefficient of A (negative)
    nu_B : int
        Stoichiometric coefficient of B (negative)

    Returns
    -------
    dict
        excess_species: Which species is in excess
        percent_excess: Percent excess
        stoichiometric_ratio: Required ratio of A:B
    """
    # Required ratio
    stoich_ratio = abs(nu_A) / abs(nu_B)
    actual_ratio = n_A / n_B

    if actual_ratio > stoich_ratio:
        # A is in excess
        n_stoich_A = n_B * stoich_ratio
        percent_excess = (n_A - n_stoich_A) / n_stoich_A * 100
        excess_species = 'A'
    else:
        # B is in excess
        n_stoich_B = n_A / stoich_ratio
        percent_excess = (n_B - n_stoich_B) / n_stoich_B * 100
        excess_species = 'B'

    return {
        'excess_species': excess_species,
        'percent_excess': float(percent_excess),
        'stoichiometric_ratio': float(stoich_ratio),
        'actual_ratio': float(actual_ratio),
        'n_A': n_A,
        'n_B': n_B,
        'equation': '% excess = (n_excess - n_stoich)/n_stoich × 100',
    }


def recycle_ratio(R: float, F_fresh: float) -> Dict[str, Any]:
    """
    Recycle ratio calculation.

    Recycle ratio = R / F_fresh

    Parameters
    ----------
    R : float
        Recycle stream flow rate [mol/s or kg/s]
    F_fresh : float
        Fresh feed flow rate [mol/s or kg/s]

    Returns
    -------
    dict
        recycle_ratio: R / F_fresh
        total_feed: F_fresh + R
        recycle_fraction: R / (F_fresh + R)
    """
    total_feed = F_fresh + R
    recycle_ratio_val = R / F_fresh if F_fresh > 0 else float('inf')
    recycle_fraction = R / total_feed if total_feed > 0 else 0

    return {
        'recycle_ratio': float(recycle_ratio_val),
        'R': R,
        'F_fresh': F_fresh,
        'total_feed': float(total_feed),
        'recycle_fraction': float(recycle_fraction),
        'equation': 'recycle_ratio = R/F_fresh',
    }


def purge_calculation(F_in: float, x_inert_in: float, x_inert_max: float,
                      conversion: float) -> Dict[str, Any]:
    """
    Purge stream calculation for inert buildup.

    At steady state, inert in = inert out (via purge)
    F_in · x_inert_in = P · x_inert_purge

    Parameters
    ----------
    F_in : float
        Fresh feed flow rate [mol/s]
    x_inert_in : float
        Inert fraction in fresh feed
    x_inert_max : float
        Maximum allowable inert fraction in recycle
    conversion : float
        Single-pass conversion (0-1)

    Returns
    -------
    dict
        purge_rate: Required purge flow rate
        recycle_rate: Recycle flow rate
        inert_accumulation: Steady state inert level
    """
    # At steady state: F_in * x_inert = P * x_inert_max
    # And mass balance: F_in * (1 - x_inert) * conversion = P * (1 - x_inert_max) * (1 - conversion)?
    # Simplified: purge removes inerts at same rate they enter

    inert_in = F_in * x_inert_in
    purge_rate = inert_in / x_inert_max

    # Unconverted reactant in recycle
    reactant_in = F_in * (1 - x_inert_in)
    reactant_out_reactor = reactant_in * (1 - conversion)

    # Recycle = unreacted - purge
    recycle_rate = max(0, reactant_out_reactor - purge_rate * (1 - x_inert_max))

    return {
        'purge_rate': float(purge_rate),
        'recycle_rate': float(recycle_rate),
        'inert_in_feed': float(inert_in),
        'x_inert_max': x_inert_max,
        'overall_conversion': float(1 - purge_rate * (1 - x_inert_max) / reactant_in) if reactant_in > 0 else conversion,
        'equation': 'F_in·x_inert = P·x_max',
    }


def bypass_calculation(F_total: float, F_bypass: float,
                       x_in: float, x_process: float) -> Dict[str, Any]:
    """
    Bypass stream calculation.

    x_out = (F_bypass · x_in + (F_total - F_bypass) · x_process) / F_total

    Parameters
    ----------
    F_total : float
        Total flow rate [mol/s]
    F_bypass : float
        Bypass flow rate [mol/s]
    x_in : float
        Inlet composition
    x_process : float
        Composition after processing

    Returns
    -------
    dict
        x_out: Final outlet composition
        bypass_fraction: F_bypass / F_total
        F_process: Flow through process unit
    """
    F_process = F_total - F_bypass
    x_out = (F_bypass * x_in + F_process * x_process) / F_total
    bypass_fraction = F_bypass / F_total

    return {
        'x_out': float(x_out),
        'x_in': x_in,
        'x_process': x_process,
        'F_total': F_total,
        'F_bypass': F_bypass,
        'F_process': float(F_process),
        'bypass_fraction': float(bypass_fraction),
        'equation': 'x_out = (F_bp·x_in + F_pr·x_pr)/F_tot',
    }


def mixing_balance(flows: List[float], compositions: List[float]) -> Dict[str, Any]:
    """
    Mixing of multiple streams.

    F_out = Σ F_i
    x_out = Σ(F_i · x_i) / F_out

    Parameters
    ----------
    flows : list of float
        Flow rates of each stream [mol/s]
    compositions : list of float
        Compositions of each stream

    Returns
    -------
    dict
        F_out: Total outlet flow
        x_out: Outlet composition
    """
    flows = np.asarray(flows)
    compositions = np.asarray(compositions)

    F_out = np.sum(flows)
    x_out = np.sum(flows * compositions) / F_out if F_out > 0 else 0

    return {
        'F_out': float(F_out),
        'x_out': float(x_out),
        'n_streams': len(flows),
        'flows': flows.tolist(),
        'compositions': compositions.tolist(),
        'equation': 'x_out = Σ(F_i·x_i)/F_out',
    }


def splitting_balance(F_in: float, x_in: float,
                      split_fractions: List[float]) -> Dict[str, Any]:
    """
    Stream splitting calculation.

    F_i = F_in · φ_i (where Σφ = 1)
    x_i = x_in (composition unchanged in splitter)

    Parameters
    ----------
    F_in : float
        Inlet flow rate [mol/s]
    x_in : float
        Inlet composition
    split_fractions : list of float
        Fraction going to each outlet stream

    Returns
    -------
    dict
        outlet_flows: Flow rate of each outlet stream
        outlet_compositions: Composition of each outlet (all equal to x_in)
    """
    split_fractions = np.asarray(split_fractions)

    # Normalize if needed
    total_split = np.sum(split_fractions)
    if abs(total_split - 1.0) > 1e-10:
        split_fractions = split_fractions / total_split

    outlet_flows = F_in * split_fractions
    outlet_compositions = [x_in] * len(split_fractions)

    return {
        'outlet_flows': outlet_flows.tolist(),
        'outlet_compositions': outlet_compositions,
        'split_fractions': split_fractions.tolist(),
        'F_in': F_in,
        'x_in': x_in,
        'equation': 'F_i = F_in·φ_i, x_i = x_in',
    }


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for material balance calculations.
    """
    if 'm_in' in kwargs and 'm_out' in kwargs:
        return total_mass_balance(kwargs['m_in'], kwargs['m_out'],
                                  kwargs.get('accumulation', 0))

    if 'flows' in kwargs and 'compositions' in kwargs:
        return mixing_balance(kwargs['flows'], kwargs['compositions'])

    return {'error': 'Insufficient parameters'}

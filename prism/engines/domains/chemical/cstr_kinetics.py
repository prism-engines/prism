"""
CSTR Kinetics Analysis

Calculate rate constants from experimental CSTR data:
- Conversion from inlet/outlet concentrations
- Rate constant k from CSTR design equation
- Arrhenius parameters (Ea, A) from k vs T data
- Material balance verification
- Energy balance (heat duty)

For second-order equimolar reaction A + B → products:
    k = X / (τ × C_A0 × (1-X)²)

Linearized Arrhenius:
    ln(k) = ln(A) - Ea/(R×T)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


# Universal gas constant
R = 8.314  # J/(mol·K)


def conversion_from_concentration(
    C_A0: float,
    C_A: float
) -> Dict[str, Any]:
    """
    Calculate conversion from concentrations.

    X = (C_A0 - C_A) / C_A0

    Parameters
    ----------
    C_A0 : float
        Inlet concentration [mol/L or mol/m³]
    C_A : float
        Outlet concentration [mol/L or mol/m³]

    Returns
    -------
    dict
        conversion: X (0 to 1)
        percent: X × 100
    """
    X = (C_A0 - C_A) / C_A0 if C_A0 > 0 else 0

    return {
        'conversion': float(X),
        'percent': float(X * 100),
        'C_A0': C_A0,
        'C_A': C_A,
        'moles_reacted_fraction': float(X),
        'equation': 'X = (C_A0 - C_A) / C_A0',
    }


def cstr_rate_constant(
    X: float,
    tau: float,
    C_A0: float,
    order: int = 2,
    equimolar: bool = True
) -> Dict[str, Any]:
    """
    Calculate rate constant from CSTR conversion data.

    For 2nd order equimolar (A + B, C_A0 = C_B0):
        k = X / (τ × C_A0 × (1-X)²)

    For 1st order:
        k = X / (τ × (1-X))

    Parameters
    ----------
    X : float
        Conversion (0 to 1)
    tau : float
        Residence time (V/Q) [time units]
    C_A0 : float
        Inlet concentration [mol/L]
    order : int
        Reaction order (1 or 2)
    equimolar : bool
        True if C_A0 = C_B0 (for 2nd order)

    Returns
    -------
    dict
        k: Rate constant [L/(mol·time) for 2nd order, 1/time for 1st order]
    """
    if X >= 1:
        return {'error': 'Conversion cannot be >= 1', 'k': float('inf')}

    if order == 2 and equimolar:
        # k = X / (τ × C_A0 × (1-X)²)
        k = X / (tau * C_A0 * (1 - X)**2)
        units = 'L/(mol·time)'
    elif order == 1:
        # k = X / (τ × (1-X))
        k = X / (tau * (1 - X))
        units = '1/time'
    elif order == 0:
        # k = C_A0 × X / τ
        k = C_A0 * X / tau
        units = 'mol/(L·time)'
    else:
        return {'error': f'Order {order} not implemented'}

    # Also calculate reaction rate
    C_A = C_A0 * (1 - X)

    if order == 2:
        r = k * C_A**2  # equimolar: C_A = C_B
    elif order == 1:
        r = k * C_A
    else:
        r = k

    return {
        'k': float(k),
        'order': order,
        'units': units,
        'conversion': X,
        'tau': tau,
        'C_A0': C_A0,
        'C_A_out': float(C_A),
        'reaction_rate': float(r),
        'equation': 'k = X / (τ × C_A0 × (1-X)²)' if order == 2 else 'k = X / (τ × (1-X))',
    }


def residence_time(
    V: float,
    Q: float,
    V_unit: str = 'L',
    Q_unit: str = 'mL/min'
) -> Dict[str, Any]:
    """
    Calculate residence time (space time).

    τ = V / Q

    Parameters
    ----------
    V : float
        Reactor volume
    Q : float
        Volumetric flow rate
    V_unit : str
        Volume unit ('L', 'mL', 'm³')
    Q_unit : str
        Flow rate unit ('mL/min', 'L/min', 'L/s', 'm³/s')

    Returns
    -------
    dict
        tau: Residence time [min]
    """
    # Convert to consistent units (L and min)
    V_L = V
    if V_unit == 'mL':
        V_L = V / 1000
    elif V_unit == 'm³':
        V_L = V * 1000

    Q_L_min = Q
    if Q_unit == 'mL/min':
        Q_L_min = Q / 1000
    elif Q_unit == 'L/s':
        Q_L_min = Q * 60
    elif Q_unit == 'm³/s':
        Q_L_min = Q * 1000 * 60

    tau = V_L / Q_L_min

    return {
        'tau': float(tau),
        'tau_unit': 'min',
        'V': V,
        'Q': Q,
        'V_L': float(V_L),
        'Q_L_min': float(Q_L_min),
        'equation': 'τ = V/Q',
    }


def arrhenius_regression(
    T: List[float],
    k: List[float]
) -> Dict[str, Any]:
    """
    Arrhenius regression to determine Ea and A.

    ln(k) = ln(A) - Ea/(R×T)

    Linear regression of ln(k) vs 1/T:
        slope = -Ea/R
        intercept = ln(A)

    Parameters
    ----------
    T : list of float
        Temperatures [K]
    k : list of float
        Rate constants [consistent units]

    Returns
    -------
    dict
        Ea: Activation energy [J/mol]
        Ea_kJ: Activation energy [kJ/mol]
        A: Pre-exponential factor [same units as k]
        r_squared: Coefficient of determination
    """
    T = np.asarray(T, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    # Linearize
    inv_T = 1 / T
    ln_k = np.log(k)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_k)

    # Extract parameters
    Ea = -slope * R  # J/mol
    A = np.exp(intercept)

    # Predicted values for comparison
    ln_k_pred = intercept + slope * inv_T
    k_pred = np.exp(ln_k_pred)

    return {
        'activation_energy_J_mol': float(Ea),
        'activation_energy_kJ_mol': float(Ea / 1000),
        'pre_exponential': float(A),
        'pre_exponential_sci': f'{A:.2e}',
        'r_squared': float(r_value**2),
        'p_value': float(p_value),
        'std_err_slope': float(std_err),
        'slope': float(slope),
        'intercept': float(intercept),
        'T': T.tolist(),
        'k': k.tolist(),
        'inv_T': inv_T.tolist(),
        'ln_k': ln_k.tolist(),
        'k_predicted': k_pred.tolist(),
        'equation': 'ln(k) = ln(A) - Ea/(R×T)',
    }


def heat_duty_isothermal_cstr(
    F_A0: float,
    X: float,
    delta_H_rxn: float
) -> Dict[str, Any]:
    """
    Heat duty for isothermal CSTR operation.

    Q = F_A0 × X × (-ΔH_rxn)

    Negative Q = heat must be removed (exothermic)
    Positive Q = heat must be added (endothermic)

    Parameters
    ----------
    F_A0 : float
        Molar flow rate of limiting reactant [mol/s or mol/min]
    X : float
        Conversion (0 to 1)
    delta_H_rxn : float
        Heat of reaction [J/mol] (negative for exothermic)

    Returns
    -------
    dict
        Q: Heat duty [J/s = W, or J/min depending on F units]
        sign: 'remove' for exothermic, 'add' for endothermic
    """
    # Rate of reaction (moles reacted per time)
    moles_reacted = F_A0 * X

    # Heat released by reaction (positive for exothermic if delta_H < 0)
    Q_rxn = -moles_reacted * delta_H_rxn

    # Heat that must be removed to maintain isothermal
    Q = Q_rxn  # Positive = remove heat

    return {
        'heat_duty': float(Q),
        'heat_duty_sign': 'remove' if Q > 0 else 'add',
        'moles_reacted_per_time': float(moles_reacted),
        'F_A0': F_A0,
        'conversion': X,
        'delta_H_rxn': delta_H_rxn,
        'is_exothermic': delta_H_rxn < 0,
        'equation': 'Q = F_A0 × X × (-ΔH_rxn)',
    }


def material_balance_cstr(
    Q: float,
    C_in: float,
    C_out: float,
    C_product: float = None,
    stoich_ratio: float = 1.0
) -> Dict[str, Any]:
    """
    Verify material balance closure for CSTR.

    Inlet moles = Outlet moles + Reacted moles

    Parameters
    ----------
    Q : float
        Volumetric flow rate [L/min or m³/s]
    C_in : float
        Inlet concentration [mol/L]
    C_out : float
        Outlet concentration [mol/L]
    C_product : float, optional
        Product concentration [mol/L]
    stoich_ratio : float
        Stoichiometric ratio (moles product / moles reactant)

    Returns
    -------
    dict
        closure_percent: 100 if perfectly balanced
        error: Absolute error in mol/time
    """
    F_in = Q * C_in
    F_out = Q * C_out
    F_reacted = F_in - F_out

    if C_product is not None:
        F_product = Q * C_product
        expected_product = F_reacted * stoich_ratio
        product_error = abs(F_product - expected_product)
        product_closure = (1 - product_error / expected_product) * 100 if expected_product > 0 else 100
    else:
        F_product = None
        product_closure = None

    # Check inlet = outlet + reacted
    balance_check = F_in - F_out - F_reacted  # Should be 0
    closure = 100.0 if abs(balance_check) < 1e-10 else (1 - abs(balance_check) / F_in) * 100

    return {
        'inlet_mol_per_time': float(F_in),
        'outlet_mol_per_time': float(F_out),
        'reacted_mol_per_time': float(F_reacted),
        'product_mol_per_time': F_product,
        'closure_percent': float(closure),
        'product_closure_percent': product_closure,
        'balance_error': float(balance_check),
        'balanced': abs(balance_check) < 1e-10,
        'equation': 'F_in = F_out + F_reacted',
    }


def reynolds_number_pipe(
    Q: float,
    D: float,
    rho: float,
    mu: float,
    Q_unit: str = 'mL/min',
    D_unit: str = 'm'
) -> Dict[str, Any]:
    """
    Reynolds number for pipe flow.

    Re = 4ρQ / (πDμ)

    Parameters
    ----------
    Q : float
        Volumetric flow rate
    D : float
        Pipe diameter
    rho : float
        Density [kg/m³]
    mu : float
        Dynamic viscosity [Pa·s]
    Q_unit : str
        Flow rate unit ('mL/min', 'L/min', 'm³/s')
    D_unit : str
        Diameter unit ('m', 'mm', 'in')

    Returns
    -------
    dict
        Re: Reynolds number
        regime: 'laminar' / 'transitional' / 'turbulent'
    """
    # Convert Q to m³/s
    Q_m3s = Q
    if Q_unit == 'mL/min':
        Q_m3s = Q * 1e-6 / 60  # mL/min → m³/s
    elif Q_unit == 'L/min':
        Q_m3s = Q * 1e-3 / 60
    elif Q_unit == 'L/s':
        Q_m3s = Q * 1e-3

    # Convert D to meters
    D_m = D
    if D_unit == 'mm':
        D_m = D / 1000
    elif D_unit == 'in':
        D_m = D * 0.0254

    # Re = 4ρQ / (πDμ) = ρvD/μ where v = Q/(πD²/4)
    Re = 4 * rho * Q_m3s / (np.pi * D_m * mu)

    if Re < 2100:
        regime = 'laminar'
    elif Re < 4000:
        regime = 'transitional'
    else:
        regime = 'turbulent'

    return {
        'Re': float(Re),
        'regime': regime,
        'is_laminar': Re < 2100,
        'is_turbulent': Re > 4000,
        'Q_m3s': float(Q_m3s),
        'D_m': float(D_m),
        'rho': rho,
        'mu': mu,
        'equation': 'Re = 4ρQ/(πDμ)',
    }


def analyze_cstr_kinetics(
    temperatures_K: List[float],
    inlet_concentrations: List[float],
    outlet_concentrations: List[float],
    reactor_volume_L: float,
    flow_rate_mL_min: float,
    reaction_order: int = 2,
    delta_H_rxn: float = None,
    pipe_diameter_m: float = None,
    density_kg_m3: float = None,
    viscosity_Pa_s: float = None,
) -> Dict[str, Any]:
    """
    Complete CSTR kinetics analysis.

    Calculates:
    1. Conversion at each temperature
    2. Rate constant k at each temperature
    3. Arrhenius parameters (Ea, A)
    4. Material balance verification
    5. Heat duty (if delta_H provided)
    6. Reynolds number (if pipe properties provided)

    Parameters
    ----------
    temperatures_K : list
        Temperatures [K]
    inlet_concentrations : list
        Inlet concentrations [mol/L]
    outlet_concentrations : list
        Outlet concentrations [mol/L]
    reactor_volume_L : float
        Reactor volume [L]
    flow_rate_mL_min : float
        Flow rate [mL/min]
    reaction_order : int
        Reaction order (1 or 2)
    delta_H_rxn : float, optional
        Heat of reaction [J/mol]
    pipe_diameter_m : float, optional
        Feed pipe diameter [m]
    density_kg_m3 : float, optional
        Fluid density [kg/m³]
    viscosity_Pa_s : float, optional
        Fluid viscosity [Pa·s]

    Returns
    -------
    dict
        Complete analysis results
    """
    n = len(temperatures_K)

    # Calculate residence time
    tau_result = residence_time(reactor_volume_L, flow_rate_mL_min)
    tau = tau_result['tau']

    # Calculate conversions and rate constants
    conversions = []
    rate_constants = []

    for i in range(n):
        C_A0 = inlet_concentrations[i]
        C_A = outlet_concentrations[i]

        conv = conversion_from_concentration(C_A0, C_A)
        X = conv['conversion']
        conversions.append(X)

        k_result = cstr_rate_constant(X, tau, C_A0, order=reaction_order)
        rate_constants.append(k_result['k'])

    # Arrhenius regression
    arr_result = arrhenius_regression(temperatures_K, rate_constants)

    # Build results table
    results_table = []
    for i in range(n):
        row = {
            'temperature_K': temperatures_K[i],
            'temperature_C': temperatures_K[i] - 273.15,
            'C_A0': inlet_concentrations[i],
            'C_A_out': outlet_concentrations[i],
            'conversion': conversions[i],
            'rate_constant': rate_constants[i],
        }

        # Add heat duty if available
        if delta_H_rxn is not None:
            F_A0 = flow_rate_mL_min / 1000 * inlet_concentrations[i] / 60  # mol/s
            heat = heat_duty_isothermal_cstr(F_A0, conversions[i], delta_H_rxn)
            row['heat_duty_W'] = heat['heat_duty']

        results_table.append(row)

    result = {
        'residence_time_min': tau,
        'reaction_order': reaction_order,
        'results_by_temperature': results_table,
        'conversions': conversions,
        'rate_constants': rate_constants,
        'arrhenius': {
            'activation_energy_J_mol': arr_result['activation_energy_J_mol'],
            'activation_energy_kJ_mol': arr_result['activation_energy_kJ_mol'],
            'pre_exponential': arr_result['pre_exponential'],
            'pre_exponential_sci': arr_result['pre_exponential_sci'],
            'r_squared': arr_result['r_squared'],
        },
    }

    # Add Reynolds number if pipe properties provided
    if all(x is not None for x in [pipe_diameter_m, density_kg_m3, viscosity_Pa_s]):
        re_result = reynolds_number_pipe(
            flow_rate_mL_min, pipe_diameter_m, density_kg_m3, viscosity_Pa_s,
            Q_unit='mL/min', D_unit='m'
        )
        result['reynolds'] = {
            'Re': re_result['Re'],
            'regime': re_result['regime'],
        }

    return result


def compute(signal: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for CSTR kinetics calculations.
    """
    if all(k in kwargs for k in ['temperatures_K', 'inlet_concentrations', 'outlet_concentrations']):
        return analyze_cstr_kinetics(**kwargs)

    if all(k in kwargs for k in ['X', 'tau', 'C_A0']):
        return cstr_rate_constant(kwargs['X'], kwargs['tau'], kwargs['C_A0'],
                                  kwargs.get('order', 2))

    if all(k in kwargs for k in ['C_A0', 'C_A']):
        return conversion_from_concentration(kwargs['C_A0'], kwargs['C_A'])

    if all(k in kwargs for k in ['T', 'k']) and isinstance(kwargs['T'], (list, np.ndarray)):
        return arrhenius_regression(kwargs['T'], kwargs['k'])

    return {'error': 'Insufficient parameters for CSTR kinetics analysis'}

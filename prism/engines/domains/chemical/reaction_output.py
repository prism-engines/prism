"""
Reaction Engineering Output → ORTHON

PRISM outputs pure numerical results to standard parquet files.
ORTHON reads discipline tag and applies discipline-specific interpretation.

PRISM outputs to physics.parquet with:
- All calculated metrics (k, X, Ea, Re, etc.)
- discipline='reaction' tag for ORTHON

ORTHON then:
- Reads discipline tag
- Applies reaction-specific formatting
- Generates reports, figures, labels
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


def calculate_reaction_metrics(
    entity_ids: List[str],
    temperatures_K: List[float],
    inlet_concentrations: List[float],
    outlet_concentrations: List[float],
    reactor_volume_L: float,
    flow_rate_mL_min: float,
    reaction_order: int = 2,
    heat_of_reaction: float = None,
    pipe_diameter_m: float = None,
    density_kg_m3: float = None,
    viscosity_Pa_s: float = None,
) -> pl.DataFrame:
    """
    Calculate all reaction metrics and return as DataFrame.

    PRISM calculates numbers. ORTHON interprets based on discipline.

    Returns DataFrame with columns:
    - entity_id: run identifier
    - discipline: 'reaction' (for ORTHON)
    - temperature_K, temperature_C
    - conversion, rate_constant, ln_rate_constant
    - inv_temperature (for Arrhenius plot)
    - reaction_rate
    - heat_duty_W (if heat_of_reaction provided)
    - reynolds (if pipe properties provided)
    - inlet_molar_flow, outlet_molar_flow, reacted_molar_flow
    - balance_closure_pct
    """
    from prism.engines.domains.chemical.cstr_kinetics import (
        conversion_from_concentration,
        cstr_rate_constant,
        residence_time,
        heat_duty_isothermal_cstr,
        reynolds_number_pipe,
    )

    n = len(entity_ids)

    # Residence time
    tau_result = residence_time(reactor_volume_L, flow_rate_mL_min)
    tau = tau_result['tau']

    # Calculate per-run metrics
    data = {
        'entity_id': entity_ids,
        'discipline': ['reaction'] * n,
        'temperature_K': temperatures_K,
        'temperature_C': [T - 273.15 for T in temperatures_K],
        'inv_temperature': [1/T for T in temperatures_K],
        'C_inlet': inlet_concentrations,
        'C_outlet': outlet_concentrations,
        'residence_time_min': [tau] * n,
        'reaction_order': [reaction_order] * n,
    }

    conversions = []
    rate_constants = []
    reaction_rates = []

    for i in range(n):
        conv = conversion_from_concentration(inlet_concentrations[i], outlet_concentrations[i])
        conversions.append(conv['conversion'])

        k_result = cstr_rate_constant(conv['conversion'], tau, inlet_concentrations[i],
                                       order=reaction_order)
        rate_constants.append(k_result['k'])
        reaction_rates.append(k_result['reaction_rate'])

    data['conversion'] = conversions
    data['rate_constant'] = rate_constants
    data['ln_rate_constant'] = [np.log(k) if k > 0 else np.nan for k in rate_constants]
    data['reaction_rate'] = reaction_rates

    # Heat duty
    if heat_of_reaction is not None:
        Q_L_min = flow_rate_mL_min / 1000
        heat_duties = []
        for i in range(n):
            F_A0_mol_s = Q_L_min * inlet_concentrations[i] / 60
            heat = heat_duty_isothermal_cstr(F_A0_mol_s, conversions[i], heat_of_reaction)
            heat_duties.append(heat['heat_duty'])
        data['heat_duty_W'] = heat_duties
        data['heat_of_reaction_J_mol'] = [heat_of_reaction] * n

    # Reynolds number
    if all(x is not None for x in [pipe_diameter_m, density_kg_m3, viscosity_Pa_s]):
        re_result = reynolds_number_pipe(flow_rate_mL_min, pipe_diameter_m,
                                          density_kg_m3, viscosity_Pa_s)
        data['reynolds'] = [re_result['Re']] * n

    # Material balance
    Q_L_min = flow_rate_mL_min / 1000
    inlet_flows = [Q_L_min * c for c in inlet_concentrations]
    outlet_flows = [Q_L_min * c for c in outlet_concentrations]
    reacted_flows = [i - o for i, o in zip(inlet_flows, outlet_flows)]

    data['inlet_molar_flow'] = inlet_flows
    data['outlet_molar_flow'] = outlet_flows
    data['reacted_molar_flow'] = reacted_flows
    data['balance_closure_pct'] = [100.0] * n  # Perfect closure by definition

    return pl.DataFrame(data)


def calculate_arrhenius_metrics(
    temperatures_K: List[float],
    rate_constants: List[float],
) -> Dict[str, float]:
    """
    Calculate Arrhenius parameters.

    Returns dict (not DataFrame) - these are aggregate metrics.
    PRISM includes in physics.parquet metadata or separate row.
    """
    from prism.engines.domains.chemical.cstr_kinetics import arrhenius_regression

    arr = arrhenius_regression(temperatures_K, rate_constants)

    return {
        'activation_energy_J_mol': arr['activation_energy_J_mol'],
        'activation_energy_kJ_mol': arr['activation_energy_kJ_mol'],
        'pre_exponential': arr['pre_exponential'],
        'arrhenius_r_squared': arr['r_squared'],
        'arrhenius_slope': arr['slope'],
        'arrhenius_intercept': arr['intercept'],
    }


def to_physics_parquet(
    df: pl.DataFrame,
    arrhenius_metrics: Dict[str, float],
    output_path: str,
) -> str:
    """
    Write to physics.parquet format.

    Adds Arrhenius metrics as columns (same value for all rows).
    ORTHON reads discipline='reaction' and knows how to interpret.
    """
    # Add Arrhenius metrics to all rows
    for key, value in arrhenius_metrics.items():
        df = df.with_columns(pl.lit(value).alias(key))

    # Write
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    return str(output_path)


def compute_reaction_to_orthon(
    output_path: str,
    entity_ids: List[str],
    temperatures_K: List[float],
    inlet_concentrations: List[float],
    outlet_concentrations: List[float],
    reactor_volume_L: float,
    flow_rate_mL_min: float,
    reaction_order: int = 2,
    heat_of_reaction: float = None,
    pipe_diameter_m: float = None,
    density_kg_m3: float = None,
    viscosity_Pa_s: float = None,
) -> Dict[str, Any]:
    """
    Complete PRISM → ORTHON pipeline for reaction discipline.

    PRISM calculates all metrics, outputs to physics.parquet.
    ORTHON reads discipline='reaction' and handles formatting.

    Parameters
    ----------
    output_path : str
        Path to physics.parquet output
    ... (other params same as calculate_reaction_metrics)

    Returns
    -------
    dict
        path: output file path
        summary: key metrics for logging
    """
    # Calculate per-run metrics
    df = calculate_reaction_metrics(
        entity_ids=entity_ids,
        temperatures_K=temperatures_K,
        inlet_concentrations=inlet_concentrations,
        outlet_concentrations=outlet_concentrations,
        reactor_volume_L=reactor_volume_L,
        flow_rate_mL_min=flow_rate_mL_min,
        reaction_order=reaction_order,
        heat_of_reaction=heat_of_reaction,
        pipe_diameter_m=pipe_diameter_m,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
    )

    # Calculate Arrhenius parameters
    arrhenius = calculate_arrhenius_metrics(
        temperatures_K=temperatures_K,
        rate_constants=df['rate_constant'].to_list(),
    )

    # Write to physics.parquet
    path = to_physics_parquet(df, arrhenius, output_path)

    return {
        'path': path,
        'discipline': 'reaction',
        'n_runs': len(entity_ids),
        'summary': {
            'residence_time_min': df['residence_time_min'][0],
            'activation_energy_kJ_mol': arrhenius['activation_energy_kJ_mol'],
            'pre_exponential': arrhenius['pre_exponential'],
            'r_squared': arrhenius['arrhenius_r_squared'],
            'reynolds': df['reynolds'][0] if 'reynolds' in df.columns else None,
        }
    }

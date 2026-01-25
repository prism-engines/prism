"""
Material & Energy Balance Engines

Conservation equations, process calculations, heat exchanger design,
degrees of freedom analysis.
"""

from .material import (
    total_mass_balance,
    component_balance,
    extent_of_reaction,
    limiting_reactant,
    excess_reactant,
    recycle_ratio,
    purge_calculation,
    bypass_calculation,
    mixing_balance,
    splitting_balance,
)

from .energy import (
    sensible_heat,
    sensible_heat_integral,
    latent_heat,
    enthalpy_balance,
    heat_of_reaction_calc,
    adiabatic_flame_temp,
    adiabatic_reaction_temp,
    heat_exchanger_duty,
    lmtd,
    effectiveness_ntu,
    ntu_from_effectiveness,
)

from .combined import (
    degrees_of_freedom,
    unit_dof,
    process_simulation_sequential,
    recycle_convergence,
    mass_energy_summary,
    equation_ordering,
)

__all__ = [
    # Material Balances
    'total_mass_balance',
    'component_balance',
    'extent_of_reaction',
    'limiting_reactant',
    'excess_reactant',
    'recycle_ratio',
    'purge_calculation',
    'bypass_calculation',
    'mixing_balance',
    'splitting_balance',
    # Energy Balances
    'sensible_heat',
    'sensible_heat_integral',
    'latent_heat',
    'enthalpy_balance',
    'heat_of_reaction_calc',
    'adiabatic_flame_temp',
    'adiabatic_reaction_temp',
    'heat_exchanger_duty',
    'lmtd',
    'effectiveness_ntu',
    'ntu_from_effectiveness',
    # Combined Balances
    'degrees_of_freedom',
    'unit_dof',
    'process_simulation_sequential',
    'recycle_convergence',
    'mass_energy_summary',
    'equation_ordering',
]

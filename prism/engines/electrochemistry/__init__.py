"""
Electrochemistry Engines

Nernst equation, Butler-Volmer kinetics, Faraday's law, battery analysis,
electrochemical mass transfer.
"""

from .nernst import (
    nernst,
    cell_potential,
    gibbs_electrochemical,
    equilibrium_constant_echem,
    pourbaix,
    concentration_cell,
    temperature_coefficient,
)

from .kinetics import (
    butler_volmer,
    tafel,
    limiting_current,
    mixed_potential,
    polarization_curve,
    exchange_current_density,
)

from .faraday import (
    faraday,
    faraday_constant,
    coulombic_efficiency,
    energy_efficiency,
    corrosion_rate,
    electroplating_thickness,
    electrolysis_power,
    specific_energy_consumption,
)

from .battery import (
    state_of_charge,
    state_of_health,
    c_rate,
    peukert,
    internal_resistance_calc,
    capacity_fade,
    cycle_life,
    impedance_spectrum,
    ocv_soc_curve,
)

from .mass_transfer import (
    limiting_current_density,
    rotating_disk,
    levich_plot,
    koutecky_levich,
    concentration_overpotential,
    diffusion_layer_thickness,
    sherwood_electrode,
)

__all__ = [
    # Thermodynamics (Nernst)
    'nernst',
    'cell_potential',
    'gibbs_electrochemical',
    'equilibrium_constant_echem',
    'pourbaix',
    'concentration_cell',
    'temperature_coefficient',
    # Kinetics
    'butler_volmer',
    'tafel',
    'limiting_current',
    'mixed_potential',
    'polarization_curve',
    'exchange_current_density',
    # Faraday / Applications
    'faraday',
    'faraday_constant',
    'coulombic_efficiency',
    'energy_efficiency',
    'corrosion_rate',
    'electroplating_thickness',
    'electrolysis_power',
    'specific_energy_consumption',
    # Battery
    'state_of_charge',
    'state_of_health',
    'c_rate',
    'peukert',
    'internal_resistance_calc',
    'capacity_fade',
    'cycle_life',
    'impedance_spectrum',
    'ocv_soc_curve',
    # Mass Transfer
    'limiting_current_density',
    'rotating_disk',
    'levich_plot',
    'koutecky_levich',
    'concentration_overpotential',
    'diffusion_layer_thickness',
    'sherwood_electrode',
]

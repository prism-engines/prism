"""
Phase Equilibria Engines

VLE, LLE, flash calculations, activity coefficient models.
"""

from .vle import (
    antoine,
    raoults_law,
    modified_raoults,
    k_value,
    relative_volatility,
    bubble_point_pressure,
    bubble_point_temperature,
    dew_point_pressure,
    dew_point_temperature,
    txy_diagram,
    pxy_diagram,
)

from .flash import (
    rachford_rice,
    isothermal_flash,
    adiabatic_flash,
    three_phase_flash,
)

from .activity_models import (
    ideal_solution,
    margules,
    margules_two_suffix,
    van_laar,
    wilson,
    wilson_binary,
    nrtl,
    nrtl_binary,
    uniquac,
    unifac,
)

from .lle import (
    tie_line,
    lever_rule,
    binodal_curve_margules,
    plait_point,
    ternary_coordinates,
    ternary_diagram,
    distribution_coefficient,
)

__all__ = [
    # VLE
    'antoine',
    'raoults_law',
    'modified_raoults',
    'k_value',
    'relative_volatility',
    'bubble_point_pressure',
    'bubble_point_temperature',
    'dew_point_pressure',
    'dew_point_temperature',
    'txy_diagram',
    'pxy_diagram',
    # Flash
    'rachford_rice',
    'isothermal_flash',
    'adiabatic_flash',
    'three_phase_flash',
    # Activity Models
    'ideal_solution',
    'margules',
    'margules_two_suffix',
    'van_laar',
    'wilson',
    'wilson_binary',
    'nrtl',
    'nrtl_binary',
    'uniquac',
    'unifac',
    # LLE
    'tie_line',
    'lever_rule',
    'binodal_curve_margules',
    'plait_point',
    'ternary_coordinates',
    'ternary_diagram',
    'distribution_coefficient',
]

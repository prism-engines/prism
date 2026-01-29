"""
Physics Engines (PR #11)

Four engines that enforce conservation laws and detect constitutive drift:
1. EnergyEngine - Energy balance: Ein = Eout + Estored + Edissipated
2. MassEngine - Mass balance: m_dot_in = m_dot_out + dm/dt
3. MomentumEngine - Momentum balance: sum(tau) = I*alpha, sum(F) = m*a
4. ConstitutiveEngine - Constitutive relationship tracking and drift detection

Conservation law violations ARE anomalies. No ML needed.
"""

from .energy_engine import EnergyEngine, EnergyConfig, run_energy_engine
from .mass_engine import MassEngine, MassConfig, run_mass_engine
from .momentum_engine import MomentumEngine, MomentumConfig, run_momentum_engine
from .constitutive_engine import ConstitutiveEngine, ConstitutiveConfig, run_constitutive_engine

__all__ = [
    # Engines
    'EnergyEngine',
    'MassEngine',
    'MomentumEngine',
    'ConstitutiveEngine',
    # Configs
    'EnergyConfig',
    'MassConfig',
    'MomentumConfig',
    'ConstitutiveConfig',
    # Runner functions
    'run_energy_engine',
    'run_mass_engine',
    'run_momentum_engine',
    'run_constitutive_engine',
]

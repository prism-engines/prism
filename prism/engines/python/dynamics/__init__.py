"""
Dynamics Engines (PR #12)

Four engines that analyze system stability and early warning signals:
1. LyapunovEngine - Lyapunov exponents and stability classification
2. AttractorEngine - Attractor dimension and structure changes
3. RecurrenceEngine - Recurrence Quantification Analysis (RQA)
4. BifurcationEngine - Critical slowing down detection

Key insight: Dynamical systems show warning signs BEFORE failure.
"""

from .lyapunov_engine import LyapunovEngine, LyapunovConfig, run_lyapunov_engine
from .attractor_engine import AttractorEngine, AttractorConfig, run_attractor_engine
from .recurrence_engine import RecurrenceEngine, RecurrenceConfig, run_recurrence_engine
from .bifurcation_engine import BifurcationEngine, BifurcationConfig, run_bifurcation_engine

__all__ = [
    # Engines
    'LyapunovEngine',
    'AttractorEngine',
    'RecurrenceEngine',
    'BifurcationEngine',
    # Configs
    'LyapunovConfig',
    'AttractorConfig',
    'RecurrenceConfig',
    'BifurcationConfig',
    # Runner functions
    'run_lyapunov_engine',
    'run_attractor_engine',
    'run_recurrence_engine',
    'run_bifurcation_engine',
]

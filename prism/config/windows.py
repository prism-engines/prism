"""
PRISM Window Configuration
==========================
Shared window parameters for all geometry computations.

Ensures Method A (cohort_geometry), Method B (system_window), and Residual
all use identical windows - eliminating alignment gaps.

FAST LOOKUP DICTS:
- WINDOW_WEIGHTS: Weights for aggregation across window tiers
- MIN_OBSERVATIONS: Minimum observations per window tier
- ENGINE_MIN_OBS: Minimum observations per vector engine
"""

from dataclasses import dataclass, field
from typing import Literal, List, Dict
import hashlib
import json


# =============================================================================
# FAST LOOKUP DICTS (Python dicts for fastest runtime access)
# =============================================================================

# Window weights for geometry/state aggregation
# Prevents overweighting short-term tiers when combining across windows
WINDOW_WEIGHTS: Dict[int, float] = {
    252: 1.0,    # anchor - structural truth
    126: 0.75,   # bridge - medium-term
    63:  0.50,   # scout - short-term
    21:  0.25,   # micro - daily dynamics
}

# Minimum observations required per window tier
MIN_OBSERVATIONS: Dict[int, int] = {
    252: 200,
    126: 100,
    63:  50,
    21:  15,
}

# Engine minimum observations (vector layer)
ENGINE_MIN_OBS: Dict[str, int] = {
    'hurst': 20,
    'entropy': 30,
    'lyapunov': 30,
    'garch': 50,
    'spectral': 40,
    'wavelet': 40,
    'rqa': 30,
    'realized_vol': 15,
}


def get_window_weight(window_days: int) -> float:
    """Get weight for a window size. Returns 0.5 if unknown."""
    return WINDOW_WEIGHTS.get(window_days, 0.5)


def get_min_obs(window_days: int) -> int:
    """Get minimum observations for a window size."""
    return MIN_OBSERVATIONS.get(window_days, 15)


# =============================================================================
# DOMAIN WINDOW CONFIGS
# =============================================================================


@dataclass
class DomainWindowConfig:
    """
    Domain-specific window configuration for signal_geometry.

    Defines the window sizes to compute for each domain based on
    data frequency (daily vs monthly vs annual).
    """
    # Window sizes in observations (not days!)
    # For daily data: 63/126/252 = quarter/half/full year
    # For annual data: 5/15/30 = 5/15/30 years
    window_sizes: List[int] = field(default_factory=lambda: [63, 126, 252])

    # Step size for rolling windows
    step_size: int = 5

    # Data frequency for labeling
    frequency: Literal['daily', 'monthly', 'annual'] = 'daily'

    # Minimum observations per signal per window
    min_observations_ratio: float = 0.5  # min_obs = window_size * ratio

    @classmethod
    def for_daily(cls) -> 'DomainWindowConfig':
        """Daily data: 63/126/252 observations = quarter/half/full year."""
        return cls(
            window_sizes=[63, 126, 252],
            step_size=5,
            frequency='daily',
        )

    @classmethod
    def for_annual(cls) -> 'DomainWindowConfig':
        """Annual data: 5/15/30 years."""
        return cls(
            window_sizes=[5, 15, 30],
            step_size=1,  # Every year
            frequency='annual',
            min_observations_ratio=0.6,  # Need 60% of window filled
        )

    @classmethod
    def for_monthly(cls) -> 'DomainWindowConfig':
        """Monthly data: 12/36/60 months = 1/3/5 years."""
        return cls(
            window_sizes=[12, 36, 60],
            step_size=1,  # Every month
            frequency='monthly',
        )


# Domain -> Window Configuration mapping
# Runner MUST find domain here or fail
DOMAIN_WINDOW_CONFIGS = {
    # Regional domains (annual data)
    'world': DomainWindowConfig.for_annual(),
    'G7': DomainWindowConfig.for_annual(),
    'NORDIC': DomainWindowConfig.for_annual(),
    'BRICS': DomainWindowConfig.for_annual(),
    'EU_CORE': DomainWindowConfig.for_annual(),
    'ASEAN': DomainWindowConfig.for_annual(),
    'ASIA_TIGERS': DomainWindowConfig.for_annual(),
    'LATAM': DomainWindowConfig.for_annual(),
    'MENA': DomainWindowConfig.for_annual(),
    'SSA': DomainWindowConfig.for_annual(),
    'OCEANIA': DomainWindowConfig.for_annual(),

    # Climate domain (monthly data)
    'climate': DomainWindowConfig.for_monthly(),
}


def get_window_config_for_domain(domain: str) -> DomainWindowConfig:
    """
    Get window configuration for a domain.

    Raises KeyError if domain not configured - this is intentional
    to prevent running with wrong window sizes.
    """
    if domain not in DOMAIN_WINDOW_CONFIGS:
        available = ', '.join(sorted(DOMAIN_WINDOW_CONFIGS.keys()))
        raise KeyError(
            f"No window config for domain '{domain}'. "
            f"Available domains: {available}. "
            f"Add config to DOMAIN_WINDOW_CONFIGS in prism/config/windows.py"
        )
    return DOMAIN_WINDOW_CONFIGS[domain]


@dataclass
class WindowConfig:
    """
    Shared window configuration for all geometry computations.

    Both Method A and Method B use these parameters, ensuring
    identical window_end dates for residual analysis.
    """

    # Window size in observations
    window_size: int = 252  # 1 year of daily data

    # Step size (how often to compute a new window)
    step_size: int = 21  # Monthly (~21 observations)

    # Window end convention
    end_convention: Literal['end_of_month', 'mid_month', 'exact'] = 'end_of_month'

    # Minimum observations required per signal
    min_observations: int = 100

    # Minimum signals required per window
    min_signals: int = 3

    @classmethod
    def monthly(cls) -> 'WindowConfig':
        """Standard monthly windows (end of month)."""
        return cls(
            window_size=252,
            step_size=21,
            end_convention='end_of_month',
        )

    @classmethod
    def weekly(cls) -> 'WindowConfig':
        """Weekly windows for higher frequency analysis."""
        return cls(
            window_size=252,
            step_size=5,
            end_convention='exact',
        )

    @classmethod
    def quarterly(cls) -> 'WindowConfig':
        """Quarterly windows for macro analysis."""
        return cls(
            window_size=252,
            step_size=63,
            end_convention='end_of_month',
        )

    @classmethod
    def for_climate(cls) -> 'WindowConfig':
        """
        Climate data uses monthly observations, not daily.
        Window size = 60 months (5 years), step = 1 month.
        """
        return cls(
            window_size=60,
            step_size=1,
            end_convention='mid_month',  # Climate data uses 15th
            min_observations=48,
            min_signals=3,
        )

    @property
    def config_hash(self) -> str:
        """Generate hash for this config (for tracking in DB)."""
        config_dict = {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'end_convention': self.end_convention,
            'min_observations': self.min_observations,
            'min_signals': self.min_signals,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return (
            f"WindowConfig(size={self.window_size}, step={self.step_size}, "
            f"end={self.end_convention}, hash={self.config_hash})"
        )


# Default configuration
DEFAULT_WINDOW_CONFIG = WindowConfig.monthly()

# Domain-specific defaults
DOMAIN_CONFIGS = {
    'macro': WindowConfig.quarterly(),
    'climate': WindowConfig.for_climate(),
}


def get_config_for_domain(domain: str) -> WindowConfig:
    """Get appropriate window config for a domain."""
    return DOMAIN_CONFIGS.get(domain, DEFAULT_WINDOW_CONFIG)

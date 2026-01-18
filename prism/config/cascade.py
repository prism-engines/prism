"""
PRISM Cascade Geometry Configuration
=====================================
Window sizes and flag thresholds for multi-pass geometry computation.

The cascade system progressively drills down on flagged pairs:
- Pass 1: All pairs at largest window (coarse scan)
- Pass 2: Flagged pairs at medium window (structural shift detection)
- Pass 3: Double-flagged pairs at finest window (event classification)

This reduces compute by ~60% while capturing structural events.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# Window Cascade Configuration
# =============================================================================

WINDOW_CASCADE = {
    'daily': {
        'pass_1': 252,  # ~1 year daily observations
        'pass_2': 126,  # ~6 months
        'pass_3': 63,   # ~3 months
    },
    'annual': {
        'pass_1': 20,   # 20 years
        'pass_2': 10,   # 10 years
        'pass_3': 5,    # 5 years
    },
    'monthly': {
        'pass_1': 60,   # 5 years
        'pass_2': 36,   # 3 years
        'pass_3': 12,   # 1 year
    }
}


# =============================================================================
# Flag Thresholds
# =============================================================================

FLAG_THRESHOLDS = {
    # Structural shift: geometry changed > 30% between windows
    'structural_shift': 0.3,

    # Entropy coupling thresholds
    'entropy_coupling_high': 0.7,   # unusually high coupling
    'entropy_coupling_low': 0.2,    # unusually low (decoupling)

    # Phase coherence breakdown threshold
    'phase_coherence_break': 0.4,

    # Correlation threshold for mismatch detection
    'correlation_threshold': 0.7,

    # Phase coherence thresholds for breakdown detection
    'phase_coherence_high': 0.6,    # was coherent
    'phase_coherence_low': 0.3,     # now broken

    # Structural variance multiplier for volatility cluster detection
    'variance_spike_multiplier': 2.0,

    # Multi-metric shift count for regime change classification
    'regime_change_shift_count': 2,
    'metric_shift_threshold': 0.3,
}


# =============================================================================
# Event Classification Labels
# =============================================================================

EVENT_TYPES = {
    'decoupling': 'Signals that were coupled became independent',
    'coupling_emergence': 'Signals that were independent became coupled',
    'phase_shift': 'Phase coherence changed sign or broke',
    'volatility_cluster': 'Structural variance spiked significantly',
    'regime_change': 'Multiple metrics shifted simultaneously',
    'relationship_inversion': 'Correlation sign flipped',
    'unclassified': 'Structural change not matching known patterns',
}


# =============================================================================
# Dataclass Configuration
# =============================================================================

@dataclass
class CascadeConfig:
    """
    Configuration for cascade geometry computation.
    """
    # Window sizes for each pass
    pass_1_window: int
    pass_2_window: int
    pass_3_window: int

    # Data frequency
    frequency: str = 'daily'

    # Flag thresholds (use defaults if not specified)
    thresholds: Dict[str, float] = field(default_factory=lambda: FLAG_THRESHOLDS.copy())

    @classmethod
    def for_daily(cls) -> 'CascadeConfig':
        """Configuration for daily data."""
        return cls(
            pass_1_window=252,
            pass_2_window=126,
            pass_3_window=63,
            frequency='daily',
        )

    @classmethod
    def for_annual(cls) -> 'CascadeConfig':
        """Configuration for annual data."""
        return cls(
            pass_1_window=20,
            pass_2_window=10,
            pass_3_window=5,
            frequency='annual',
        )

    @classmethod
    def for_monthly(cls) -> 'CascadeConfig':
        """Configuration for monthly data (climate)."""
        return cls(
            pass_1_window=60,
            pass_2_window=36,
            pass_3_window=12,
            frequency='monthly',
        )

    @property
    def windows(self) -> Dict[str, int]:
        """Get windows as dict."""
        return {
            'pass_1': self.pass_1_window,
            'pass_2': self.pass_2_window,
            'pass_3': self.pass_3_window,
        }


# =============================================================================
# Domain Configuration Mapping
# =============================================================================

DOMAIN_CASCADE_CONFIGS = {
    # Regional domains (annual data)
    'world': CascadeConfig.for_annual(),
    'G7': CascadeConfig.for_annual(),
    'NORDIC': CascadeConfig.for_annual(),
    'BRICS': CascadeConfig.for_annual(),
    'EU_CORE': CascadeConfig.for_annual(),
    'ASEAN': CascadeConfig.for_annual(),
    'ASIA_TIGERS': CascadeConfig.for_annual(),
    'LATAM': CascadeConfig.for_annual(),
    'MENA': CascadeConfig.for_annual(),
    'SSA': CascadeConfig.for_annual(),
    'OCEANIA': CascadeConfig.for_annual(),

    # Climate domain (monthly data)
    'climate': CascadeConfig.for_monthly(),
}


# =============================================================================
# API Functions
# =============================================================================

def get_cascade_config(conn=None, frequency: str = 'daily', domain: str = None) -> Dict[str, int]:
    """
    Load cascade config from domain_config table or use defaults.

    Priority:
    1. Database raw.domain_config if conn provided
    2. Domain-specific config if domain provided
    3. Frequency-based defaults

    Args:
        conn: Optional database connection for database lookup
        frequency: Data frequency ('daily', 'annual', 'monthly')
        domain: Optional domain name for domain-specific config

    Returns:
        Dict with pass_1, pass_2, pass_3 window sizes
    """
    # Try database config first
    if conn is not None:
        try:
            result = conn.execute("""
                SELECT value FROM raw.domain_config
                WHERE key = 'cascade_windows'
            """).fetchone()
            if result:
                return json.loads(result[0])
        except Exception:
            pass

    # Try domain-specific config
    if domain and domain in DOMAIN_CASCADE_CONFIGS:
        return DOMAIN_CASCADE_CONFIGS[domain].windows

    # Fall back to frequency defaults
    if frequency in WINDOW_CASCADE:
        return WINDOW_CASCADE[frequency]

    # Ultimate fallback
    return WINDOW_CASCADE['daily']


def get_cascade_config_for_domain(domain: str) -> CascadeConfig:
    """
    Get full cascade configuration for a domain.

    Args:
        domain: Domain name

    Returns:
        CascadeConfig instance

    Raises:
        KeyError: If domain not configured
    """
    if domain not in DOMAIN_CASCADE_CONFIGS:
        available = ', '.join(sorted(DOMAIN_CASCADE_CONFIGS.keys()))
        raise KeyError(
            f"No cascade config for domain '{domain}'. "
            f"Available domains: {available}. "
            f"Add config to DOMAIN_CASCADE_CONFIGS in prism/config/cascade.py"
        )
    return DOMAIN_CASCADE_CONFIGS[domain]


def get_thresholds(conn=None) -> Dict[str, float]:
    """
    Get flag thresholds from database or defaults.

    Args:
        conn: Optional database connection for database lookup

    Returns:
        Dict of threshold name -> value
    """
    if conn is not None:
        try:
            result = conn.execute("""
                SELECT value FROM raw.domain_config
                WHERE key = 'cascade_thresholds'
            """).fetchone()
            if result:
                return json.loads(result[0])
        except Exception:
            pass

    return FLAG_THRESHOLDS.copy()

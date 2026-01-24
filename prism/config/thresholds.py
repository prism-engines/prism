"""
ORTHON Classification Thresholds
================================

Centralized configuration for all classification thresholds across
the four ORTHON analytical layers.

Principle: Thresholds define when states change, marking "relevant times".
Adjusting these parameters changes transition sensitivity without recomputation.

Usage:
    from prism.config.thresholds import (
        TYPOLOGY_THRESHOLDS,
        GEOMETRY_THRESHOLDS,
        DYNAMICS_THRESHOLDS,
        TRANSITION_THRESHOLDS,
        get_domain_thresholds,
    )

Domain Overrides:
    thresholds = get_domain_thresholds('cmapss')
    # Returns merged default + domain-specific thresholds

Future: Streamlit UI for runtime threshold adjustment (see TODO below)
"""

from typing import Dict, Any, Optional
from copy import deepcopy


# =============================================================================
# SIGNAL TYPOLOGY THRESHOLDS
# =============================================================================
# Used in: prism/signal_typology/classify.py
# Purpose: Map 0-1 normalized scores to 5-level labels

TYPOLOGY_CLASSIFICATION = {
    # Default thresholds: [strong_low, weak_low, weak_high, strong_high]
    # Scores map to: [0, t1) -> strong_low, [t1, t2) -> weak_low, etc.
    'default': [0.25, 0.40, 0.60, 0.75],

    # Alternative presets
    'strict': [0.20, 0.35, 0.65, 0.80],   # Wider indeterminate zone
    'loose': [0.30, 0.45, 0.55, 0.70],    # Narrower indeterminate zone
    'binary': [0.50, 0.50, 0.50, 0.50],   # Simple high/low split
}

# Per-axis threshold overrides (axis-specific tuning)
TYPOLOGY_AXIS_THRESHOLDS = {
    'memory': [0.30, 0.45, 0.55, 0.70],       # Hurst clusters around 0.5
    'information': [0.25, 0.40, 0.60, 0.75],  # Default
    'frequency': [0.25, 0.40, 0.60, 0.75],    # Default
    'volatility': [0.25, 0.40, 0.60, 0.75],   # Default
    'wavelet': [0.25, 0.40, 0.60, 0.75],      # Default
    'derivatives': [0.20, 0.35, 0.65, 0.80],  # Kurtosis can be extreme
    'recurrence': [0.25, 0.40, 0.60, 0.75],   # Default
    'discontinuity': [0.15, 0.30, 0.50, 0.70],# Level shifts are rare
    'momentum': [0.30, 0.45, 0.55, 0.70],     # Same as memory (Hurst-based)
}

# Signal typology regime change detection
TYPOLOGY_TRANSITION = {
    'distance_threshold': 0.3,  # Fingerprint distance for regime change
    'axis_threshold': 0.3,      # Per-axis change for "moving" classification
}


# =============================================================================
# STRUCTURAL GEOMETRY THRESHOLDS
# =============================================================================
# Used in: prism/structural_geometry/engine_mapping.py
# Purpose: Classify network topology, stability, and leadership

GEOMETRY_TOPOLOGY = {
    'highly_connected_density': 0.7,   # Density above this = HIGHLY_CONNECTED
    'modular_silhouette': 0.4,         # Silhouette above this + >2 clusters = MODULAR (was 0.3)
    'hierarchical_density': 0.5,       # Density below this + hubs = HIERARCHICAL
}

GEOMETRY_STABILITY = {
    'breaking_severe_base': 2,         # Base severe decouplings for BREAKING (was 1)
    'breaking_severe_ratio': 0.1,      # Or: severe > this fraction of signals = BREAKING
    'weakening_pairs': 0.25,           # Decoupling rate above this = WEAKENING
}

GEOMETRY_LEADERSHIP = {
    'bidirectional_ratio': 0.5,        # If bidirectional > this * causal = BIDIRECTIONAL
    'contemporaneous_correlation': 0.5, # Correlation above this with no causality = CONTEMPORANEOUS
}

# Structure change detection
GEOMETRY_TRANSITION = {
    'distance_threshold': 0.3,         # Fingerprint distance for structure change
    'correlation_threshold': 0.5,      # Edge threshold for network construction
    'historical_window': 100,          # Window for baseline correlation
    'recent_window': 20,               # Window for recent correlation
}


# =============================================================================
# DYNAMICAL SYSTEMS THRESHOLDS
# =============================================================================
# Used in: prism/dynamical_systems/engine_mapping.py
# Purpose: Classify regime, stability, trajectory, attractor

DYNAMICS_REGIME = {
    'coupled': 0.7,                    # Correlation above this = COUPLED
    'decoupled': 0.3,                  # Correlation below this = DECOUPLED
    'transitioning': 0.2,              # Change rate above this = TRANSITIONING
}

DYNAMICS_STABILITY = {
    'stable_change': 0.08,             # Max change for STABLE (was 0.05, too tight)
    'evolving_change': 0.15,           # Max change for EVOLVING
    'unstable_change': 0.30,           # Max change for UNSTABLE
    # Above 0.30 = CRITICAL
}

DYNAMICS_TRAJECTORY = {
    'convergence_trend': 0.02,         # Positive trend = CONVERGING (was 0.01)
    'divergence_trend': -0.02,         # Negative trend = DIVERGING (was -0.01)
    'oscillation_ratio': 0.5,          # Sign changes above this = OSCILLATING
    'min_observations': 5,             # Minimum obs for trajectory classification
}

# Regime transition detection
DYNAMICS_TRANSITION = {
    'distance_threshold': 0.3,         # Fingerprint distance for regime change
    'escalation_threshold': 0.25,      # Change rate to escalate to mechanics
}


# =============================================================================
# TRANSITION DETECTION THRESHOLDS
# =============================================================================
# Used in: prism/dynamical_systems/transitions.py
# Purpose: Detect meaningful state changes between windows

TRANSITION_NUMERIC = {
    # Thresholds for "meaningful" change in numeric fields
    'stability': 0.2,                  # 20% of range
    'predictability': 0.15,
    'coupling': 0.15,
    'memory': 0.1,
}

TRANSITION_SEVERITY = {
    # Severity classification multipliers
    'mild_max': 2.0,                   # Delta > threshold but < 2x threshold
    'moderate_max': 3.0,               # Delta > 2x threshold OR sign change
    # Above 3x threshold = SEVERE
}

TRANSITION_TYPES = {
    # Conditions for transition type classification
    'bifurcation': 'stability crosses zero (stable → unstable)',
    'collapse': 'predictability or coupling drops > 2x threshold',
    'recovery': 'metrics improving after previous decline',
    'shift': 'categorical change (trajectory or attractor type)',
    'flip': 'memory crosses 0.5 (persistent ↔ anti-persistent)',
}


# =============================================================================
# STATE LAYER THRESHOLDS
# =============================================================================
# Used in: prism/engines/state/transition_detector.py
# Purpose: Detect regime transitions via divergence spikes

STATE_TRANSITION = {
    'zscore_threshold': 3.0,           # Z-score threshold for transition detection
    'stability_window': 5,             # Windows needed for full stability score
    'early_warning_ratio': 2.0,        # Gradient must be 2x baseline for warning
}


# =============================================================================
# DOMAIN-SPECIFIC OVERRIDES
# =============================================================================
# Each domain can override any threshold above.
# Use get_domain_thresholds('domain_name') to get merged config.

DOMAIN_OVERRIDES = {
    'cmapss': {
        # Turbofan degradation: tighter thresholds (failures are costly)
        'description': 'NASA C-MAPSS turbofan engine degradation',
        'DYNAMICS_STABILITY': {
            'stable_change': 0.05,         # Tighter than default
            'evolving_change': 0.12,
            'unstable_change': 0.25,
        },
        'TRANSITION_NUMERIC': {
            'stability': 0.15,             # More sensitive to stability changes
            'predictability': 0.12,
            'coupling': 0.12,
            'memory': 0.08,
        },
        'STATE_TRANSITION': {
            'zscore_threshold': 2.5,       # Earlier warning
        },
    },

    'femto': {
        # Bearing degradation: focus on vibration patterns
        'description': 'FEMTO/PRONOSTIA bearing degradation',
        'TYPOLOGY_AXIS_THRESHOLDS': {
            'frequency': [0.20, 0.35, 0.65, 0.80],  # Tighter on periodicity
            'volatility': [0.20, 0.35, 0.65, 0.80], # Tighter on clustering
        },
        'DYNAMICS_STABILITY': {
            'stable_change': 0.06,
            'evolving_change': 0.14,
        },
    },

    'hydraulic': {
        # Hydraulic system: multi-condition monitoring
        'description': 'UCI hydraulic system condition monitoring',
        'GEOMETRY_STABILITY': {
            'breaking_severe_base': 3,     # More tolerant (multiple subsystems)
            'weakening_pairs': 0.30,
        },
        'DYNAMICS_STABILITY': {
            'stable_change': 0.10,         # Looser (inherent system variation)
        },
    },

    'tep': {
        # Tennessee Eastman: chemical process with high variability
        'description': 'Tennessee Eastman chemical process',
        'DYNAMICS_STABILITY': {
            'stable_change': 0.12,         # Looser thresholds
            'evolving_change': 0.20,
            'unstable_change': 0.35,
        },
        'DYNAMICS_TRAJECTORY': {
            'convergence_trend': 0.03,     # Require stronger trends
            'divergence_trend': -0.03,
        },
        'STATE_TRANSITION': {
            'zscore_threshold': 3.5,       # Higher bar for process variability
        },
    },

    'cwru': {
        # CWRU bearing faults: classification focus
        'description': 'Case Western Reserve bearing fault classification',
        'TYPOLOGY_AXIS_THRESHOLDS': {
            'frequency': [0.15, 0.30, 0.70, 0.85],  # Emphasize periodicity
            'derivatives': [0.15, 0.30, 0.70, 0.85], # Emphasize spikiness
        },
    },

    'financial': {
        # Financial time series: high inherent volatility
        'description': 'Financial market data',
        'DYNAMICS_STABILITY': {
            'stable_change': 0.15,         # Much looser
            'evolving_change': 0.25,
            'unstable_change': 0.40,
        },
        'DYNAMICS_TRAJECTORY': {
            'convergence_trend': 0.05,     # Strong trends only
            'divergence_trend': -0.05,
            'oscillation_ratio': 0.4,      # More sensitive to oscillation
        },
        'TRANSITION_NUMERIC': {
            'stability': 0.25,
            'predictability': 0.20,
            'coupling': 0.20,
            'memory': 0.15,
        },
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_typology_thresholds(axis: str, preset: str = 'default') -> list:
    """Get thresholds for a specific axis."""
    if axis in TYPOLOGY_AXIS_THRESHOLDS:
        return TYPOLOGY_AXIS_THRESHOLDS[axis]
    return TYPOLOGY_CLASSIFICATION.get(preset, TYPOLOGY_CLASSIFICATION['default'])


def is_meaningful_change(field: str, delta: float) -> bool:
    """Check if a numeric change is meaningful."""
    threshold = TRANSITION_NUMERIC.get(field, 0.15)
    return abs(delta) > threshold


def classify_severity(field: str, delta: float) -> str:
    """Classify change severity."""
    threshold = TRANSITION_NUMERIC.get(field, 0.15)
    ratio = abs(delta) / threshold if threshold > 0 else 0

    if ratio <= 1.0:
        return 'none'
    elif ratio <= TRANSITION_SEVERITY['mild_max']:
        return 'mild'
    elif ratio <= TRANSITION_SEVERITY['moderate_max']:
        return 'moderate'
    else:
        return 'severe'


def get_transition_type(
    field: str,
    from_value: float,
    to_value: float,
) -> str:
    """Determine transition type based on field and values."""
    delta = to_value - from_value

    if field == 'stability':
        if (from_value > 0 and to_value < 0) or (from_value < 0 and to_value > 0):
            return 'bifurcation'

    if field == 'memory':
        if (from_value > 0.5 and to_value < 0.5) or (from_value < 0.5 and to_value > 0.5):
            return 'flip'

    if field in ('predictability', 'coupling'):
        if delta < -2 * TRANSITION_NUMERIC.get(field, 0.15):
            return 'collapse'
        elif delta > 2 * TRANSITION_NUMERIC.get(field, 0.15):
            return 'recovery'

    return 'shift'


def get_breaking_threshold(n_signals: int) -> int:
    """
    Calculate BREAKING threshold scaled by network size.

    Args:
        n_signals: Number of signals in the network

    Returns:
        Number of severe decouplings to trigger BREAKING
    """
    base = GEOMETRY_STABILITY['breaking_severe_base']
    ratio = GEOMETRY_STABILITY['breaking_severe_ratio']

    # Use whichever is larger: base count or ratio of signals
    scaled = max(base, int(n_signals * ratio))
    return scaled


def get_domain_thresholds(domain: str) -> Dict[str, Any]:
    """
    Get merged thresholds for a specific domain.

    Args:
        domain: Domain name (cmapss, femto, hydraulic, tep, cwru, financial)

    Returns:
        Dict with all threshold categories, with domain overrides applied
    """
    # Start with defaults
    result = {
        'TYPOLOGY_CLASSIFICATION': deepcopy(TYPOLOGY_CLASSIFICATION),
        'TYPOLOGY_AXIS_THRESHOLDS': deepcopy(TYPOLOGY_AXIS_THRESHOLDS),
        'TYPOLOGY_TRANSITION': deepcopy(TYPOLOGY_TRANSITION),
        'GEOMETRY_TOPOLOGY': deepcopy(GEOMETRY_TOPOLOGY),
        'GEOMETRY_STABILITY': deepcopy(GEOMETRY_STABILITY),
        'GEOMETRY_LEADERSHIP': deepcopy(GEOMETRY_LEADERSHIP),
        'GEOMETRY_TRANSITION': deepcopy(GEOMETRY_TRANSITION),
        'DYNAMICS_REGIME': deepcopy(DYNAMICS_REGIME),
        'DYNAMICS_STABILITY': deepcopy(DYNAMICS_STABILITY),
        'DYNAMICS_TRAJECTORY': deepcopy(DYNAMICS_TRAJECTORY),
        'DYNAMICS_TRANSITION': deepcopy(DYNAMICS_TRANSITION),
        'TRANSITION_NUMERIC': deepcopy(TRANSITION_NUMERIC),
        'TRANSITION_SEVERITY': deepcopy(TRANSITION_SEVERITY),
        'STATE_TRANSITION': deepcopy(STATE_TRANSITION),
    }

    # Apply domain overrides if they exist
    domain_lower = domain.lower()
    if domain_lower in DOMAIN_OVERRIDES:
        overrides = DOMAIN_OVERRIDES[domain_lower]
        for key, values in overrides.items():
            if key == 'description':
                result['description'] = values
            elif key in result and isinstance(values, dict):
                result[key].update(values)

    result['domain'] = domain_lower
    return result


def list_domains() -> list:
    """List all available domain configurations."""
    return list(DOMAIN_OVERRIDES.keys())


def get_domain_description(domain: str) -> str:
    """Get description for a domain."""
    domain_lower = domain.lower()
    if domain_lower in DOMAIN_OVERRIDES:
        return DOMAIN_OVERRIDES[domain_lower].get('description', 'No description')
    return 'Default configuration'


# =============================================================================
# RUNTIME OVERRIDE SUPPORT (for future Streamlit integration)
# =============================================================================
# TODO: Add Streamlit UI for threshold adjustment
#
# Implementation notes for future:
# 1. Create streamlit/pages/thresholds.py with sliders for each threshold
# 2. Store user overrides in session_state or a user config file
# 3. Call set_runtime_overrides() to apply changes
# 4. Changes take effect immediately (no recomputation needed)
#
# Example future API:
#
#   from prism.config.thresholds import set_runtime_overrides, get_active_thresholds
#
#   # User adjusts slider in Streamlit
#   set_runtime_overrides({
#       'DYNAMICS_STABILITY': {'stable_change': 0.10}
#   })
#
#   # Code automatically uses updated thresholds
#   thresholds = get_active_thresholds()

_runtime_overrides: Dict[str, Any] = {}


def set_runtime_overrides(overrides: Dict[str, Any]) -> None:
    """
    Set runtime threshold overrides (for Streamlit integration).

    Args:
        overrides: Dict of threshold category -> values to override
    """
    global _runtime_overrides
    _runtime_overrides = deepcopy(overrides)


def clear_runtime_overrides() -> None:
    """Clear all runtime overrides, reverting to defaults."""
    global _runtime_overrides
    _runtime_overrides = {}


def get_active_thresholds(domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Get currently active thresholds with all overrides applied.

    Priority: runtime_overrides > domain_overrides > defaults

    Args:
        domain: Optional domain for domain-specific defaults

    Returns:
        Dict with all active thresholds
    """
    # Start with domain or default thresholds
    if domain:
        result = get_domain_thresholds(domain)
    else:
        result = get_domain_thresholds('default')

    # Apply runtime overrides
    for key, values in _runtime_overrides.items():
        if key in result and isinstance(values, dict):
            result[key].update(values)

    return result


# =============================================================================
# SUMMARY TABLE (for reference)
# =============================================================================
"""
Layer               | Parameter                  | Default | Effect
--------------------|----------------------------|---------|----------------------------------
Signal Typology     | axis_threshold             | 0.30    | Change per axis for "moving"
Signal Typology     | distance_threshold         | 0.30    | Profile distance for regime change
Signal Typology     | classification[0]          | 0.25    | Score < this = strong_low
Signal Typology     | classification[3]          | 0.75    | Score > this = strong_high

Struct. Geometry    | highly_connected_density   | 0.70    | Above = HIGHLY_CONNECTED
Struct. Geometry    | modular_silhouette         | 0.40    | Above + clusters = MODULAR
Struct. Geometry    | breaking_severe_base       | 2       | Base count for BREAKING
Struct. Geometry    | breaking_severe_ratio      | 0.10    | Or 10% of signals = BREAKING
Struct. Geometry    | weakening_pairs            | 0.25    | Decoupling rate = WEAKENING

Dynamical Systems   | coupled                    | 0.70    | Correlation = COUPLED
Dynamical Systems   | decoupled                  | 0.30    | Correlation = DECOUPLED
Dynamical Systems   | transitioning              | 0.20    | Change rate = TRANSITIONING
Dynamical Systems   | stable_change              | 0.08    | Below = STABLE
Dynamical Systems   | convergence_trend          | 0.02    | Above = CONVERGING
Dynamical Systems   | critical_change            | 0.30    | Above = CRITICAL

Transitions         | stability_threshold        | 0.20    | Meaningful stability change
Transitions         | memory_threshold           | 0.10    | Meaningful memory change
Transitions         | zscore_threshold           | 3.0     | State divergence spike

Domain Overrides:
    cmapss     - Tighter thresholds, earlier warnings (failures costly)
    femto      - Focus on periodicity and volatility (vibration)
    hydraulic  - More tolerant (multiple subsystems)
    tep        - Looser thresholds (process variability)
    cwru       - Emphasize frequency/spikiness (fault classification)
    financial  - Much looser (inherent market volatility)
"""

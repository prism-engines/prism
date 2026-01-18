"""
PRISM State Layer - The Final Layer of the PRISM Trilogy
=========================================================

Layer 3: STATE - "What is the system DOING?"

Completes the PRISM framework with system-level dynamic state detection,
transition classification, and regime tracking.

The PRISM Trilogy:
    Layer 1 (Vector):   "How does each signal behave?"
    Layer 2 (Geometry): "How do signals relate?"
    Layer 3 (State):    "What is the system DOING?"

Key insight: PRISM measures DYNAMICS. Classification asks "which picture?"
State asks "what just happened?" - transitions manifest as divergence
singularities in the Laplace field.

Modules:
    transition_detector: Detect regime boundaries via divergence spikes
    state_signature:     Extract transition fingerprints for classification
    state_classifier:    Classify transition types (supervised/unsupervised)
    regime_tracker:      Track state over time with early warning

Output: cohort_state.parquet
"""

from .transition_detector import (
    Transition,
    compute_system_divergence,
    detect_transitions,
    find_leading_signals,
    summarize_leading_signals,
)

from .state_signature import (
    StateSignature,
    extract_signature,
    extract_all_signatures,
    signatures_to_features,
)

from .state_classifier import StateClassifier

from .regime_tracker import (
    RegimeState,
    RegimeHistory,
    RegimeTracker,
)

__all__ = [
    # Transition detection
    'Transition',
    'compute_system_divergence',
    'detect_transitions',
    'find_leading_signals',
    'summarize_leading_signals',
    # State signature
    'StateSignature',
    'extract_signature',
    'extract_all_signatures',
    'signatures_to_features',
    # Classification
    'StateClassifier',
    # Regime tracking
    'RegimeState',
    'RegimeHistory',
    'RegimeTracker',
]

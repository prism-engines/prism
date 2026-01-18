"""
PRISM Assessments - Domain-agnostic fault/regime detection
============================================================

Three-layer detection architecture:
    WHAT  - Classification features (volatility, entropy)
    WHEN  - Break detection (structural changes, impulses)
    MODE  - Behavioral trajectory (precursor mode detection)

Main Entry Point:
    python -m prism.assessments.run --domain cheme

Configuration:
    All settings in config/assessment.yaml
    Domain-specific overrides for windows, thresholds, features

Available Modules:
    run              - Main assessment runner (config-driven)
    config           - Configuration loader
    mode_discovery   - Discover behavioral modes from Laplace fingerprints
    tep_integrated   - TEP-specific integrated assessment

Legacy (TEP-specific):
    tep_fault_eval      - Classification baseline
    tep_break_detection - Break detection evaluation
    tep_modes           - Mode computation
    tep_summary         - Quick summary

Usage:
    python -m prism.assessments.run --domain cheme
    python -m prism.assessments.run --domain turbofan
    python -m prism.assessments.run --domain cheme --show-config
"""

from .mode_discovery import run_mode_discovery, get_assessment_output_path
from .config import (
    get_domain_config,
    get_windows,
    get_thresholds,
    get_what_features,
    get_when_features,
    get_mode_features,
    get_signal_patterns,
    get_precursor_mode,
)
from .run import run_assessment

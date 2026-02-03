"""
Statistics & Summary Engines (PR #14)

Four engines for fleet-wide analytics:
1. BaselineEngine - Compute fleet/entity baselines for all metrics
2. AnomalyEngine - Score deviations from baseline (z-scores)
3. FleetEngine - Rankings, clusters, cohort analysis
4. SummaryEngine - Generate executive reports

These engines operate ACROSS entities to provide comparative analytics.
"""

from .baseline_engine import BaselineEngine, BaselineConfig, run_baseline_engine
from .anomaly_engine import AnomalyEngine, AnomalyConfig, run_anomaly_engine
from .fleet_engine import FleetEngine, FleetConfig, run_fleet_engine
from .summary_engine import (
    SummaryEngine, SummaryConfig, run_summary_engine,
    generate_text_report, generate_markdown_report
)

__all__ = [
    # Engines
    'BaselineEngine',
    'AnomalyEngine',
    'FleetEngine',
    'SummaryEngine',
    # Configs
    'BaselineConfig',
    'AnomalyConfig',
    'FleetConfig',
    'SummaryConfig',
    # Runner functions
    'run_baseline_engine',
    'run_anomaly_engine',
    'run_fleet_engine',
    'run_summary_engine',
    # Report generators
    'generate_text_report',
    'generate_markdown_report',
]

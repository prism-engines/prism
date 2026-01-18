"""
PRISM Cohort Definitions

Cohort classification for signals by domain.
Used for memory-efficient batched geometry and categorization.

Hierarchy:
    Domain (data/raw/) → Cohort → Signal
"""

from prism.cohorts.climate import (
    CLIMATE_SUB_COHORTS,  # Legacy alias
    CLIMATE_COHORTS,
    get_climate_subcohort,  # Legacy alias
    get_climate_cohort,
)


def get_cohort(signal_id: str, domain: str) -> str:
    """
    Get cohort for an signal.

    Args:
        signal_id: The signal ID
        domain: Domain name ('climate', etc.)

    Returns:
        Cohort name or 'other' if not classified
    """
    if domain == 'climate':
        return get_climate_cohort(signal_id)
    else:
        return 'other'


# Legacy alias
get_subcohort = get_cohort


__all__ = [
    # New names
    'CLIMATE_COHORTS',
    'get_cohort',
    'get_climate_cohort',
    # Legacy aliases
    'CLIMATE_SUB_COHORTS',
    'get_subcohort',
    'get_climate_subcohort',
]

"""
Climate Cohort Definitions

Categorizes climate signals into cohorts by type and region.
Note: Most climate data is hemispheric or global, not continent-level.

Hierarchy:
    Domain: data/raw/observations.parquet
    └── Cohorts: temperature_global, greenhouse_gases, cryosphere_arctic, etc.
        └── Signals: GISS_TEMP_GLOBAL, CO2_MONTHLY, ARCTIC_SEA_ICE_EXTENT, etc.
"""

from typing import Dict, List


CLIMATE_COHORTS: Dict[str, List[str]] = {
    # Temperature anomalies
    'temperature_global': [
        'GISS_TEMP_GLOBAL',
        'NOAA_TEMP_GLOBAL',
    ],

    'temperature_nh': [
        'GISS_TEMP_NH',
        'NOAA_TEMP_LAND',  # Primarily NH land mass
    ],

    'temperature_sh': [
        'GISS_TEMP_SH',
        'NOAA_TEMP_OCEAN',  # Primarily SH ocean
    ],

    # Greenhouse gases (global measurements from Mauna Loa)
    'greenhouse_gases': [
        'CO2_MONTHLY',
        'CO2_ANNUAL',
        'CO2_GROWTH_RATE',
        'CH4_MONTHLY',
        'N2O_MONTHLY',
        'SF6_MONTHLY',
    ],

    # Teleconnection indices - Northern Hemisphere
    'teleconnection_nh': [
        'NAO_INDEX',  # North Atlantic Oscillation
        'AO_INDEX',   # Arctic Oscillation
        'PNA_INDEX',  # Pacific-North American
    ],

    # Teleconnection indices - Southern Hemisphere
    'teleconnection_sh': [
        'AAO_INDEX',  # Antarctic Oscillation (SAM)
    ],

    # Teleconnection indices - Pacific/Tropical
    'teleconnection_pacific': [
        'SOI_INDEX',   # Southern Oscillation Index
        'SST_NINO34',  # El Nino 3.4 region SST
    ],

    # Cryosphere - Arctic
    'cryosphere_arctic': [
        'ARCTIC_SEA_ICE_EXTENT',
        'ARCTIC_SEA_ICE_AREA',
    ],

    # Cryosphere - Antarctic
    'cryosphere_antarctic': [
        'ANTARCTIC_SEA_ICE_EXTENT',
        'ANTARCTIC_SEA_ICE_AREA',
    ],

    # Sea level
    'sea_level': [
        'SEA_LEVEL_GLOBAL',
    ],

    # Solar forcing
    'solar': [
        'SUNSPOT_NUMBER',
    ],
}


# Legacy alias
CLIMATE_SUB_COHORTS = CLIMATE_COHORTS


# Reverse lookup: signal -> cohort
_INDICATOR_TO_COHORT: Dict[str, str] = {}
for cohort, signals in CLIMATE_COHORTS.items():
    for signal in signals:
        _INDICATOR_TO_COHORT[signal] = cohort


def get_climate_cohort(signal_id: str) -> str:
    """
    Get cohort for a climate signal.

    Args:
        signal_id: The signal ID

    Returns:
        Cohort name or 'other' if not classified
    """
    return _INDICATOR_TO_COHORT.get(signal_id, 'other')


# Legacy alias
get_climate_subcohort = get_climate_cohort


# Full list for domain operations
CLIMATE_ALL = [ind for inds in CLIMATE_COHORTS.values() for ind in inds]


# Regional groupings (for higher-level analysis)
CLIMATE_REGIONS = {
    'global': ['temperature_global', 'greenhouse_gases', 'sea_level', 'solar'],
    'northern_hemisphere': ['temperature_nh', 'teleconnection_nh', 'cryosphere_arctic'],
    'southern_hemisphere': ['temperature_sh', 'teleconnection_sh', 'cryosphere_antarctic'],
    'pacific': ['teleconnection_pacific'],
}

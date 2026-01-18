"""
Chemical Engineering Domain - Cohort Definitions

Process/Equipment Monitoring Datasets:
- NASA C-MAPSS: Turbofan engine degradation (21 sensors × 4 datasets)
- Tennessee Eastman: Chemical process simulation (52 process variables)

These datasets are ideal for testing PRISM's behavioral geometry
against known degradation patterns and equipment failure modes.

Domain: cheme
Cohorts:
  - turbofan: Engine sensors (21 per engine unit)
  - tep: Tennessee Eastman process variables (52 sensors)
"""

from typing import Dict, List


# =============================================================================
# NASA C-MAPSS TURBOFAN DEGRADATION
# =============================================================================
# 21 sensors per engine, measuring temperature, pressure, speed, etc.
# Sensor descriptions from NASA documentation

TURBOFAN_SENSORS: Dict[str, List[str]] = {
    # Primary operating conditions (not sensors but operational settings)
    'operating': [
        'CMAPSS_op1',      # Operational setting 1 (altitude)
        'CMAPSS_op2',      # Operational setting 2 (Mach number)
        'CMAPSS_op3',      # Operational setting 3 (throttle resolver angle)
    ],

    # Temperature sensors
    'temperature': [
        'CMAPSS_T2',       # Total temperature at fan inlet (R)
        'CMAPSS_T24',      # Total temperature at LPC outlet (R)
        'CMAPSS_T30',      # Total temperature at HPC outlet (R)
        'CMAPSS_T50',      # Total temperature at LPT outlet (R)
    ],

    # Pressure sensors
    'pressure': [
        'CMAPSS_P2',       # Pressure at fan inlet (psia)
        'CMAPSS_P15',      # Total pressure in bypass-duct (psia)
        'CMAPSS_P30',      # Total pressure at HPC outlet (psia)
        'CMAPSS_Ps30',     # Static pressure at HPC outlet (psia)
    ],

    # Speed sensors
    'speed': [
        'CMAPSS_Nf',       # Physical fan speed (rpm)
        'CMAPSS_Nc',       # Physical core speed (rpm)
        'CMAPSS_NRf',      # Corrected fan speed (rpm)
        'CMAPSS_NRc',      # Corrected core speed (rpm)
    ],

    # Flow and ratio sensors
    'flow': [
        'CMAPSS_epr',      # Engine pressure ratio (P50/P2)
        'CMAPSS_phi',      # Ratio of fuel flow to Ps30
        'CMAPSS_BPR',      # Bypass ratio
        'CMAPSS_farB',     # Burner fuel-air ratio
        'CMAPSS_W31',      # HPT coolant bleed (lbm/s)
        'CMAPSS_W32',      # LPT coolant bleed (lbm/s)
    ],

    # Efficiency/performance
    'efficiency': [
        'CMAPSS_htBleed',  # Bleed enthalpy
    ],
}

# All turbofan sensor IDs
TURBOFAN_ALL = [
    sensor for group in TURBOFAN_SENSORS.values() for sensor in group
]

# Sensor column mapping (from raw file column index to sensor name)
# Columns 0-1: unit, cycle; 2-4: op settings; 5-25: sensors
TURBOFAN_COLUMN_MAP = {
    5: 'CMAPSS_T2',
    6: 'CMAPSS_T24',
    7: 'CMAPSS_T30',
    8: 'CMAPSS_T50',
    9: 'CMAPSS_P2',
    10: 'CMAPSS_P15',
    11: 'CMAPSS_P30',
    12: 'CMAPSS_Nf',
    13: 'CMAPSS_Nc',
    14: 'CMAPSS_epr',
    15: 'CMAPSS_Ps30',
    16: 'CMAPSS_phi',
    17: 'CMAPSS_NRf',
    18: 'CMAPSS_NRc',
    19: 'CMAPSS_BPR',
    20: 'CMAPSS_farB',
    21: 'CMAPSS_htBleed',
    22: 'CMAPSS_Nf_dmd',   # Demanded fan speed
    23: 'CMAPSS_PCNfR_dmd', # Demanded fan speed (corrected)
    24: 'CMAPSS_W31',
    25: 'CMAPSS_W32',
}

# Operating condition columns
TURBOFAN_OP_MAP = {
    2: 'CMAPSS_op1',
    3: 'CMAPSS_op2',
    4: 'CMAPSS_op3',
}


# =============================================================================
# TENNESSEE EASTMAN PROCESS (TEP)
# =============================================================================
# 52 process variables from the Tennessee Eastman chemical plant simulation
# Variables from Downs & Vogel (1993) benchmark

TEP_SENSORS: Dict[str, List[str]] = {
    # Manipulated Variables (MVs) - 12 control valves
    'manipulated': [
        'TEP_XMV01',       # D feed flow (stream 2)
        'TEP_XMV02',       # E feed flow (stream 3)
        'TEP_XMV03',       # A feed flow (stream 1)
        'TEP_XMV04',       # A and C feed flow (stream 4)
        'TEP_XMV05',       # Compressor recycle valve
        'TEP_XMV06',       # Purge valve (stream 9)
        'TEP_XMV07',       # Separator pot liquid flow (stream 10)
        'TEP_XMV08',       # Stripper liquid product flow (stream 11)
        'TEP_XMV09',       # Stripper steam valve
        'TEP_XMV10',       # Reactor cooling water flow
        'TEP_XMV11',       # Condenser cooling water flow
        'TEP_XMV12',       # Agitator speed
    ],

    # Process Measurements (continuous) - 22 sensors
    'process': [
        'TEP_XMEAS01',     # A feed (stream 1) - kscmh
        'TEP_XMEAS02',     # D feed (stream 2) - kg/hr
        'TEP_XMEAS03',     # E feed (stream 3) - kg/hr
        'TEP_XMEAS04',     # A and C feed (stream 4) - kscmh
        'TEP_XMEAS05',     # Recycle flow (stream 8) - kscmh
        'TEP_XMEAS06',     # Reactor feed rate (stream 6) - kscmh
        'TEP_XMEAS07',     # Reactor pressure - kPa gauge
        'TEP_XMEAS08',     # Reactor level - %
        'TEP_XMEAS09',     # Reactor temperature - C
        'TEP_XMEAS10',     # Purge rate (stream 9) - kscmh
        'TEP_XMEAS11',     # Product separator temperature - C
        'TEP_XMEAS12',     # Product separator level - %
        'TEP_XMEAS13',     # Product separator pressure - kPa gauge
        'TEP_XMEAS14',     # Product separator underflow (stream 10) - m3/hr
        'TEP_XMEAS15',     # Stripper level - %
        'TEP_XMEAS16',     # Stripper pressure - kPa gauge
        'TEP_XMEAS17',     # Stripper underflow (stream 11) - m3/hr
        'TEP_XMEAS18',     # Stripper temperature - C
        'TEP_XMEAS19',     # Stripper steam flow - kg/hr
        'TEP_XMEAS20',     # Compressor work - kW
        'TEP_XMEAS21',     # Reactor cooling water outlet temp - C
        'TEP_XMEAS22',     # Separator cooling water outlet temp - C
    ],

    # Composition measurements (sampled - less frequent)
    'composition': [
        'TEP_XMEAS23',     # Reactor feed analysis - A (mol%)
        'TEP_XMEAS24',     # Reactor feed analysis - B (mol%)
        'TEP_XMEAS25',     # Reactor feed analysis - C (mol%)
        'TEP_XMEAS26',     # Reactor feed analysis - D (mol%)
        'TEP_XMEAS27',     # Reactor feed analysis - E (mol%)
        'TEP_XMEAS28',     # Reactor feed analysis - F (mol%)
        'TEP_XMEAS29',     # Purge gas analysis - A (mol%)
        'TEP_XMEAS30',     # Purge gas analysis - B (mol%)
        'TEP_XMEAS31',     # Purge gas analysis - C (mol%)
        'TEP_XMEAS32',     # Purge gas analysis - D (mol%)
        'TEP_XMEAS33',     # Purge gas analysis - E (mol%)
        'TEP_XMEAS34',     # Purge gas analysis - F (mol%)
        'TEP_XMEAS35',     # Purge gas analysis - G (mol%)
        'TEP_XMEAS36',     # Purge gas analysis - H (mol%)
        'TEP_XMEAS37',     # Product analysis - D (mol%)
        'TEP_XMEAS38',     # Product analysis - E (mol%)
        'TEP_XMEAS39',     # Product analysis - F (mol%)
        'TEP_XMEAS40',     # Product analysis - G (mol%)
        'TEP_XMEAS41',     # Product analysis - H (mol%)
    ],
}

# All TEP sensor IDs
TEP_ALL = [
    sensor for group in TEP_SENSORS.values() for sensor in group
]

# TEP column mapping (from CSV column index to sensor name)
# First columns are typically: timestamp, fault_code
TEP_COLUMN_MAP = {i + 2: f'TEP_XMEAS{i+1:02d}' for i in range(41)}
TEP_COLUMN_MAP.update({i + 43: f'TEP_XMV{i+1:02d}' for i in range(12)})


# =============================================================================
# CONSOLIDATED CHEME COHORTS
# =============================================================================

CHEME_COHORTS: Dict[str, List[str]] = {
    # Turbofan sensors (all 21 + 3 operating conditions)
    'turbofan': TURBOFAN_ALL,

    # Tennessee Eastman (all 52 process variables + 12 manipulated)
    'tep': TEP_ALL,
}

# All ChemE signals
CHEME_ALL = TURBOFAN_ALL + TEP_ALL


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================

# Reverse lookup: signal -> cohort
_INDICATOR_TO_COHORT: Dict[str, str] = {}
for cohort, signals in CHEME_COHORTS.items():
    for signal in signals:
        _INDICATOR_TO_COHORT[signal] = cohort


def get_cheme_cohort(signal_id: str) -> str:
    """
    Get cohort for a ChemE signal.

    Args:
        signal_id: The signal ID (e.g., 'CMAPSS_T2', 'TEP_XMEAS01')

    Returns:
        Cohort name ('turbofan', 'tep') or 'other' if not classified
    """
    return _INDICATOR_TO_COHORT.get(signal_id, 'other')


def get_signal_source(signal_id: str) -> str:
    """
    Get data source for a ChemE signal.

    Args:
        signal_id: The signal ID

    Returns:
        'cmapss', 'tep', or 'unknown'
    """
    if signal_id.startswith('CMAPSS_'):
        return 'cmapss'
    elif signal_id.startswith('TEP_'):
        return 'tep'
    else:
        return 'unknown'

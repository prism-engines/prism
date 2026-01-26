"""
Engine Constant Validation
==========================

Simple validation: constants must EXIST or engine returns NaN.
No silent fallbacks. No sneaky defaults.

Usage:
    from prism.engines.validation import require_constants, MissingConstantError

    # In engine:
    missing = require_constants(config, ['mass', 'spring_constant'])
    if missing:
        return {'error': f'Missing constants: {missing}', **nan_result}
"""

import math
from typing import Dict, Any, List, Set, Optional


class MissingConstantError(Exception):
    """Raised when required physical constant is not provided."""
    pass


# =============================================================================
# REQUIRED CONSTANTS BY ENGINE
# =============================================================================

REQUIRED_CONSTANTS: Dict[str, List[str]] = {
    # PHYSICS
    'kinetic_energy': ['mass'],
    'potential_energy': ['mass', 'spring_constant'],
    'momentum': ['mass'],
    'hamiltonian': ['mass', 'spring_constant'],
    'lagrangian': ['mass', 'spring_constant'],
    'work_energy': ['mass'],

    # FIELDS
    'navier_stokes': ['nu', 'rho', 'dx', 'dy', 'dz'],

    # THERMODYNAMICS
    'gibbs_free_energy': ['temperature'],  # needs T, plus (H,S) or P
    'thermodynamics': ['Cp'],

    # FLUID
    'reynolds': ['nu', 'rho'],
    'pressure_drop': ['nu', 'rho', 'diameter'],
    'fluid_mechanics': ['nu', 'rho'],

    # HEAT TRANSFER
    'fourier': ['thermal_conductivity'],
    'heat_transfer': ['thermal_conductivity'],

    # MASS TRANSFER
    'fick': ['diffusivity'],
    'mass_transfer': ['diffusivity'],

    # ELECTROCHEMISTRY
    'faraday': ['n_electrons', 'molar_mass'],
    'nernst': ['n_electrons', 'temperature'],
    'echem_kinetics': ['n_electrons', 'temperature'],
    'echem_mass_transfer': ['diffusivity', 'nu'],

    # CHEMICAL
    'reaction_kinetics': ['rate_constant'],
    'cstr_kinetics': ['rate_constant', 'volume'],

    # PHASE EQUILIBRIA
    'vle': ['temperature', 'pressure'],
    'lle': ['temperature'],
    'flash': ['temperature', 'pressure'],

    # SEPARATIONS
    'distillation': ['relative_volatility'],
    'absorption': ['equilibrium_constant'],
    'membrane': ['permeability'],

    # BALANCES
    'energy_balance': ['Cp'],
    'material_balance': [],  # just needs flows

    # BATTERY
    'battery': ['capacity', 'voltage_nominal'],
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def require_constants(
    config: Dict[str, Any],
    required: List[str],
    engine_name: str = None,
) -> List[str]:
    """
    Check if required constants exist in config.

    Args:
        config: Config dict (should have 'constants' or 'global_constants' key)
        required: List of required constant names
        engine_name: Optional engine name for error messages

    Returns:
        List of missing constant names (empty if all present)
    """
    # Get constants from config
    constants = config.get('constants', config.get('global_constants', {}))

    # Also check top-level config for constants
    if not constants:
        constants = {k: v for k, v in config.items()
                    if isinstance(v, (int, float)) and not k.startswith('_')}

    missing = [c for c in required if c not in constants or constants[c] is None]

    return missing


def validate_engine(
    engine_name: str,
    config: Dict[str, Any],
    strict: bool = True,
) -> Optional[List[str]]:
    """
    Validate config has required constants for engine.

    Args:
        engine_name: Name of engine to validate
        config: Config dict
        strict: If True, raise error; if False, return missing list

    Returns:
        None if valid, list of missing constants if not strict

    Raises:
        MissingConstantError: If strict=True and constants missing
    """
    required = REQUIRED_CONSTANTS.get(engine_name, [])
    if not required:
        return None

    missing = require_constants(config, required, engine_name)

    if missing:
        if strict:
            raise MissingConstantError(
                f"Engine '{engine_name}' requires constants: {required}. "
                f"Missing: {missing}. "
                f"Add to config.json under 'global_constants' or 'constants'."
            )
        return missing

    return None


def get_constant(
    config: Dict[str, Any],
    name: str,
    default: float = float('nan'),
) -> float:
    """
    Get constant from config, return NaN if missing.

    Args:
        config: Config dict
        name: Constant name
        default: Default value if missing (default: NaN)

    Returns:
        Constant value or default
    """
    constants = config.get('constants', config.get('global_constants', {}))

    if name in constants and constants[name] is not None:
        return float(constants[name])

    # Check top-level
    if name in config and config[name] is not None:
        return float(config[name])

    return default


def nan_result(keys: List[str]) -> Dict[str, float]:
    """
    Create a result dict with NaN values for all keys.

    Args:
        keys: List of result keys

    Returns:
        Dict with all keys set to NaN
    """
    return {k: float('nan') for k in keys}


# =============================================================================
# UNIT VALIDATION
# =============================================================================

EXPECTED_UNITS: Dict[str, Dict[str, str]] = {
    # Physics constants
    'mass': 'kg',
    'spring_constant': 'N/m',
    'velocity': 'm/s',
    'position': 'm',
    'force': 'N',
    'energy': 'J',

    # Fluid properties
    'nu': 'm²/s',
    'rho': 'kg/m³',
    'mu': 'Pa·s',
    'diameter': 'm',

    # Thermal properties
    'thermal_conductivity': 'W/(m·K)',
    'Cp': 'J/(kg·K)',
    'temperature': 'K',

    # Mass transfer
    'diffusivity': 'm²/s',
    'concentration': 'mol/m³',

    # Electrochemistry
    'n_electrons': 'dimensionless',
    'molar_mass': 'kg/mol',

    # Grid spacing
    'dx': 'm',
    'dy': 'm',
    'dz': 'm',
    'dt': 's',
}


def check_units(
    config: Dict[str, Any],
    required: List[str],
) -> Dict[str, str]:
    """
    Check if units are specified for constants.

    Returns dict of constant -> expected unit for documentation.
    """
    units = config.get('units', {})

    result = {}
    for const in required:
        expected = EXPECTED_UNITS.get(const, 'unknown')
        provided = units.get(const, 'not specified')
        result[const] = f"expected: {expected}, provided: {provided}"

    return result


# =============================================================================
# HELPER FOR ENGINE IMPLEMENTATION
# =============================================================================

def validate_or_nan(
    config: Dict[str, Any],
    required: List[str],
    result_keys: List[str],
    engine_name: str = None,
) -> Optional[Dict[str, float]]:
    """
    Validate constants exist, return NaN result if missing.

    Use at start of engine compute function:

        nan = validate_or_nan(config, ['mass'], ['kinetic_energy', 'momentum'])
        if nan:
            return nan

    Args:
        config: Config dict
        required: Required constant names
        result_keys: Keys to include in NaN result
        engine_name: Optional engine name for error field

    Returns:
        None if valid, NaN result dict if constants missing
    """
    missing = require_constants(config, required)

    if missing:
        result = nan_result(result_keys)
        result['error'] = f"Missing constants: {missing}"
        if engine_name:
            result['engine'] = engine_name
        return result

    return None

"""
Ghia et al. (1982) Benchmark Data for Lid-Driven Cavity Flow

The definitive benchmark for incompressible Navier-Stokes solvers.
Over 10,000 citations. Everyone validates against this.

Reference:
    Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
    "High-Re solutions for incompressible flow using the
    Navier-Stokes equations and a multigrid method."
    Journal of Computational Physics, 48(3), 387-411.

Usage:
    from domains.fluid.benchmark_ghia import GHIA_U, validate_against_ghia
    
    # Get reference data
    y, u_ref = GHIA_U[100]  # Re=100
    
    # Validate your solution
    error = validate_against_ghia(u_computed, y_computed, Re=100)
"""

import numpy as np
from typing import Dict, Tuple, List

# =============================================================================
# TABLE I: u-velocity along vertical centerline (x=0.5)
# =============================================================================

# y-coordinates for u-velocity sampling
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
    0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
    0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
    0.0547, 0.0000
])

# u-velocity values at each Re
GHIA_U_VALUES = {
    100: np.array([
        1.00000,  0.84123,  0.78871,  0.73722,  0.68717,
        0.23151,  0.00332, -0.13641, -0.20581, -0.21090,
       -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
       -0.03717,  0.00000
    ]),
    400: np.array([
        1.00000,  0.75837,  0.68439,  0.61756,  0.55892,
        0.29093,  0.16256,  0.02135, -0.11477, -0.17119,
       -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
       -0.08186,  0.00000
    ]),
    1000: np.array([
        1.00000,  0.65928,  0.57492,  0.51117,  0.46604,
        0.33304,  0.18719,  0.05702, -0.06080, -0.10648,
       -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
       -0.18109,  0.00000
    ]),
    3200: np.array([
        1.00000,  0.53236,  0.48296,  0.46547,  0.46101,
        0.34682,  0.19791,  0.07156, -0.04272, -0.08664,
       -0.24427, -0.34323, -0.41933, -0.37827, -0.35344,
       -0.32407,  0.00000
    ]),
    5000: np.array([
        1.00000,  0.48223,  0.46120,  0.45992,  0.46036,
        0.33556,  0.20087,  0.08183, -0.03039, -0.07404,
       -0.22855, -0.33050, -0.40435, -0.43643, -0.42901,
       -0.41165,  0.00000
    ]),
    7500: np.array([
        1.00000,  0.47244,  0.47048,  0.47323,  0.47167,
        0.34228,  0.20591,  0.08342, -0.03800, -0.07503,
       -0.23176, -0.32393, -0.38324, -0.43025, -0.43590,
       -0.43154,  0.00000
    ]),
    10000: np.array([
        1.00000,  0.47221,  0.47783,  0.48070,  0.47804,
        0.34635,  0.20673,  0.08344,  0.03111, -0.07540,
       -0.23186, -0.32709, -0.38000, -0.41657, -0.42537,
       -0.42735,  0.00000
    ]),
}

# =============================================================================
# TABLE II: v-velocity along horizontal centerline (y=0.5)
# =============================================================================

GHIA_X = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
    0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
    0.2266, 0.1563, 0.0938, 0.0781, 0.0703,
    0.0625, 0.0000
])

GHIA_V_VALUES = {
    100: np.array([
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
       -0.16914, -0.22445, -0.24533,  0.05454,  0.17527,
        0.17507,  0.16077,  0.12317,  0.10890,  0.10091,
        0.09233,  0.00000
    ]),
    400: np.array([
        0.00000, -0.12146, -0.15663, -0.19254, -0.22847,
       -0.23827, -0.44993, -0.38598,  0.05186,  0.30174,
        0.30203,  0.28124,  0.22965,  0.20920,  0.19713,
        0.18360,  0.00000
    ]),
    1000: np.array([
        0.00000, -0.21388, -0.27669, -0.33714, -0.39188,
       -0.51550, -0.42665, -0.31966,  0.02526,  0.32235,
        0.33075,  0.37095,  0.32627,  0.30353,  0.29012,
        0.27485,  0.00000
    ]),
    3200: np.array([
        0.00000, -0.39017, -0.47425, -0.52357, -0.54053,
       -0.44307, -0.37401, -0.31184,  0.00999,  0.28188,
        0.29030,  0.37119,  0.42768,  0.41906,  0.40917,
        0.39560,  0.00000
    ]),
    5000: np.array([
        0.00000, -0.42735, -0.51060, -0.55216, -0.55674,
       -0.41496, -0.36737, -0.30719,  0.00945,  0.27280,
        0.28066,  0.35368,  0.42951,  0.43648,  0.43329,
        0.42447,  0.00000
    ]),
}

# =============================================================================
# Convenient dictionary format
# =============================================================================

def get_u_profile(Re: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get (y, u) profile for u-velocity along vertical centerline."""
    if Re not in GHIA_U_VALUES:
        raise ValueError(f"Re={Re} not available. Choose from {list(GHIA_U_VALUES.keys())}")
    return GHIA_Y.copy(), GHIA_U_VALUES[Re].copy()


def get_v_profile(Re: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get (x, v) profile for v-velocity along horizontal centerline."""
    if Re not in GHIA_V_VALUES:
        raise ValueError(f"Re={Re} not available. Choose from {list(GHIA_V_VALUES.keys())}")
    return GHIA_X.copy(), GHIA_V_VALUES[Re].copy()


# =============================================================================
# Primary Vortex Centers (Table III in Ghia)
# =============================================================================

GHIA_PRIMARY_VORTEX = {
    # Re: (x_center, y_center, psi_min)
    100:   (0.6172, 0.7344, -0.10342),
    400:   (0.5547, 0.6055, -0.11391),
    1000:  (0.5313, 0.5625, -0.11792),
    3200:  (0.5165, 0.5469, -0.12040),
    5000:  (0.5117, 0.5352, -0.11897),
    7500:  (0.5117, 0.5322, -0.11765),
    10000: (0.5117, 0.5333, -0.11969),
}


# =============================================================================
# Validation Function
# =============================================================================

def validate_against_ghia(
    u_computed: np.ndarray,
    y_computed: np.ndarray,
    Re: int,
    tolerance: float = 0.05,
) -> Dict:
    """
    Validate computed u-velocity profile against Ghia benchmark.
    
    Args:
        u_computed: Computed u-velocity values
        y_computed: y-coordinates of computed values
        Re: Reynolds number
        tolerance: Acceptable relative error
        
    Returns:
        Dict with error metrics and pass/fail
    """
    y_ref, u_ref = get_u_profile(Re)
    
    # Interpolate computed to reference y-locations
    u_interp = np.interp(y_ref, y_computed, u_computed)
    
    # Compute errors
    abs_error = np.abs(u_interp - u_ref)
    max_abs_error = float(np.max(abs_error))
    
    # Relative error (avoiding division by zero)
    rel_error = abs_error / (np.abs(u_ref) + 1e-10)
    max_rel_error = float(np.max(rel_error[np.abs(u_ref) > 0.01]))
    
    # L2 norm
    l2_error = float(np.sqrt(np.mean(abs_error**2)))
    
    # Pass/fail
    passed = max_abs_error < tolerance
    
    return {
        'Re': Re,
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'l2_error': l2_error,
        'tolerance': tolerance,
        'passed': passed,
        'y_ref': y_ref,
        'u_ref': u_ref,
        'u_computed': u_interp,
    }


def available_reynolds() -> List[int]:
    """List available Reynolds numbers."""
    return list(GHIA_U_VALUES.keys())


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ghia Benchmark Data — Self Test")
    print("=" * 60)
    
    print("\nAvailable Reynolds numbers:", available_reynolds())
    
    print("\n--- u-velocity at Re=100 along y-centerline ---")
    y, u = get_u_profile(100)
    print(f"{'y':>8} {'u':>10}")
    print("-" * 20)
    for yi, ui in zip(y, u):
        print(f"{yi:8.4f} {ui:10.5f}")
    
    print("\n--- Primary vortex centers ---")
    print(f"{'Re':>6} {'x_c':>8} {'y_c':>8} {'ψ_min':>10}")
    print("-" * 35)
    for Re, (xc, yc, psi) in GHIA_PRIMARY_VORTEX.items():
        print(f"{Re:6d} {xc:8.4f} {yc:8.4f} {psi:10.5f}")
    
    print("\n--- Validation example ---")
    # Perfect match
    y_test, u_test = get_u_profile(100)
    result = validate_against_ghia(u_test, y_test, Re=100)
    print(f"Perfect match: L2 error = {result['l2_error']:.2e}, passed = {result['passed']}")
    
    # With noise
    u_noisy = u_test + np.random.normal(0, 0.02, len(u_test))
    result = validate_against_ghia(u_noisy, y_test, Re=100)
    print(f"With 2% noise: L2 error = {result['l2_error']:.2e}, passed = {result['passed']}")
    
    print("\n" + "=" * 60)

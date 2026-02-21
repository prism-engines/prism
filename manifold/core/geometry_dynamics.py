"""
ENGINES Geometry Dynamics Engine

The complete differential geometry framework.
Computes derivatives and curvature for the geometry evolution over windows.

"You have position (cohort_vector).
 You have shape (eigenvalues).
 Now here are the derivatives."

Computes:
- First derivatives (velocity/tangent)
- Second derivatives (acceleration/curvature)
- Third derivatives (jerk/torsion)
- Collapse detection (computed index, not interpretation)
- Phase space analysis

NOTE: ENGINES computes, never classifies. All classification logic
has been removed. Prime interprets the computed values.

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to primitives.
All mathematical operations are performed by directly-imported primitive functions.

INPUT:
- cohort_geometry.parquet (eigenvalues per window)
- signal_geometry.parquet (signal positions per window)

OUTPUT:
- geometry_dynamics.parquet (system-level dynamics)
- signal_dynamics.parquet (per-signal dynamics)

Credit: The emeritus PhD mathematicians who would accept nothing less.
"""

import numpy as np
import polars as pl
from typing import Dict, Optional, Any

# Import primitives for all mathematical computation
from manifold.core._pmtvs import first_derivative, second_derivative, jerk, attractor_reconstruction

# Import configuration
from manifold.config import get_config


# ============================================================
# CONFIGURATION-DRIVEN DEFAULTS
# ============================================================

def _get_dynamics_config() -> Dict[str, Any]:
    """Get dynamics computation config."""
    config = get_config()
    return {
        'dt': config.get('dynamics.derivatives.dt', 1.0),
        'smooth_window': config.get('dynamics.derivatives.smooth_window', 3),
        'method': config.get('dynamics.derivatives.method', 'central'),
    }


def _get_collapse_config() -> Dict[str, Any]:
    """Get collapse detection config."""
    config = get_config()
    return {
        'threshold_velocity': config.get('dynamics.collapse.threshold_velocity', -0.1),
        'sustained_fraction': config.get('dynamics.collapse.sustained_fraction', 0.3),
        'min_collapse_length': config.get('dynamics.collapse.min_collapse_length', 5),
    }


def _get_phase_space_config() -> Dict[str, Any]:
    """Get phase space config."""
    config = get_config()
    return {
        'embedding_dim': config.get('dynamics.phase_space.embedding_dim', 2),
        'tau': config.get('dynamics.phase_space.tau', 1),
    }


# ============================================================
# DERIVATIVE COMPUTATION
# ============================================================

def compute_derivatives(
    x: np.ndarray,
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None,
    coords: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives up to third order with optional smoothing.

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Args:
        x: Ordered series values
        dt: Index step scalar (from config if not provided). Ignored if coords given.
        smooth_window: Smoothing window for noise reduction (from config if not provided)
        coords: Coordinate array (e.g. signal_0_end values). When provided,
                handles non-uniform spacing via np.gradient(x, coords). On uniform
                time axis this produces identical results to scalar dt.

    Returns:
        Dict with velocity, acceleration, jerk, curvature, speed
    """
    # Get config values
    dynamics_config = _get_dynamics_config()
    if dt is None:
        dt = dynamics_config['dt']
    if smooth_window is None:
        smooth_window = dynamics_config['smooth_window']

    n = len(x)

    if n < 3:
        return {
            'velocity': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'curvature': np.zeros(n),
            'speed': np.zeros(n),
        }

    # Optional smoothing (preprocessing, not math)
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        x_smooth = np.convolve(x, kernel, mode='same')
    else:
        x_smooth = x

    # ─────────────────────────────────────────────────
    # DERIVATIVES — coordinate-aware
    # When coords is provided, np.gradient handles non-uniform spacing.
    # When coords is None, fall back to scalar dt (original behavior).
    # ─────────────────────────────────────────────────

    if coords is not None:
        dx = np.gradient(x_smooth, coords)
        d2x = np.gradient(dx, coords)
        d3x = np.gradient(d2x, coords)
    else:
        dx = first_derivative(x_smooth, dt=dt, method='central')
        d2x = second_derivative(x_smooth, dt=dt, method='central')
        d3x = jerk(x_smooth, dt=dt)

    # ─────────────────────────────────────────────────
    # 1D CURVATURE: κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
    # For 1D series, we compute curvature directly
    # ─────────────────────────────────────────────────
    denom = (1 + dx**2)**1.5
    curvature = np.where(denom > 1e-10, np.abs(d2x) / denom, 0)

    # Speed (magnitude of velocity)
    speed = np.abs(dx)

    return {
        'velocity': dx,
        'acceleration': d2x,
        'jerk': d3x,
        'curvature': curvature,
        'speed': speed,
    }


def compute_phase_space(
    x: np.ndarray,
    embedding_dim: Optional[int] = None,
    tau: Optional[int] = None
) -> np.ndarray:
    """
    Reconstruct phase space using time-delay embedding.

    ARCHITECTURE: Pure orchestration - delegates to ENGINES primitive.

    Takens' theorem: The attractor can be reconstructed from
    a single ordered series using delay coordinates.

    Args:
        x: Ordered series
        embedding_dim: Number of dimensions (from config if not provided)
        tau: Time delay (from config if not provided)

    Returns:
        Phase space coordinates (n_points × embedding_dim)
    """
    # Get config values
    phase_config = _get_phase_space_config()
    if embedding_dim is None:
        embedding_dim = phase_config['embedding_dim']
    if tau is None:
        tau = phase_config['tau']
    # ─────────────────────────────────────────────────
    # ATTRACTOR RECONSTRUCTION → ENGINES PRIMITIVE
    # ─────────────────────────────────────────────────
    return attractor_reconstruction(x, embed_dim=embedding_dim, tau=tau)


# ============================================================
# COLLAPSE DETECTION
# ============================================================

def detect_collapse(
    effective_dim: np.ndarray,
    threshold_velocity: Optional[float] = None,
    sustained_fraction: Optional[float] = None,
    min_collapse_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect dimensional collapse in effective_dim series.

    Collapse = sustained negative velocity in effective_dim
    indicating the system is losing degrees of freedom.

    Args:
        effective_dim: Effective dimension per window
        threshold_velocity: Velocity below this = collapsing (from config if not provided)
        sustained_fraction: Fraction of points that must be collapsing (from config if not provided)
        min_collapse_length: Minimum consecutive points for collapse (from config if not provided)

    Returns:
        Collapse detection results
    """
    # Get config values
    collapse_config = _get_collapse_config()
    if threshold_velocity is None:
        threshold_velocity = collapse_config['threshold_velocity']
    if sustained_fraction is None:
        sustained_fraction = collapse_config['sustained_fraction']
    if min_collapse_length is None:
        min_collapse_length = collapse_config['min_collapse_length']

    n = len(effective_dim)

    # Return computed values only - no boolean classification
    # collapse_onset_idx = -1 means no collapse detected (sentinel)
    if n < min_collapse_length:
        return {
            'collapse_onset_idx': -1,
            'collapse_onset_fraction': -1.0,
        }

    # Compute derivatives
    deriv = compute_derivatives(effective_dim)
    velocity = deriv['velocity']

    # Identify collapsing regions
    collapsing = velocity < threshold_velocity

    # Find sustained collapse (consecutive points)
    collapse_runs = []
    run_start = None

    for i in range(n):
        if collapsing[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_collapse_length:
                    collapse_runs.append((run_start, i, run_length))
                run_start = None

    # Handle run that extends to end
    if run_start is not None:
        run_length = n - run_start
        if run_length >= min_collapse_length:
            collapse_runs.append((run_start, n, run_length))

    if not collapse_runs:
        return {
            'collapse_onset_idx': -1,
            'collapse_onset_fraction': -1.0,
        }

    # Take the longest collapse run
    longest_run = max(collapse_runs, key=lambda x: x[2])
    onset_idx, end_idx, duration = longest_run

    # Return computed index/fraction only - no boolean
    return {
        'collapse_onset_idx': int(onset_idx),
        'collapse_onset_fraction': onset_idx / n,
    }


# ============================================================
# GEOMETRY DYNAMICS COMPUTATION
# ============================================================

def compute_geometry_dynamics(
    cohort_geometry: pl.DataFrame,
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute dynamics of geometry evolution.

    For each (unit_id, engine), computes derivatives of:
    - effective_dim
    - eigenvalues
    - total_variance

    Args:
        cohort_geometry: State geometry DataFrame
        dt: Index step between consecutive windows (from config if not provided)
        smooth_window: Smoothing window (from config if not provided)
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
    # Get config values
    dynamics_config = _get_dynamics_config()
    if dt is None:
        dt = dynamics_config['dt']
    if smooth_window is None:
        smooth_window = dynamics_config['smooth_window']

    if verbose:
        print("=" * 70)
        print("GEOMETRY DYNAMICS ENGINE")
        print("Differential geometry of state evolution")
        print("=" * 70)
        print(f"Input: {len(cohort_geometry)} rows")
        print(f"Columns: {cohort_geometry.columns}")

    # Guard: empty input — return without writing an invalid 0-column parquet
    if len(cohort_geometry) == 0 or 'engine' not in cohort_geometry.columns:
        if verbose:
            print("  No state geometry data to process (empty input)")
        return pl.DataFrame()

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in cohort_geometry.columns
    group_cols = ['cohort', 'engine'] if has_cohort else ['engine']

    # Process each (cohort, engine) or just engine
    results = []
    groups = cohort_geometry.group_by(group_cols, maintain_order=True)

    for group_key, group in groups:
        if has_cohort:
            if isinstance(group_key, tuple):
                cohort, engine = group_key
            else:
                cohort, engine = None, group_key
        else:
            cohort = None
            engine = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
        # Sort by signal_0_end
        group = group.sort('signal_0_end')

        s0_values = group['signal_0_end'].to_numpy()
        n = len(s0_values)

        if n < 3:
            continue

        # Extract series
        effective_dim = group['effective_dim'].to_numpy()
        eigenvalue_1 = group['eigenvalue_1'].to_numpy()
        total_variance = group['total_variance'].to_numpy()

        # Compute derivatives with actual signal_0_end spacing
        # On time axis: s0_values = [0, 24, 48, ...] → uniform → same as scalar dt
        # On reaxis: s0_values = [390.17, 390.33, 392.0, ...] → correct non-uniform
        coords = s0_values.astype(float)
        eff_dim_deriv = compute_derivatives(effective_dim, dt, smooth_window, coords=coords)
        eigen_1_deriv = compute_derivatives(eigenvalue_1, dt, smooth_window, coords=coords)
        variance_deriv = compute_derivatives(total_variance, dt, smooth_window, coords=coords)

        # Detect collapse (computed values only, no classification)
        collapse = detect_collapse(effective_dim)

        if verbose and collapse['collapse_onset_idx'] == -1:
            vel = eff_dim_deriv['velocity']
            min_v = float(np.min(vel))
            cfg = _get_collapse_config()
            below = vel < cfg['threshold_velocity']
            if below.any():
                # Count longest run below threshold for verbose output
                changes = np.concatenate(([below[0]], below[:-1] != below[1:], [True]))
                run_lengths = np.diff(np.where(changes)[0])
                below_runs = run_lengths[::2] if below[0] else run_lengths[1::2]
                max_run = int(max(below_runs)) if len(below_runs) > 0 else 0
            else:
                max_run = 0
            print(f"    {engine}: no collapse (min_vel={min_v:.3f}, longest_run={max_run}, need={cfg['min_collapse_length']})")

        # Build result rows - computed values only, NO classification
        for i in range(n):
            row = {
                'signal_0_end': float(s0_values[i]),
                'engine': engine,

                # Effective dimension dynamics
                'effective_dim': effective_dim[i],
                'effective_dim_velocity': eff_dim_deriv['velocity'][i],
                'effective_dim_acceleration': eff_dim_deriv['acceleration'][i],
                'effective_dim_jerk': eff_dim_deriv['jerk'][i],
                'effective_dim_curvature': eff_dim_deriv['curvature'][i],

                # Eigenvalue dynamics
                'eigenvalue_1': eigenvalue_1[i],
                'eigenvalue_1_velocity': eigen_1_deriv['velocity'][i],

                # Variance dynamics
                'total_variance': total_variance[i],
                'variance_velocity': variance_deriv['velocity'][i],

                # Collapse detection (computed index/fraction, not interpretation)
                'collapse_onset_idx': collapse['collapse_onset_idx'],
                'collapse_onset_fraction': collapse['collapse_onset_fraction'],
            }
            # Include cohort if available
            if cohort:
                row['cohort'] = cohort
            if unit_id:
                row['unit_id'] = unit_id
            results.append(row)

    # Build DataFrame
    result = pl.DataFrame(results)

    return result


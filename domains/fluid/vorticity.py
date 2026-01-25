"""
Vorticity — Rotation in a Velocity Field

Vorticity measures the local spinning motion of a fluid. In 2D:
    ω = ∂v/∂x - ∂u/∂y

Physical meaning:
    - ω > 0: Counter-clockwise rotation
    - ω < 0: Clockwise rotation
    - ω = 0: Irrotational flow (potential flow)

In 3D, vorticity is a vector:
    ω = ∇ × u = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)

For lid-driven cavity:
    - High vorticity at the moving lid
    - Vortex cores in corner eddies
    - Maximum |ω| at Re=100 is ~3.0 (normalized)

References:
    - Batchelor, "An Introduction to Fluid Dynamics"
    - Ghia et al. (1982) for cavity vorticity contours
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VorticityResult:
    """Result of vorticity calculation"""
    vorticity: np.ndarray          # ω field (2D array or scalar)
    max_vorticity: float           # Maximum |ω|
    min_vorticity: float           # Minimum ω (most negative)
    mean_vorticity: float          # Should be ~0 for closed domain
    circulation: Optional[float]   # ∮ u·dl if boundary provided
    vortex_centers: List[Tuple[float, float]]  # (x, y) of vortex cores
    confidence: float
    warnings: List[str]


def compute_2d(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    method: str = "central",
    **kwargs
) -> VorticityResult:
    """
    Compute 2D vorticity from velocity field.
    
    Args:
        u: x-velocity field, shape (ny, nx)
        v: y-velocity field, shape (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        method: Differentiation method ("central", "forward", "backward")
        
    Returns:
        VorticityResult with vorticity field and statistics
    """
    warnings = []
    confidence = 1.0
    
    # Validate inputs
    if u.shape != v.shape:
        warnings.append(f"Shape mismatch: u={u.shape}, v={v.shape}")
        confidence = 0.0
        return VorticityResult(
            vorticity=np.array([]),
            max_vorticity=np.nan,
            min_vorticity=np.nan,
            mean_vorticity=np.nan,
            circulation=None,
            vortex_centers=[],
            confidence=0.0,
            warnings=warnings,
        )
    
    ny, nx = u.shape
    
    if nx < 3 or ny < 3:
        warnings.append("Grid too small for derivative calculation")
        confidence *= 0.5
    
    # Compute derivatives
    if method == "central":
        # Central differences (2nd order accurate)
        # ∂v/∂x
        dvdx = np.zeros_like(v)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx      # Forward at left
        dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx   # Backward at right
        
        # ∂u/∂y
        dudy = np.zeros_like(u)
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dudy[0, :] = (u[1, :] - u[0, :]) / dy      # Forward at bottom
        dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy   # Backward at top
        
    elif method == "numpy":
        # Use numpy gradient (handles boundaries)
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        
    else:
        warnings.append(f"Unknown method: {method}, using central")
        return compute_2d(u, v, dx, dy, method="central")
    
    # Vorticity: ω = ∂v/∂x - ∂u/∂y
    omega = dvdx - dudy
    
    # Statistics
    max_vort = float(np.max(omega))
    min_vort = float(np.min(omega))
    mean_vort = float(np.mean(omega))
    
    # Mean should be ~0 for closed domain (Stokes theorem)
    if abs(mean_vort) > 0.1 * max(abs(max_vort), abs(min_vort)):
        warnings.append(f"Non-zero mean vorticity ({mean_vort:.3f}) - check boundaries")
        confidence *= 0.8
    
    # Find vortex centers (local extrema of |ω|)
    vortex_centers = find_vortex_centers(omega, dx, dy)
    
    return VorticityResult(
        vorticity=omega,
        max_vorticity=max_vort,
        min_vorticity=min_vort,
        mean_vorticity=mean_vort,
        circulation=None,  # Would need boundary integral
        vortex_centers=vortex_centers,
        confidence=confidence,
        warnings=warnings,
    )


def find_vortex_centers(
    omega: np.ndarray,
    dx: float,
    dy: float,
    threshold: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Find approximate vortex center locations.
    
    Uses local extrema of vorticity magnitude.
    """
    from scipy import ndimage
    
    ny, nx = omega.shape
    centers = []
    
    # Find local maxima of |ω|
    omega_abs = np.abs(omega)
    max_val = np.max(omega_abs)
    
    if max_val < 1e-10:
        return []
    
    # Threshold to find significant vortices
    mask = omega_abs > threshold * max_val
    
    # Label connected regions
    try:
        labeled, num_features = ndimage.label(mask)
        
        for i in range(1, num_features + 1):
            region = labeled == i
            # Find centroid weighted by vorticity magnitude
            y_coords, x_coords = np.where(region)
            weights = omega_abs[region]
            
            if np.sum(weights) > 0:
                x_center = np.average(x_coords, weights=weights) * dx
                y_center = np.average(y_coords, weights=weights) * dy
                centers.append((x_center, y_center))
    except ImportError:
        # scipy not available, skip vortex detection
        pass
    
    return centers


def compute_from_points(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> VorticityResult:
    """
    Compute vorticity from scattered point data.
    
    Interpolates to regular grid first.
    """
    from scipy.interpolate import griddata
    
    # Create regular grid
    nx = ny = 50  # Default resolution
    x_grid = np.linspace(x.min(), x.max(), nx)
    y_grid = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocities
    points = np.column_stack([x.ravel(), y.ravel()])
    u_grid = griddata(points, u.ravel(), (X, Y), method='cubic')
    v_grid = griddata(points, v.ravel(), (X, Y), method='cubic')
    
    # Handle NaN at boundaries
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    v_grid = np.nan_to_num(v_grid, nan=0.0)
    
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    
    return compute_2d(u_grid, v_grid, dx, dy)


# Convenience function for enstrophy (vorticity squared, useful for turbulence)
def enstrophy(omega: np.ndarray) -> float:
    """
    Compute enstrophy = ½∫ω² dA
    
    Enstrophy is the "intensity" of vorticity.
    In 2D turbulence, enstrophy cascades to small scales.
    """
    return 0.5 * np.mean(omega ** 2)


ENGINE_META = {
    'name': 'vorticity',
    'capability': 'VORTICITY',
    'description': 'Local rotation rate in velocity field',
    'requires_signals': ['u', 'v'],
    'requires_grid': True,
    'output_unit': '1/s',
}


# =============================================================================
# SELF-TEST with Lid-Driven Cavity
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Vorticity — Self Test with Lid-Driven Cavity")
    print("=" * 60)
    
    # Create a simple lid-driven cavity approximation
    # Real solution would come from CFD/PINN
    
    nx, ny = 32, 32
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Approximate velocity field (not accurate, just for testing)
    # Primary vortex rotating clockwise
    r = np.sqrt((X - 0.5)**2 + (Y - 0.6)**2)
    theta = np.arctan2(Y - 0.6, X - 0.5)
    
    # Vortex velocity profile (Rankine vortex)
    r_core = 0.2
    v_theta = np.where(r < r_core, r / r_core, r_core / r)
    
    u_vortex = -v_theta * np.sin(theta)
    v_vortex = v_theta * np.cos(theta)
    
    # Add lid velocity influence
    u = u_vortex * (1 - Y**3) + Y**3  # Blend to u=1 at top
    v = v_vortex * (1 - Y**3)
    
    # Apply no-slip boundaries
    u[0, :] = 0   # Bottom
    u[:, 0] = 0   # Left
    u[:, -1] = 0  # Right
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    u[-1, :] = 1  # Lid velocity
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    print(f"\nGrid: {nx}×{ny}, dx={dx:.4f}, dy={dy:.4f}")
    
    result = compute_2d(u, v, dx, dy)
    
    print(f"\nVorticity Statistics:")
    print(f"  Max ω:  {result.max_vorticity:+.3f}")
    print(f"  Min ω:  {result.min_vorticity:+.3f}")
    print(f"  Mean ω: {result.mean_vorticity:+.3f}")
    print(f"  Enstrophy: {enstrophy(result.vorticity):.3f}")
    
    print(f"\nVortex Centers Found: {len(result.vortex_centers)}")
    for i, (xc, yc) in enumerate(result.vortex_centers):
        print(f"  Vortex {i+1}: ({xc:.3f}, {yc:.3f})")
    
    print(f"\nConfidence: {result.confidence:.0%}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Compare with expected lid-cavity behavior
    print("\n--- Comparison with Ghia Re=100 ---")
    print("Expected: Primary vortex near (0.62, 0.73)")
    print("Expected: Max vorticity at lid corners")
    print("Expected: ω < 0 in primary vortex (clockwise)")
    
    print("\n" + "=" * 60)

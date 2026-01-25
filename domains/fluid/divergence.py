"""
Divergence — Mass Conservation Check

For incompressible flow, the continuity equation requires:
    ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

Non-zero divergence indicates:
    - Mass is being created/destroyed (unphysical)
    - Numerical errors in the solution
    - The flow is actually compressible

This is a CRITICAL diagnostic:
    - |∇·u| < 1e-6: Excellent (spectral methods)
    - |∇·u| < 1e-4: Good (high-order FEM/FVM)
    - |∇·u| < 1e-2: Acceptable (standard CFD)
    - |∇·u| > 0.1: Problematic - check your solver

For PINN solutions, divergence error is often part of the loss function,
so checking it validates that the physics constraint was learned.

References:
    - Any CFD textbook on incompressible flow
    - PINN papers check this as "continuity loss"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class DivergenceResult:
    """Result of divergence calculation"""
    divergence: np.ndarray         # ∇·u field
    max_divergence: float          # Maximum |∇·u|
    mean_divergence: float         # Should be ~0
    rms_divergence: float          # RMS value (L2 norm-ish)
    continuity_satisfied: bool     # Is |∇·u| < tolerance?
    confidence: float
    warnings: List[str]


def compute_2d(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    tolerance: float = 1e-3,
    **kwargs
) -> DivergenceResult:
    """
    Compute 2D divergence (continuity check).
    
    Args:
        u: x-velocity field, shape (ny, nx)
        v: y-velocity field, shape (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        tolerance: Acceptable divergence magnitude
        
    Returns:
        DivergenceResult with divergence field and diagnostics
    """
    warnings = []
    confidence = 1.0
    
    # Validate inputs
    if u.shape != v.shape:
        warnings.append(f"Shape mismatch: u={u.shape}, v={v.shape}")
        return DivergenceResult(
            divergence=np.array([]),
            max_divergence=np.nan,
            mean_divergence=np.nan,
            rms_divergence=np.nan,
            continuity_satisfied=False,
            confidence=0.0,
            warnings=warnings,
        )
    
    ny, nx = u.shape
    
    # Compute derivatives using central differences
    # ∂u/∂x
    dudx = np.zeros_like(u)
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    
    # ∂v/∂y
    dvdy = np.zeros_like(v)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdy[0, :] = (v[1, :] - v[0, :]) / dy
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    
    # Divergence: ∇·u = ∂u/∂x + ∂v/∂y
    div = dudx + dvdy
    
    # Statistics
    max_div = float(np.max(np.abs(div)))
    mean_div = float(np.mean(div))
    rms_div = float(np.sqrt(np.mean(div**2)))
    
    # Check continuity
    continuity_ok = max_div < tolerance
    
    if not continuity_ok:
        warnings.append(f"Continuity violated: max|∇·u| = {max_div:.2e} > {tolerance:.2e}")
        confidence *= 0.5
    
    # Classify quality
    if max_div < 1e-6:
        pass  # Excellent
    elif max_div < 1e-4:
        pass  # Good
    elif max_div < 1e-2:
        warnings.append(f"Divergence {max_div:.2e} is marginal - check grid resolution")
    else:
        warnings.append(f"High divergence {max_div:.2e} - solution may be unreliable")
        confidence *= 0.3
    
    return DivergenceResult(
        divergence=div,
        max_divergence=max_div,
        mean_divergence=mean_div,
        rms_divergence=rms_div,
        continuity_satisfied=continuity_ok,
        confidence=confidence,
        warnings=warnings,
    )


def compute_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    tolerance: float = 1e-3,
) -> DivergenceResult:
    """
    Compute 3D divergence.
    
    Args:
        u, v, w: Velocity components, shape (nz, ny, nx)
        dx, dy, dz: Grid spacings
        tolerance: Acceptable divergence magnitude
    """
    warnings = []
    
    # Use numpy gradient for simplicity
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=1)
    dwdz = np.gradient(w, dz, axis=0)
    
    div = dudx + dvdy + dwdz
    
    max_div = float(np.max(np.abs(div)))
    mean_div = float(np.mean(div))
    rms_div = float(np.sqrt(np.mean(div**2)))
    continuity_ok = max_div < tolerance
    
    if not continuity_ok:
        warnings.append(f"Continuity violated: max|∇·u| = {max_div:.2e}")
    
    return DivergenceResult(
        divergence=div,
        max_divergence=max_div,
        mean_divergence=mean_div,
        rms_divergence=rms_div,
        continuity_satisfied=continuity_ok,
        confidence=1.0 if continuity_ok else 0.5,
        warnings=warnings,
    )


ENGINE_META = {
    'name': 'divergence',
    'capability': 'DIVERGENCE',
    'description': 'Check mass conservation (∇·u = 0 for incompressible)',
    'requires_signals': ['u', 'v'],
    'requires_grid': True,
    'output_unit': '1/s',
}


if __name__ == "__main__":
    print("=" * 60)
    print("Divergence — Self Test")
    print("=" * 60)
    
    # Test 1: Divergence-free field (rotating vortex)
    print("\n--- Test 1: Divergence-free vortex ---")
    nx, ny = 32, 32
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Stream function: ψ = exp(-(x²+y²))
    # u = ∂ψ/∂y = -2y·exp(-(x²+y²))
    # v = -∂ψ/∂x = 2x·exp(-(x²+y²))
    # This satisfies ∂u/∂x + ∂v/∂y = 0 exactly
    
    psi = np.exp(-(X**2 + Y**2))
    u = -2 * Y * psi
    v = 2 * X * psi
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    result = compute_2d(u, v, dx, dy)
    print(f"  Max |∇·u|: {result.max_divergence:.2e}")
    print(f"  RMS |∇·u|: {result.rms_divergence:.2e}")
    print(f"  Continuity satisfied: {result.continuity_satisfied}")
    print(f"  (Should be ~1e-3 due to finite differences)")
    
    # Test 2: Non-divergence-free (source)
    print("\n--- Test 2: Field with source (not divergence-free) ---")
    u_source = X  # ∂u/∂x = 1
    v_source = Y  # ∂v/∂y = 1
    # ∇·u = 2 everywhere
    
    result = compute_2d(u_source, v_source, dx, dy)
    print(f"  Max |∇·u|: {result.max_divergence:.2e}")
    print(f"  Mean ∇·u:  {result.mean_divergence:.2f}")
    print(f"  Continuity satisfied: {result.continuity_satisfied}")
    print(f"  Warnings: {result.warnings}")
    
    print("\n" + "=" * 60)

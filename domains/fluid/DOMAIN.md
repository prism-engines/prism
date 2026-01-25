# Fluid Dynamics Domain

Physics engines for incompressible viscous flows, validated against
the classic Ghia et al. (1982) lid-driven cavity benchmark.

## Description

This domain covers:
- **Lid-driven cavity flow** (classic CFD benchmark)
- **Pipe/channel flow** (Poiseuille, Couette)
- **Flow around obstacles** (cylinder, airfoils)
- **General 2D/3D velocity fields**

Level 4 analysis: requires spatial velocity field data u(x,y,t), v(x,y,t).

## Required Signals

For spatial field analysis (Level 4):

| Pattern | Unit Category | Description |
|---------|---------------|-------------|
| `u`, `u_velocity`, `vx` | velocity | x-component of velocity |
| `v`, `v_velocity`, `vy` | velocity | y-component of velocity |
| `x`, `x_coord` | length | x-position in field |
| `y`, `y_coord` | length | y-position in field |

Optional:
| Pattern | Unit Category | Description |
|---------|---------------|-------------|
| `w`, `w_velocity`, `vz` | velocity | z-component (3D) |
| `p`, `pressure` | pressure | Pressure field |
| `t`, `time` | time | Time (for unsteady flows) |

## Required Constants

| Name | Unit | Default | Description |
|------|------|---------|-------------|
| `rho` | kg/m³ | 1000 | Fluid density |
| `nu` | m²/s | 1e-6 | Kinematic viscosity |
| `mu` | Pa·s | None | Dynamic viscosity (alternative) |
| `U_ref` | m/s | 1.0 | Reference velocity (for Re) |
| `L_ref` | m | 1.0 | Reference length (for Re) |

## Capabilities Provided

| Capability | Engine | Output | Description |
|------------|--------|--------|-------------|
| `VORTICITY` | vorticity.py | ω (1/s) | ω = ∂v/∂x - ∂u/∂y |
| `STRAIN_RATE` | strain_rate.py | S (1/s) | Strain rate tensor |
| `DIVERGENCE` | divergence.py | ∇·v | Continuity check |
| `Q_CRITERION` | q_criterion.py | Q | Vortex identification |
| `TURBULENT_KE` | tke.py | k (m²/s²) | k = ½(u'² + v'² + w'²) |
| `ENERGY_SPECTRUM` | energy_spectrum.py | E(k) | Kolmogorov cascade |
| `REYNOLDS` | reynolds.py | Re | Reynolds number |

## Benchmark Data: Ghia et al. (1982)

The lid-driven cavity is THE classic CFD benchmark. A square cavity
with the top lid moving at velocity U=1. Validated data for Re=100-10000.

### Geometry
```
        u=1, v=0 (moving lid)
    ┌─────────────────────┐
    │                     │
u=0 │                     │ u=0
v=0 │                     │ v=0
    │                     │
    └─────────────────────┘
        u=0, v=0 (no-slip)
```

### Reference: u-velocity along vertical centerline (x=0.5)

```
# y       Re=100    Re=400    Re=1000   Re=3200   Re=5000
1.0000    1.00000   1.00000   1.00000   1.00000   1.00000
0.9766    0.84123   0.75837   0.65928   0.53236   0.48223
0.9688    0.78871   0.68439   0.57492   0.48296   0.46120
0.9609    0.73722   0.61756   0.51117   0.46547   0.45992
0.9531    0.68717   0.55892   0.46604   0.46101   0.46036
0.8516    0.23151   0.29093   0.33304   0.34682   0.33556
0.7344    0.00332   0.16256   0.18719   0.19791   0.20087
0.6172   -0.13641   0.02135   0.05702   0.07156   0.08183
0.5000   -0.20581  -0.11477  -0.06080  -0.04272  -0.03039
0.4531   -0.21090  -0.17119  -0.10648  -0.08664  -0.07404
0.2813   -0.15662  -0.32726  -0.27805  -0.24427  -0.22855
0.1719   -0.10150  -0.24299  -0.38289  -0.34323  -0.33050
0.1016   -0.06434  -0.14612  -0.29730  -0.41933  -0.40435
0.0703   -0.04775  -0.10338  -0.22220  -0.37827  -0.43643
0.0625   -0.04192  -0.09266  -0.20196  -0.35344  -0.42901
0.0547   -0.03717  -0.08186  -0.18109  -0.32407  -0.41165
0.0000    0.00000   0.00000   0.00000   0.00000   0.00000
```

### Key features by Reynolds number:

| Re | Flow character | Primary vortex center | Secondary vortices |
|----|----------------|----------------------|-------------------|
| 100 | Laminar, symmetric | (0.6172, 0.7344) | Weak corners |
| 400 | Laminar | (0.5547, 0.6055) | Visible corners |
| 1000 | Laminar, stretched | (0.5313, 0.5625) | Strong BR, weak BL |
| 5000 | Near-turbulent | (0.5117, 0.5352) | All corners active |
| 10000 | Turbulent transition | (0.5117, 0.5333) | Tertiary vortices |

BR = Bottom-Right, BL = Bottom-Left

## Equations

### Navier-Stokes (incompressible)
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u

Continuity: ∇·u = 0
```

### Vorticity (2D)
```
ω = ∂v/∂x - ∂u/∂y
```

### Stream function (2D, satisfies continuity automatically)
```
u = ∂ψ/∂y,  v = -∂ψ/∂x
```

### Q-criterion (vortex identification)
```
Q = ½(||Ω||² - ||S||²)

where Ω = antisymmetric part of ∇u (rotation)
      S = symmetric part of ∇u (strain)

Q > 0 indicates vortex core
```

### Turbulent Kinetic Energy
```
k = ½(u'² + v'² + w'²)

where u' = u - ū (fluctuation from mean)
```

## Validated Against

| Test Case | Re | Expected | Notes |
|-----------|-----|----------|-------|
| Lid cavity | 100 | u_min ≈ -0.21 | At y≈0.45 |
| Lid cavity | 1000 | u_min ≈ -0.38 | At y≈0.17 |
| Poiseuille | any | u_max = 1.5×u_avg | Parabolic profile |
| Couette | any | Linear profile | u = y×U_wall |

## References

1. Ghia, U., Ghia, K.N., & Shin, C.T. (1982). "High-Re solutions for
   incompressible flow using the Navier-Stokes equations and a
   multigrid method." J. Computational Physics, 48(3), 387-411.
   **THE benchmark paper - 10,000+ citations**

2. Botella, O., & Peyret, R. (1998). "Benchmark spectral results on
   the lid-driven cavity flow." Computers & Fluids, 27(4), 421-433.

3. Erturk, E., Corke, T.C., & Gökçöl, C. (2005). "Numerical solutions
   of 2-D steady incompressible driven cavity flow at high Reynolds
   numbers." Int. J. Numer. Meth. Fluids, 48, 747-774.

## Contributors

- PRISM Team (initial implementation)
- Benchmark data from Ghia et al. (1982)

#!/usr/bin/env python
"""
Dynamical Systems Data Generator
================================

Generates 4 deterministic dynamical systems for PRISM validation:
1. Lorenz System - chaotic attractor with two lobes
2. Rössler Attractor - single folded band chaos
3. Double Pendulum - Lagrangian chaos
4. Lotka-Volterra - predator-prey oscillations

Each system creates a separate domain with observations.parquet.

Usage:
    python scripts/dynamical_systems.py

Then run PRISM pipelines:
    python -m prism.entry_points.signal_vector --signal --domain lorenz --report signal
    python -m prism.entry_points.signal_vector --signal --domain rossler --report signal
    python -m prism.entry_points.signal_vector --signal --domain pendulum --report signal
    python -m prism.entry_points.signal_vector --signal --domain lotka_volterra --report signal
"""

import numpy as np
import polars as pl
from datetime import date, timedelta
from scipy.integrate import solve_ivp
from pathlib import Path

from prism.db.parquet_store import get_data_root
from prism.db.polars_io import write_parquet_atomic


def create_observations(df: pl.DataFrame, signals: list, domain: str) -> pl.DataFrame:
    """Convert trajectory to PRISM observations format."""
    base_date = date(2020, 1, 1)
    n_points = len(df)

    rows = []
    for i in range(n_points):
        obs_date = base_date + timedelta(days=i)
        for ind in signals:
            rows.append({
                'signal_id': f'{domain}_{ind}',
                'obs_date': obs_date,
                'value': float(df[ind][i])
            })

    return pl.DataFrame(rows)


def save_domain(domain: str, trajectory_df: pl.DataFrame, obs_df: pl.DataFrame):
    """Save trajectory and observations to domain directory."""
    data_root = get_data_root(domain)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / 'raw').mkdir(exist_ok=True)
    (data_root / 'vector').mkdir(exist_ok=True)
    (data_root / 'geometry').mkdir(exist_ok=True)

    # Save trajectory
    traj_path = data_root / 'raw' / f'{domain}_trajectory.parquet'
    write_parquet_atomic(trajectory_df, traj_path)

    # Save observations
    obs_path = data_root / 'raw' / 'observations.parquet'
    write_parquet_atomic(obs_df, obs_path)

    return obs_path


def generate_lorenz():
    """Lorenz System: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz"""
    print("\n[1/4] LORENZ SYSTEM")
    print("    Equations: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz")
    print("    Parameters: σ=10, ρ=28, β=8/3")

    def lorenz(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(lorenz, [0, 50], [1.0, 1.0, 1.0],
                    t_eval=np.linspace(0, 50, 5000), method='RK45')

    trajectory_df = pl.DataFrame({
        't': sol.t, 'x': sol.y[0], 'y': sol.y[1], 'z': sol.y[2]
    })

    obs_df = create_observations(trajectory_df, ['x', 'y', 'z'], 'lorenz')
    path = save_domain('lorenz', trajectory_df, obs_df)

    # Ground truth
    lobe_transitions = np.sum(np.abs(np.diff(np.sign(sol.y[0]))) > 0)

    print(f"    Points: {len(trajectory_df):,}")
    print(f"    Lobe transitions: {lobe_transitions}")
    print(f"    Saved: {path}")

    return {'domain': 'lorenz', 'points': len(trajectory_df), 'lobe_transitions': int(lobe_transitions)}


def generate_rossler():
    """Rössler Attractor: dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)"""
    print("\n[2/4] RÖSSLER ATTRACTOR")
    print("    Equations: dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)")
    print("    Parameters: a=0.2, b=0.2, c=5.7")

    def rossler(t, state, a=0.2, b=0.2, c=5.7):
        x, y, z = state
        return [-y - z, x + a * y, b + z * (x - c)]

    sol = solve_ivp(rossler, [0, 100], [0.1, 0.0, 0.1],
                    t_eval=np.linspace(0, 100, 10000), method='RK45')

    trajectory_df = pl.DataFrame({
        't': sol.t, 'x': sol.y[0], 'y': sol.y[1], 'z': sol.y[2]
    })

    obs_df = create_observations(trajectory_df, ['x', 'y', 'z'], 'rossler')
    path = save_domain('rossler', trajectory_df, obs_df)

    # Ground truth - z-spikes
    z_threshold = np.percentile(sol.y[2], 90)
    n_spikes = np.sum(np.diff((sol.y[2] > z_threshold).astype(int)) == 1)

    print(f"    Points: {len(trajectory_df):,}")
    print(f"    Z-spikes (90th pct): {n_spikes}")
    print(f"    Saved: {path}")

    return {'domain': 'rossler', 'points': len(trajectory_df), 'z_spikes': int(n_spikes)}


def generate_double_pendulum():
    """Double Pendulum: Lagrangian chaotic dynamics"""
    print("\n[3/4] DOUBLE PENDULUM")
    print("    Lagrangian dynamics with m1=m2=1, L1=L2=1, g=9.81")
    print("    Initial: θ1=θ2=π/2, ω1=ω2=0")

    def double_pendulum(t, state, m1=1, m2=1, L1=1, L2=1, g=9.81):
        th1, w1, th2, w2 = state
        delta = th2 - th1

        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        d_w1 = (m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(th2) * np.cos(delta) +
                m2 * L2 * w2**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(th1)) / den1

        den2 = (L2 / L1) * den1
        d_w2 = (-m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
                (m1 + m2) * g * np.sin(th1) * np.cos(delta) -
                (m1 + m2) * L1 * w1**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(th2)) / den2

        return [w1, d_w1, w2, d_w2]

    sol = solve_ivp(double_pendulum, [0, 20], [np.pi/2, 0, np.pi/2, 0],
                    t_eval=np.linspace(0, 20, 2000), method='RK45')

    trajectory_df = pl.DataFrame({
        't': sol.t,
        'theta1': sol.y[0], 'omega1': sol.y[1],
        'theta2': sol.y[2], 'omega2': sol.y[3]
    })

    obs_df = create_observations(trajectory_df, ['theta1', 'omega1', 'theta2', 'omega2'], 'pendulum')
    path = save_domain('pendulum', trajectory_df, obs_df)

    # Ground truth - energy (should be conserved)
    m1, m2, L1, L2, g = 1, 1, 1, 1, 9.81
    # Kinetic + Potential energy
    T = 0.5 * m1 * (L1 * sol.y[1])**2 + \
        0.5 * m2 * ((L1 * sol.y[1])**2 + (L2 * sol.y[3])**2 +
                    2 * L1 * L2 * sol.y[1] * sol.y[3] * np.cos(sol.y[0] - sol.y[2]))
    V = -(m1 + m2) * g * L1 * np.cos(sol.y[0]) - m2 * g * L2 * np.cos(sol.y[2])
    E = T + V
    energy_drift = (E[-1] - E[0]) / E[0] * 100

    print(f"    Points: {len(trajectory_df):,}")
    print(f"    Energy drift: {energy_drift:.2f}%")
    print(f"    Saved: {path}")

    return {'domain': 'pendulum', 'points': len(trajectory_df), 'energy_drift_pct': float(energy_drift)}


def generate_lotka_volterra():
    """Lotka-Volterra: dx/dt = αx - βxy, dy/dt = δxy - γy"""
    print("\n[4/4] LOTKA-VOLTERRA (Predator-Prey)")
    print("    Equations: dx/dt = αx - βxy, dy/dt = δxy - γy")
    print("    Parameters: α=1.1, β=0.4, δ=0.1, γ=0.4")

    def lotka_volterra(t, state, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    sol = solve_ivp(lotka_volterra, [0, 100], [10, 5],
                    t_eval=np.linspace(0, 100, 2000), method='RK45')

    trajectory_df = pl.DataFrame({
        't': sol.t, 'prey': sol.y[0], 'predator': sol.y[1]
    })

    obs_df = create_observations(trajectory_df, ['prey', 'predator'], 'lotka')
    path = save_domain('lotka_volterra', trajectory_df, obs_df)

    # Ground truth - oscillation period
    prey_peaks = np.where((sol.y[0][1:-1] > sol.y[0][:-2]) &
                          (sol.y[0][1:-1] > sol.y[0][2:]))[0]
    if len(prey_peaks) > 1:
        period = np.mean(np.diff(sol.t[prey_peaks + 1]))
    else:
        period = 0

    print(f"    Points: {len(trajectory_df):,}")
    print(f"    Oscillation period: {period:.2f} time units")
    print(f"    Saved: {path}")

    return {'domain': 'lotka_volterra', 'points': len(trajectory_df), 'period': float(period)}


def main():
    print("=" * 70)
    print("DYNAMICAL SYSTEMS DATA GENERATOR")
    print("=" * 70)

    results = []
    results.append(generate_lorenz())
    results.append(generate_rossler())
    results.append(generate_double_pendulum())
    results.append(generate_lotka_volterra())

    print("\n" + "=" * 70)
    print("COMPLETE - 4 DOMAINS CREATED")
    print("=" * 70)

    print("\nSummary:")
    for r in results:
        print(f"  {r['domain']}: {r['points']:,} points")

    print("\nRun PRISM pipelines:")
    for r in results:
        domain = r['domain']
        print(f"  python -m prism.entry_points.signal_vector --signal --domain {domain} --force --report signal")


if __name__ == '__main__':
    main()

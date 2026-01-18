#!/usr/bin/env python
"""
Lorenz System Data Generator
============================

Generates Lorenz attractor observations for PRISM validation.

Lorenz Equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Parameters: σ=10, ρ=28, β=8/3

Usage:
    python scripts/lorenz_validation.py

Output:
    data/lorenz/raw/observations.parquet
    data/lorenz/raw/lorenz_trajectory.parquet

Then run PRISM:
    python -m prism.entry_points.signal_vector --signal --domain lorenz --report signal
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import date, timedelta
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

from prism.db.parquet_store import get_data_root, ensure_directories
from prism.db.polars_io import write_parquet_atomic


def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    """Lorenz system differential equations."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def generate_lorenz_data(t_span=(0, 100), dt=0.01, initial_state=(1.0, 1.0, 1.0)):
    """Generate Lorenz system trajectory."""
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        lorenz_system,
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45'
    )

    # Detect lobe (regime): left wing (x < 0) vs right wing (x > 0)
    lobe = np.where(sol.y[0] > 0, 'right', 'left')
    transitions = np.abs(np.diff(np.where(sol.y[0] > 0, 1, 0)))
    is_transition = np.zeros(len(sol.t), dtype=bool)
    is_transition[1:] = transitions.astype(bool)

    return pl.DataFrame({
        'time': sol.t,
        'x': sol.y[0],
        'y': sol.y[1],
        'z': sol.y[2],
        'lobe': lobe,
        'is_transition': is_transition
    })


def compute_ground_truth(lorenz_df: pl.DataFrame) -> dict:
    """Compute ground truth statistics."""
    n_transitions = lorenz_df.filter(pl.col('is_transition')).height

    # Compute dwell times
    lobe_runs = []
    current_lobe = lorenz_df['lobe'][0]
    run_start = 0

    for i, lobe in enumerate(lorenz_df['lobe']):
        if lobe != current_lobe:
            lobe_runs.append({
                'lobe': current_lobe,
                'duration': lorenz_df['time'][i] - lorenz_df['time'][run_start]
            })
            current_lobe = lobe
            run_start = i

    runs_df = pl.DataFrame(lobe_runs) if lobe_runs else pl.DataFrame({'lobe': [], 'duration': []})
    left_runs = runs_df.filter(pl.col('lobe') == 'left')['duration'] if len(runs_df) > 0 else []
    right_runs = runs_df.filter(pl.col('lobe') == 'right')['duration'] if len(runs_df) > 0 else []

    return {
        'n_transitions': n_transitions,
        'mean_left_dwell': left_runs.mean() if len(left_runs) > 0 else 0,
        'mean_right_dwell': right_runs.mean() if len(right_runs) > 0 else 0,
    }


def create_prism_observations(lorenz_df: pl.DataFrame) -> pl.DataFrame:
    """Convert Lorenz data to PRISM observation format."""
    base_date = date(2020, 1, 1)
    n_points = len(lorenz_df)

    rows = []
    for i, row in enumerate(lorenz_df.iter_rows(named=True)):
        obs_date = base_date + timedelta(days=i)
        rows.append({'signal_id': 'lorenz_x', 'obs_date': obs_date, 'value': row['x']})
        rows.append({'signal_id': 'lorenz_y', 'obs_date': obs_date, 'value': row['y']})
        rows.append({'signal_id': 'lorenz_z', 'obs_date': obs_date, 'value': row['z']})
        rows.append({'signal_id': 'lorenz_lobe', 'obs_date': obs_date,
                     'value': 1.0 if row['lobe'] == 'right' else 0.0})

    return pl.DataFrame(rows)


def main():
    print("=" * 70)
    print("LORENZ SYSTEM DATA GENERATOR")
    print("=" * 70)

    # Generate data
    print("\n--- Generating Lorenz trajectory ---")
    lorenz_df = generate_lorenz_data(t_span=(0, 100), dt=0.01)
    print(f"Points: {len(lorenz_df):,}")
    print(f"Time span: {lorenz_df['time'].min():.1f} to {lorenz_df['time'].max():.1f}")

    # Ground truth
    ground_truth = compute_ground_truth(lorenz_df)
    print(f"\nGround truth:")
    print(f"  Lobe transitions: {ground_truth['n_transitions']}")
    print(f"  Mean left dwell: {ground_truth['mean_left_dwell']:.2f}")
    print(f"  Mean right dwell: {ground_truth['mean_right_dwell']:.2f}")

    # Create observations
    print("\n--- Creating PRISM observations ---")
    obs_df = create_prism_observations(lorenz_df)
    print(f"Observations: {len(obs_df):,}")
    print(f"Signals: {obs_df['signal_id'].unique().to_list()}")

    # Save to lorenz domain
    data_root = get_data_root('lorenz')
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / 'raw').mkdir(exist_ok=True)
    (data_root / 'vector').mkdir(exist_ok=True)

    obs_path = data_root / 'raw' / 'observations.parquet'
    write_parquet_atomic(obs_df, obs_path)
    print(f"\nSaved: {obs_path}")

    traj_path = data_root / 'raw' / 'lorenz_trajectory.parquet'
    write_parquet_atomic(lorenz_df, traj_path)
    print(f"Saved: {traj_path}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nNext: Run PRISM pipeline:")
    print("  python -m prism.entry_points.signal_vector --signal --domain lorenz --report signal")


if __name__ == '__main__':
    main()

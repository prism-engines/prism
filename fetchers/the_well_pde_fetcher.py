#!/usr/bin/env python3
"""
The Well PDE Dataset Fetcher for PRISM
======================================

Downloads and processes multiple PDE simulation datasets from PolymathicAI's "The Well"
for PRISM validation.

Available Datasets:
    - euler_multi_quadrants_periodicBC: Compressible gas with shocks
    - planetswe: Shallow water equations
    - MHD_64: Magnetohydrodynamic turbulence (64³)

Data Source:
    GitHub: https://github.com/PolymathicAI/the_well
    Paper: NeurIPS 2024 Datasets and Benchmarks Track

Usage:
    python fetchers/the_well_pde_fetcher.py --dataset euler --trajectories 12
    python fetchers/the_well_pde_fetcher.py --dataset planetswe --trajectories 12
    python fetchers/the_well_pde_fetcher.py --dataset mhd --trajectories 12
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta


# Dataset configurations
DATASET_CONFIGS = {
    "euler": {
        "well_name": "euler_multi_quadrants_openBC",
        "description": "Euler equations - compressible gas with shocks",
        "physics": "Inviscid compressible flow with open boundary conditions",
        "fields": ["density", "velocity_x", "velocity_y", "pressure"],
        "regime_field": None,  # No predefined regimes
    },
    "planetswe": {
        "well_name": "planetswe",
        "description": "Shallow water equations - planetary fluid waves",
        "physics": "Depth-integrated incompressible flow on spherical geometry",
        "fields": ["height", "velocity_u", "velocity_v"],
        "regime_field": None,
    },
    "mhd": {
        "well_name": "MHD_64",
        "description": "MHD turbulence - magnetohydrodynamics",
        "physics": "Isothermal MHD without self-gravity (64³ grid)",
        "fields": ["density", "velocity", "magnetic_field"],
        "regime_field": None,
    },
    "rayleigh_taylor": {
        "well_name": "rayleigh_taylor_instability",
        "description": "Rayleigh-Taylor instability",
        "physics": "Buoyancy-driven fluid instability",
        "fields": ["density", "velocity"],
        "regime_field": None,
    },
    "rayleigh_benard": {
        "well_name": "rayleigh_benard",
        "description": "Rayleigh-Benard convection",
        "physics": "Thermal convection between heated plates",
        "fields": ["temperature", "velocity"],
        "regime_field": None,
    },
    "active_matter": {
        "well_name": "active_matter",
        "description": "Active matter dynamics",
        "physics": "Self-propelled particle systems",
        "fields": ["density", "polarization"],
        "regime_field": None,
    },
}


def fetch_pde_dataset(dataset_key: str, output_dir: Path, n_trajectories: int = 12):
    """
    Fetch a PDE dataset from The Well.
    """
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_key]
    well_name = config["well_name"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    (output_dir / "vector").mkdir(exist_ok=True)

    print(f"Loading {dataset_key} data via the-well package...")
    print(f"Well dataset name: {well_name}")
    print(f"Requesting {n_trajectories} trajectories")

    try:
        from the_well.data import WellDataset

        # Load dataset streaming from HF
        dataset = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/",
            well_dataset_name=well_name,
            well_split_name="train",
        )

        print(f"Dataset loaded. Length: {len(dataset)}")

        # Get first sample to understand structure
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")

        # Determine fields available
        if "input_fields" in sample:
            input_shape = sample["input_fields"].shape
            print(f"Input fields shape: {input_shape}")
        elif "fields" in sample:
            input_shape = sample["fields"].shape
            print(f"Fields shape: {input_shape}")
        else:
            print(f"Available keys: {list(sample.keys())}")
            # Try to find any tensor field
            for key in sample.keys():
                if hasattr(sample[key], 'shape'):
                    print(f"  {key}: shape={sample[key].shape}")

        # Infer trajectory structure
        # Most datasets: n_samples = n_trajectories × n_timesteps
        total_samples = len(dataset)

        # Try to determine from metadata or sample structure
        n_timesteps = 100  # Default guess
        if hasattr(dataset, 'n_trajectories'):
            n_total_traj = dataset.n_trajectories
            n_timesteps = total_samples // n_total_traj
        else:
            # Guess based on total size
            if total_samples > 10000:
                n_total_traj = total_samples // 1000
                n_timesteps = 1000
            else:
                n_total_traj = max(1, total_samples // 100)
                n_timesteps = min(100, total_samples // n_total_traj)

        print(f"Inferred: ~{n_total_traj} trajectories × {n_timesteps} timesteps")

        # Limit trajectories
        n_trajectories = min(n_trajectories, n_total_traj)

        obs_rows = []
        ind_rows = []
        base_date = datetime(2020, 1, 1)

        for traj_idx in range(n_trajectories):
            print(f"  Processing trajectory {traj_idx}...")

            # Collect spatial statistics for this trajectory
            spatial_means = []
            spatial_stds = []
            spatial_mins = []
            spatial_maxs = []

            start_sample = traj_idx * n_timesteps
            end_sample = min(start_sample + n_timesteps, total_samples)

            # Subsample timesteps (every 10th to keep manageable)
            step = max(1, (end_sample - start_sample) // 100)

            for sample_idx in range(start_sample, end_sample, step):
                if sample_idx >= total_samples:
                    break

                sample = dataset[sample_idx]

                # Extract field data
                if "input_fields" in sample:
                    fields = sample["input_fields"].numpy()
                elif "fields" in sample:
                    fields = sample["fields"].numpy()
                else:
                    # Find first tensor
                    for key in sample.keys():
                        if hasattr(sample[key], 'numpy'):
                            fields = sample[key].numpy()
                            break

                # Compute spatial statistics
                spatial_means.append(float(fields.mean()))
                spatial_stds.append(float(fields.std()))
                spatial_mins.append(float(fields.min()))
                spatial_maxs.append(float(fields.max()))

            if len(spatial_means) < 10:
                print(f"    Skipping trajectory {traj_idx} - too few timesteps")
                continue

            # Create signals for different statistics
            for stat_name, stat_values in [
                ("mean", spatial_means),
                ("std", spatial_stds),
                ("range", [mx - mn for mx, mn in zip(spatial_maxs, spatial_mins)]),
            ]:
                signal_id = f"{dataset_key}_{traj_idx}_{stat_name}"

                # Observations
                for t, value in enumerate(stat_values):
                    obs_rows.append({
                        "signal_id": signal_id,
                        "obs_date": base_date + timedelta(seconds=t),
                        "value": value,
                    })

                # Signal metadata
                ind_rows.append({
                    "signal_id": signal_id,
                    "dataset": dataset_key,
                    "well_name": well_name,
                    "trajectory": traj_idx,
                    "statistic": stat_name,
                    "n_points": len(stat_values),
                })

        if not obs_rows:
            print("No valid data extracted!")
            return

        # Save to parquet
        obs_df = pl.DataFrame(obs_rows)
        ind_df = pl.DataFrame(ind_rows)

        obs_df.write_parquet(output_dir / "raw" / "observations.parquet")
        ind_df.write_parquet(output_dir / "raw" / "signals.parquet")

        # Create cohorts
        cohorts = pl.DataFrame([{
            "cohort_id": f"the_well_{dataset_key}",
            "name": config["description"],
            "description": f"The Well: {config['physics']}"
        }])
        cohorts.write_parquet(output_dir / "config" / "cohorts.parquet")

        cohort_members = pl.DataFrame([
            {"cohort_id": f"the_well_{dataset_key}", "signal_id": ind_id}
            for ind_id in ind_df["signal_id"].to_list()
        ])
        cohort_members.write_parquet(output_dir / "config" / "cohort_members.parquet")

        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"Observations: {len(obs_df)}")
        print(f"Signals: {len(ind_df)}")
        print()
        print("Signal summary:")
        print(ind_df.group_by("statistic").len())

    except ImportError as e:
        print(f"Error: the-well package not installed. Run: pip install the-well")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Fetch PDE datasets from The Well")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
        help="Dataset to fetch"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: data/the_well_{dataset})"
    )
    parser.add_argument(
        "--trajectories", "-n",
        type=int,
        default=12,
        help="Number of trajectories to fetch"
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(f"data/the_well_{args.dataset}")

    print("=" * 60)
    print(f"The Well PDE Fetcher: {args.dataset}")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print()

    fetch_pde_dataset(args.dataset, output_dir, args.trajectories)


if __name__ == "__main__":
    main()

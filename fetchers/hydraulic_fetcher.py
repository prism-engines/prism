#!/usr/bin/env python3
"""
UCI Hydraulic Systems Condition Monitoring Fetcher

Downloads the Hydraulic Systems dataset from UCI ML Repository.
Contains sensor data from a hydraulic test rig with various degradation states.

Dataset Info:
- 17 sensors measuring pressure, volume flow, temperature, vibration, power
- 4 components monitored: cooler, valve, pump, accumulator
- Each component has degradation levels (healthy → failure)
- 2205 cycles of operation

Components & States:
- Cooler: 3% (close to failure), 20%, 100% (full efficiency)
- Valve: 100% (optimal), 90%, 80%, 73% (severe lag)
- Pump: 0 (no leakage), 1, 2 (severe leakage)
- Accumulator: 130 bar (optimal), 115, 100, 90 bar (close to failure)

Sensors (17):
- PS1-6: Pressure sensors (bar)
- EPS1: Motor power (W)
- FS1-2: Volume flow sensors (l/min)
- TS1-4: Temperature sensors (C)
- VS1: Vibration sensor (mm/s)
- CE: Cooling efficiency (virtual)
- CP: Cooling power (kW)
- SE: Efficiency factor

Reference: UCI ML Repository
Link: https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems
"""

import os
import tempfile
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# UCI download URL
HYDRAULIC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip"
SOURCE = "hydraulic"

# Sensor files and their sampling rates
SENSOR_FILES = {
    'PS1.txt': {'name': 'PS1', 'desc': 'Pressure bar', 'hz': 100},
    'PS2.txt': {'name': 'PS2', 'desc': 'Pressure bar', 'hz': 100},
    'PS3.txt': {'name': 'PS3', 'desc': 'Pressure bar', 'hz': 100},
    'PS4.txt': {'name': 'PS4', 'desc': 'Pressure bar', 'hz': 100},
    'PS5.txt': {'name': 'PS5', 'desc': 'Pressure bar', 'hz': 100},
    'PS6.txt': {'name': 'PS6', 'desc': 'Pressure bar', 'hz': 100},
    'EPS1.txt': {'name': 'EPS1', 'desc': 'Motor power W', 'hz': 100},
    'FS1.txt': {'name': 'FS1', 'desc': 'Volume flow l/min', 'hz': 10},
    'FS2.txt': {'name': 'FS2', 'desc': 'Volume flow l/min', 'hz': 10},
    'TS1.txt': {'name': 'TS1', 'desc': 'Temperature C', 'hz': 1},
    'TS2.txt': {'name': 'TS2', 'desc': 'Temperature C', 'hz': 1},
    'TS3.txt': {'name': 'TS3', 'desc': 'Temperature C', 'hz': 1},
    'TS4.txt': {'name': 'TS4', 'desc': 'Temperature C', 'hz': 1},
    'VS1.txt': {'name': 'VS1', 'desc': 'Vibration mm/s', 'hz': 1},
    'CE.txt': {'name': 'CE', 'desc': 'Cooling efficiency', 'hz': 1},
    'CP.txt': {'name': 'CP', 'desc': 'Cooling power kW', 'hz': 1},
    'SE.txt': {'name': 'SE', 'desc': 'Efficiency factor', 'hz': 1},
}

# Component labels
COMPONENT_NAMES = ['cooler', 'valve', 'pump', 'accumulator', 'stable_flag']


def download_hydraulic(cache_dir: Optional[Path] = None) -> Path:
    """
    Download and extract Hydraulic Systems data.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "hydraulic_data"

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "data.zip"

    # Check if already downloaded
    if (cache_dir / "PS1.txt").exists():
        print(f"  Using cached data at {cache_dir}")
        return cache_dir

    # Download
    print(f"  Downloading Hydraulic Systems data from UCI...")
    urllib.request.urlretrieve(HYDRAULIC_URL, zip_path)

    # Extract
    print(f"  Extracting to {cache_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)

    return cache_dir


def load_profile(data_dir: Path) -> pd.DataFrame:
    """Load the component condition labels (profile.txt)."""
    profile_path = data_dir / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    # Profile has 5 columns: cooler, valve, pump, accumulator, stable_flag
    profile = pd.read_csv(profile_path, sep='\t', header=None, names=COMPONENT_NAMES)
    return profile


def load_sensor(data_dir: Path, filename: str) -> np.ndarray:
    """Load a single sensor file (rows=cycles, cols=samples within cycle)."""
    filepath = data_dir / filename
    if not filepath.exists():
        return None

    # Each row is one cycle, columns are samples within the cycle
    data = pd.read_csv(filepath, sep='\t', header=None)
    return data.values


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch Hydraulic Systems observations.

    Args:
        config: Dict with keys:
            - aggregate: How to aggregate within-cycle samples
                         "mean", "std", "max", "min", or "all" (default: "mean")
            - include_labels: Include component conditions as signals
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts
    """
    aggregate = config.get("aggregate", "mean")
    include_labels = config.get("include_labels", True)
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)

    # Download data
    data_dir = download_hydraulic(cache_dir)

    # Load component labels
    profile = load_profile(data_dir)
    n_cycles = len(profile)
    print(f"  Loaded profile: {n_cycles} cycles")

    all_observations = []
    base_date = date(2000, 1, 1)

    # Load and process each sensor
    for filename, info in SENSOR_FILES.items():
        sensor_name = info['name']
        print(f"  Loading {sensor_name}...")

        data = load_sensor(data_dir, filename)
        if data is None:
            print(f"    SKIP: {filename} not found")
            continue

        # Aggregate within-cycle samples
        if aggregate == "mean":
            values = np.nanmean(data, axis=1)
        elif aggregate == "std":
            values = np.nanstd(data, axis=1)
        elif aggregate == "max":
            values = np.nanmax(data, axis=1)
        elif aggregate == "min":
            values = np.nanmin(data, axis=1)
        elif aggregate == "all":
            # Flatten - each sample is an observation
            for cycle_idx in range(n_cycles):
                cycle_data = data[cycle_idx]
                obs_date = base_date + timedelta(days=cycle_idx)
                for sample_idx, value in enumerate(cycle_data):
                    if not np.isnan(value):
                        all_observations.append({
                            "signal_id": f"HYD_{sensor_name}",
                            "observed_at": obs_date,
                            "value": float(value),
                            "source": SOURCE,
                        })
            continue
        else:
            values = np.nanmean(data, axis=1)

        # One observation per cycle
        for cycle_idx, value in enumerate(values):
            if not np.isnan(value):
                obs_date = base_date + timedelta(days=cycle_idx)
                all_observations.append({
                    "signal_id": f"HYD_{sensor_name}",
                    "observed_at": obs_date,
                    "value": float(value),
                    "source": SOURCE,
                })

    # Add component condition labels
    if include_labels:
        print("  Adding component labels...")
        for col in COMPONENT_NAMES:
            for cycle_idx, value in enumerate(profile[col]):
                obs_date = base_date + timedelta(days=cycle_idx)
                all_observations.append({
                    "signal_id": f"HYD_{col.upper()}",
                    "observed_at": obs_date,
                    "value": float(value),
                    "source": SOURCE,
                })

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    config = {
        "aggregate": "mean",
        "include_labels": True,
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    if results:
        df = pd.DataFrame(results[:20])
        print("\nSample observations:")
        print(df)

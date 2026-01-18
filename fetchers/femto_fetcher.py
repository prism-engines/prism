#!/usr/bin/env python3
"""
FEMTO/PRONOSTIA Bearing Dataset Fetcher

Downloads the PRONOSTIA bearing degradation dataset from the PHM 2012 Challenge.
This is a run-to-failure accelerated degradation test.

Dataset Info:
- 17 bearings tested under accelerated degradation conditions
- 3 operating conditions (speed × load):
  - Condition 1: 1800 rpm, 4000 N
  - Condition 2: 1650 rpm, 4200 N
  - Condition 3: 1500 rpm, 5000 N
- Horizontal and vertical accelerometer readings
- Sampling: 25.6 kHz for 0.1s every 10s (2560 samples per snapshot)
- Temperature readings every 60s

Training vs Test:
- Learning set: Bearings 1_1, 1_2, 2_1, 2_2, 3_1, 3_2 (full life)
- Test set: Truncated data for RUL prediction

Reference: PHM 2012 Prognostics Challenge
Link: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
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

SOURCE = "femto"

# FEMTO/PRONOSTIA dataset structure
# The data is organized by condition (1, 2, 3) and bearing number
OPERATING_CONDITIONS = {
    1: {"speed": 1800, "load": 4000},
    2: {"speed": 1650, "load": 4200},
    3: {"speed": 1500, "load": 5000},
}

LEARNING_BEARINGS = [
    ("1", "1"), ("1", "2"),
    ("2", "1"), ("2", "2"),
    ("3", "1"), ("3", "2"),
]

TEST_BEARINGS = [
    ("1", "3"), ("1", "4"), ("1", "5"), ("1", "6"), ("1", "7"),
    ("2", "3"), ("2", "4"), ("2", "5"), ("2", "6"), ("2", "7"),
    ("3", "3"),
]

# NASA hosted mirror
FEMTO_URL = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/archive/master.zip"


def download_femto(cache_dir: Optional[Path] = None) -> Path:
    """
    Download and extract FEMTO/PRONOSTIA data.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "femto_data"

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "femto.zip"

    # Check if already downloaded
    extracted_dir = cache_dir / "phm-ieee-2012-data-challenge-dataset-master"
    if extracted_dir.exists():
        print(f"  Using cached data at {extracted_dir}")
        return extracted_dir

    # Download
    print(f"  Downloading FEMTO/PRONOSTIA data...")
    try:
        urllib.request.urlretrieve(FEMTO_URL, zip_path)
    except Exception as e:
        print(f"  Warning: Could not download ({e})")
        print(f"  Try manual download from: {FEMTO_URL}")
        raise

    # Extract
    print(f"  Extracting to {cache_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)

    return extracted_dir


def parse_acc_file(filepath: Path) -> np.ndarray:
    """
    Parse a FEMTO accelerometer file.
    Returns array of shape (n_samples, 2) for horizontal and vertical.
    """
    try:
        # FEMTO files are CSV with columns: hour, min, sec, microsec, horiz, vert
        df = pd.read_csv(filepath, header=None)
        # Return just the accelerometer data (columns 4 and 5, 0-indexed)
        return df.iloc[:, 4:6].values
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return None


def extract_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from accelerometer data.
    """
    if data is None or len(data) == 0:
        return {}

    features = {}
    for i, axis in enumerate(['horiz', 'vert']):
        axis_data = data[:, i] if len(data.shape) > 1 else data

        features[f'{axis}_rms'] = float(np.sqrt(np.mean(axis_data**2)))
        features[f'{axis}_mean'] = float(np.mean(axis_data))
        features[f'{axis}_std'] = float(np.std(axis_data))
        features[f'{axis}_max'] = float(np.max(np.abs(axis_data)))
        features[f'{axis}_peak_to_peak'] = float(np.max(axis_data) - np.min(axis_data))
        features[f'{axis}_kurtosis'] = float(pd.Series(axis_data).kurtosis())
        features[f'{axis}_skewness'] = float(pd.Series(axis_data).skew())

        rms = np.sqrt(np.mean(axis_data**2))
        features[f'{axis}_crest'] = float(np.max(np.abs(axis_data)) / rms) if rms > 0 else 0

    return features


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch FEMTO/PRONOSTIA Bearing observations.

    Args:
        config: Dict with keys:
            - data_type: "learning" or "test" or "all"
            - conditions: List of operating conditions (1, 2, 3)
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts
    """
    data_type = config.get("data_type", "learning")
    conditions = config.get("conditions", [1, 2, 3])
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)

    # Download data
    try:
        data_dir = download_femto(cache_dir)
    except Exception as e:
        print(f"  Error downloading FEMTO data: {e}")
        return []

    # Select bearings based on data_type
    if data_type == "learning":
        bearings = LEARNING_BEARINGS
        data_subdir = "Learning_set"
    elif data_type == "test":
        bearings = TEST_BEARINGS
        data_subdir = "Test_set"
    else:  # all
        bearings = LEARNING_BEARINGS + TEST_BEARINGS
        data_subdir = None

    all_observations = []
    base_date = date(2000, 1, 1)

    for cond_str, bearing_num in bearings:
        cond = int(cond_str)
        if cond not in conditions:
            continue

        bearing_id = f"Bearing{cond}_{bearing_num}"
        print(f"  Processing {bearing_id}...")

        # Find the bearing directory
        # Structure: Learning_set/Bearing1_1/acc_00001.csv
        if data_subdir:
            bearing_dir = data_dir / data_subdir / bearing_id
        else:
            # Try both
            bearing_dir = data_dir / "Learning_set" / bearing_id
            if not bearing_dir.exists():
                bearing_dir = data_dir / "Test_set" / bearing_id

        if not bearing_dir.exists():
            print(f"    SKIP: Directory not found: {bearing_dir}")
            continue

        # Find all accelerometer files
        acc_files = sorted(bearing_dir.glob("acc_*.csv"))
        print(f"    Found {len(acc_files)} accelerometer files")

        for file_idx, filepath in enumerate(acc_files):
            data = parse_acc_file(filepath)
            if data is None:
                continue

            features = extract_features(data)
            obs_date = base_date + timedelta(days=file_idx)

            # Add operating condition as metadata
            op_cond = OPERATING_CONDITIONS[cond]

            for feat_name, feat_value in features.items():
                all_observations.append({
                    "signal_id": f"FEMTO_{bearing_id}_{feat_name.upper()}",
                    "observed_at": obs_date,
                    "value": feat_value,
                    "source": SOURCE,
                })

            # Add speed and load as reference signals
            all_observations.append({
                "signal_id": f"FEMTO_{bearing_id}_SPEED",
                "observed_at": obs_date,
                "value": float(op_cond["speed"]),
                "source": SOURCE,
            })
            all_observations.append({
                "signal_id": f"FEMTO_{bearing_id}_LOAD",
                "observed_at": obs_date,
                "value": float(op_cond["load"]),
                "source": SOURCE,
            })

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    config = {
        "data_type": "learning",
        "conditions": [1],
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    if results:
        df = pd.DataFrame(results[:20])
        print("\nSample observations:")
        print(df)

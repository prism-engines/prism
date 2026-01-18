#!/usr/bin/env python3
"""
Case Western Reserve University (CWRU) Bearing Dataset Fetcher

Downloads the CWRU Bearing Data Center fault datasets.
This is the most widely used bearing fault benchmark in ML research.

Dataset Info:
- SKF bearings with seeded faults
- 4 fault conditions: Normal, Ball, Inner Race, Outer Race
- Fault diameters: 0.007", 0.014", 0.021", 0.028" (inches)
- Motor loads: 0-3 HP (0, 1, 2, 3 horsepower)
- Sampling rates: 12 kHz (drive end) and 48 kHz (fan end)
- ~250k samples per recording

Sensor Locations:
- DE: Drive End accelerometer
- FE: Fan End accelerometer
- BA: Base accelerometer

Reference: CWRU Bearing Data Center
Link: https://engineering.case.edu/bearingdatacenter
"""

import os
import tempfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SOURCE = "cwru_bearing"

# CWRU data URLs - these are .mat files
# Format: {condition}_{fault_size}_{motor_load}
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Key datasets
CWRU_FILES = {
    # Normal baseline
    "normal_0": "97.mat",
    "normal_1": "98.mat",
    "normal_2": "99.mat",
    "normal_3": "100.mat",

    # Inner race faults (0.007")
    "ir007_0": "105.mat",
    "ir007_1": "106.mat",
    "ir007_2": "107.mat",
    "ir007_3": "108.mat",

    # Ball faults (0.007")
    "ball007_0": "118.mat",
    "ball007_1": "119.mat",
    "ball007_2": "120.mat",
    "ball007_3": "121.mat",

    # Outer race faults (0.007") - centered
    "or007_0": "130.mat",
    "or007_1": "131.mat",
    "or007_2": "132.mat",
    "or007_3": "133.mat",

    # Inner race faults (0.014")
    "ir014_0": "169.mat",
    "ir014_1": "170.mat",
    "ir014_2": "171.mat",
    "ir014_3": "172.mat",

    # Ball faults (0.014")
    "ball014_0": "185.mat",
    "ball014_1": "186.mat",
    "ball014_2": "187.mat",
    "ball014_3": "188.mat",

    # Outer race faults (0.014") - centered
    "or014_0": "197.mat",
    "or014_1": "198.mat",
    "or014_2": "199.mat",
    "or014_3": "200.mat",

    # Inner race faults (0.021")
    "ir021_0": "209.mat",
    "ir021_1": "210.mat",
    "ir021_2": "211.mat",
    "ir021_3": "212.mat",

    # Ball faults (0.021")
    "ball021_0": "222.mat",
    "ball021_1": "223.mat",
    "ball021_2": "224.mat",
    "ball021_3": "225.mat",

    # Outer race faults (0.021") - centered
    "or021_0": "234.mat",
    "or021_1": "235.mat",
    "or021_2": "236.mat",
    "or021_3": "237.mat",
}


def download_cwru_file(filename: str, cache_dir: Path) -> Path:
    """Download a single CWRU .mat file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename

    if local_path.exists():
        return local_path

    url = CWRU_BASE_URL + filename
    print(f"    Downloading {filename}...")

    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        print(f"    Warning: Could not download {filename}: {e}")
        raise

    return local_path


def load_mat_file(filepath: Path) -> Dict[str, np.ndarray]:
    """Load a CWRU .mat file and extract accelerometer data."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for loading .mat files: pip install scipy")

    mat_data = loadmat(str(filepath))

    # Find the data arrays (they have names like X097_DE_time, X097_FE_time)
    result = {}
    for key, value in mat_data.items():
        if key.startswith('X') and '_time' in key:
            # Parse the key to get location (DE, FE, BA)
            parts = key.split('_')
            if len(parts) >= 2:
                location = parts[1]  # DE, FE, or BA
                result[location] = value.flatten()

    return result


def extract_features(data: np.ndarray, window_size: int = 4096) -> List[Dict[str, float]]:
    """
    Extract statistical features from vibration data using sliding windows.
    """
    features_list = []
    n_windows = len(data) // window_size

    for i in range(n_windows):
        window = data[i * window_size:(i + 1) * window_size]

        features = {
            'rms': float(np.sqrt(np.mean(window**2))),
            'mean': float(np.mean(window)),
            'std': float(np.std(window)),
            'max': float(np.max(np.abs(window))),
            'peak_to_peak': float(np.max(window) - np.min(window)),
            'kurtosis': float(pd.Series(window).kurtosis()),
            'skewness': float(pd.Series(window).skew()),
            'crest_factor': float(np.max(np.abs(window)) / np.sqrt(np.mean(window**2)))
                            if np.mean(window**2) > 0 else 0,
        }
        features_list.append(features)

    return features_list


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch CWRU Bearing observations.

    Args:
        config: Dict with keys:
            - conditions: List of fault conditions ("normal", "ir007", "ball007", "or007", etc.)
            - loads: List of motor loads (0, 1, 2, 3)
            - feature: Which feature to extract ("rms", "mean", "std", etc.)
            - window_size: Samples per window for feature extraction
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts
    """
    conditions = config.get("conditions", ["normal", "ir007", "ball007", "or007"])
    loads = config.get("loads", [0, 1, 2, 3])
    feature = config.get("feature", "rms")
    window_size = config.get("window_size", 4096)
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)
    else:
        cache_dir = Path(tempfile.gettempdir()) / "cwru_bearing_data"

    all_observations = []
    base_date = date(2000, 1, 1)
    obs_idx = 0

    for condition in conditions:
        for load in loads:
            key = f"{condition}_{load}"

            if key not in CWRU_FILES:
                print(f"  SKIP: Unknown condition/load: {key}")
                continue

            filename = CWRU_FILES[key]
            print(f"  Processing {key} ({filename})...")

            try:
                filepath = download_cwru_file(filename, cache_dir)
                data_dict = load_mat_file(filepath)
            except Exception as e:
                print(f"    Error: {e}")
                continue

            # Process each sensor location
            for location, data in data_dict.items():
                features_list = extract_features(data, window_size)

                for feat_idx, features in enumerate(features_list):
                    obs_date = base_date + timedelta(days=obs_idx)

                    # Main feature as primary signal
                    if feature in features:
                        all_observations.append({
                            "signal_id": f"CWRU_{condition.upper()}_{location}_{feature.upper()}",
                            "observed_at": obs_date,
                            "value": features[feature],
                            "source": SOURCE,
                        })

                    # All features as separate signals
                    for feat_name, feat_value in features.items():
                        all_observations.append({
                            "signal_id": f"CWRU_{condition.upper()}_{location}_{feat_name.upper()}",
                            "observed_at": obs_date,
                            "value": feat_value,
                            "source": SOURCE,
                        })

                    obs_idx += 1

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    config = {
        "conditions": ["normal", "ir007"],
        "loads": [0],
        "feature": "rms",
        "window_size": 4096,
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    if results:
        df = pd.DataFrame(results[:20])
        print("\nSample observations:")
        print(df)

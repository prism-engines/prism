#!/usr/bin/env python3
"""
NASA IMS Bearings Dataset Fetcher

Downloads the NASA Intelligent Maintenance Systems (IMS) bearing dataset.
This is a run-to-failure experiment from the Center for Intelligent Maintenance Systems.

Dataset Info:
- 4 bearings on a loaded shaft running at 2000 RPM
- 3 test runs (test sets), each running to bearing failure
- Accelerometer data sampled at 20 kHz
- Each file contains 1 second of data (20,480 points)
- Test 1: 2156 files over 35 days
- Test 2: 984 files over 7 days
- Test 3: 6324 files over 45 days

Failure Modes:
- Test 1: Inner race fault on bearing 3, roller fault on bearing 4
- Test 2: Outer race fault on bearing 1
- Test 3: Outer race fault on bearing 3

Reference: NASA Prognostics Data Repository
Link: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
"""

import os
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# NASA data URLs (hosted on various mirrors)
NASA_BEARING_URLS = {
    "1st_test": "https://data.nasa.gov/download/7wwx-fk77/application%2Fzip",
    "2nd_test": "https://data.nasa.gov/download/brfb-gzcv/application%2Fzip",
    "3rd_test": "https://data.nasa.gov/download/hmnm-nxsc/application%2Fzip",
}

SOURCE = "nasa_bearing"

# Bearings in each test
BEARINGS = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4']


def download_nasa_bearings(test_set: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Download and extract NASA bearing data for a specific test set.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "nasa_bearing_data"

    test_dir = cache_dir / test_set
    test_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{test_set}.zip"

    # Check if already downloaded
    if list(test_dir.glob("*")):
        print(f"  Using cached data at {test_dir}")
        return test_dir

    # Download
    url = NASA_BEARING_URLS.get(test_set)
    if not url:
        raise ValueError(f"Unknown test set: {test_set}")

    print(f"  Downloading {test_set} from NASA...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  Warning: Could not download from NASA ({e})")
        print(f"  Try manual download from: {url}")
        raise

    # Extract
    print(f"  Extracting to {test_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(test_dir)

    return test_dir


def parse_bearing_file(filepath: Path) -> np.ndarray:
    """
    Parse a single bearing data file.
    Returns array of shape (n_samples, n_bearings).
    """
    # NASA bearing files are space/tab delimited with 4 columns (one per bearing)
    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None)
        return data.values
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return None


def extract_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from raw vibration data.
    """
    if data is None or len(data) == 0:
        return {}

    features = {
        'rms': float(np.sqrt(np.mean(data**2))),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'max': float(np.max(np.abs(data))),
        'min': float(np.min(data)),
        'peak_to_peak': float(np.max(data) - np.min(data)),
        'kurtosis': float(pd.Series(data).kurtosis()),
        'skewness': float(pd.Series(data).skew()),
        'crest_factor': float(np.max(np.abs(data)) / np.sqrt(np.mean(data**2))) if np.mean(data**2) > 0 else 0,
    }
    return features


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch NASA Bearing observations.

    Args:
        config: Dict with keys:
            - test_sets: List of test sets to include ("1st_test", "2nd_test", "3rd_test")
            - feature: Which feature to extract ("rms", "mean", "std", "max", etc.)
                       Default: "rms" (most common for bearing analysis)
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts
    """
    test_sets = config.get("test_sets", ["1st_test", "2nd_test", "3rd_test"])
    feature = config.get("feature", "rms")
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)

    all_observations = []

    for test_set in test_sets:
        print(f"  Processing {test_set}...")

        try:
            data_dir = download_nasa_bearings(test_set, cache_dir)
        except Exception as e:
            print(f"  SKIP {test_set}: {e}")
            continue

        # Find all data files (they're in subdirectories by timestamp)
        # Structure varies by test set
        data_files = sorted(data_dir.rglob("*.txt")) + sorted(data_dir.rglob("*.csv"))

        if not data_files:
            # Try looking for directories with numeric names (timestamps)
            for subdir in sorted(data_dir.iterdir()):
                if subdir.is_dir():
                    data_files.extend(sorted(subdir.iterdir()))

        print(f"    Found {len(data_files)} files")

        # Process each file
        base_date = datetime(2000, 1, 1)

        for file_idx, filepath in enumerate(data_files):
            if not filepath.is_file():
                continue

            # Try to parse the file
            data = parse_bearing_file(filepath)
            if data is None:
                continue

            # Create observation time (each file is ~10 minutes apart in real experiment)
            obs_time = base_date + timedelta(minutes=file_idx * 10)
            obs_date = obs_time.date()

            # Extract features for each bearing
            n_bearings = min(data.shape[1], 4) if len(data.shape) > 1 else 1

            for bearing_idx in range(n_bearings):
                bearing_data = data[:, bearing_idx] if len(data.shape) > 1 else data
                features = extract_features(bearing_data)

                if feature in features:
                    all_observations.append({
                        "signal_id": f"NASA_{test_set.upper()}_B{bearing_idx + 1}_{feature.upper()}",
                        "observed_at": obs_date,
                        "value": features[feature],
                        "source": SOURCE,
                    })

                    # Also add all features as separate signals
                    for feat_name, feat_value in features.items():
                        if feat_name != feature:  # Already added main feature
                            all_observations.append({
                                "signal_id": f"NASA_{test_set.upper()}_B{bearing_idx + 1}_{feat_name.upper()}",
                                "observed_at": obs_date,
                                "value": feat_value,
                                "source": SOURCE,
                            })

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    config = {
        "test_sets": ["1st_test"],
        "feature": "rms",
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    if results:
        df = pd.DataFrame(results[:20])
        print("\nSample observations:")
        print(df)

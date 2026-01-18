#!/usr/bin/env python3
"""
C-MAPSS: Load into PRISM with proper cohort structure.

Structure:
- Domain: cmapss_fd001 (one dataset)
- Cohorts: u001, u002, ... u100 (one per engine)
- Signals: u001_s1, u001_s2, ... (21 sensors per engine)

This matches PRISM's design:
- Signals = individual sensors
- Cohort = one engine unit (21 sensors)
- Domain = all engines of same type
"""

import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================

SOURCE_DIR = Path("/Users/jasonrudder/prism-mac/data/CMAPSSData")
PRISM_DATA = Path("/Users/jasonrudder/prism-mac/data")

SENSOR_COLS = [f's{i}' for i in range(1, 22)]
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
ALL_COLUMNS = ['unit', 'cycle'] + SETTING_COLS + SENSOR_COLS

RUL_CAP = 125


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_cmapss(filepath: Path) -> pd.DataFrame:
    """Load C-MAPSS text file."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=ALL_COLUMNS)
    return df


def create_prism_domain(dataset: str) -> str:
    """
    Create PRISM domain with cohort structure.

    Returns domain name.
    """
    domain = f"cmapss_{dataset.lower()}"
    domain_dir = PRISM_DATA / domain / 'raw'
    config_dir = PRISM_DATA / domain / 'config'
    domain_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = SOURCE_DIR / f"train_{dataset}.txt"
    df = load_cmapss(train_path)

    # Compute RUL for metadata
    max_cycles = df.groupby('unit')['cycle'].max()
    df = df.merge(max_cycles.rename('max_cycle'), left_on='unit', right_index=True)
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=RUL_CAP)

    units = sorted(df['unit'].unique())
    print(f"  Units: {len(units)}")
    print(f"  Sensors per unit: {len(SENSOR_COLS)}")
    print(f"  Total signals: {len(units) * len(SENSOR_COLS)}")

    # ==========================================================================
    # 1. OBSERVATIONS: One row per (unit, cycle, sensor)
    # ==========================================================================
    base_date = datetime(2000, 1, 1)
    obs_records = []

    for _, row in df.iterrows():
        unit = int(row['unit'])
        cycle = int(row['cycle'])
        obs_date = base_date + timedelta(days=cycle)

        for sensor in SENSOR_COLS:
            obs_records.append({
                'signal_id': f"u{unit:03d}_{sensor}",
                'obs_date': obs_date,
                'value': float(row[sensor]),
            })

    observations = pl.DataFrame(obs_records)
    observations.write_parquet(domain_dir / 'observations.parquet')
    print(f"  Observations: {len(observations):,} rows")

    # ==========================================================================
    # 2. INDICATORS: Metadata for each signal
    # ==========================================================================
    signal_records = []
    for unit in units:
        for sensor in SENSOR_COLS:
            signal_records.append({
                'signal_id': f"u{unit:03d}_{sensor}",
                'name': f"Unit {unit} {sensor}",
                'unit': unit,
                'sensor': sensor,
            })

    signals = pl.DataFrame(signal_records)
    signals.write_parquet(domain_dir / 'signals.parquet')
    print(f"  Signals: {len(signals):,}")

    # ==========================================================================
    # 3. COHORTS: One cohort per engine unit
    # ==========================================================================
    cohort_records = []
    for unit in units:
        unit_data = df[df['unit'] == unit]
        max_cycle = int(unit_data['max_cycle'].iloc[0])
        cohort_records.append({
            'cohort_id': f"u{unit:03d}",
            'name': f"Engine Unit {unit}",
            'unit': unit,
            'total_cycles': max_cycle,
            'final_rul': 0,  # All training engines run to failure
        })

    cohorts = pl.DataFrame(cohort_records)
    cohorts.write_parquet(config_dir / 'cohorts.parquet')
    print(f"  Cohorts: {len(cohorts):,}")

    # ==========================================================================
    # 4. COHORT_MEMBERS: Map signals to cohorts
    # ==========================================================================
    member_records = []
    for unit in units:
        for sensor in SENSOR_COLS:
            member_records.append({
                'cohort_id': f"u{unit:03d}",
                'signal_id': f"u{unit:03d}_{sensor}",
            })

    cohort_members = pl.DataFrame(member_records)
    cohort_members.write_parquet(config_dir / 'cohort_members.parquet')
    print(f"  Cohort members: {len(cohort_members):,}")

    # ==========================================================================
    # 5. ENGINE METADATA: RUL ground truth per engine
    # ==========================================================================
    engine_records = []
    for unit in units:
        unit_data = df[df['unit'] == unit]
        max_cycle = int(unit_data['max_cycle'].iloc[0])
        engine_records.append({
            'unit': unit,
            'dataset': dataset,
            'total_cycles': max_cycle,
            'final_rul': 0,
        })

    engine_meta = pl.DataFrame(engine_records)
    engine_meta.write_parquet(domain_dir / 'engine_metadata.parquet')
    print(f"  Engine metadata: {len(engine_meta):,}")

    return domain


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FD001', help='FD001, FD002, FD003, or FD004')
    args = parser.parse_args()

    print("=" * 70)
    print(f"C-MAPSS {args.dataset} - LOAD INTO PRISM")
    print("=" * 70)
    print()
    print("Structure:")
    print("  Domain: one per dataset")
    print("  Cohorts: one per engine unit")
    print("  Signals: 21 sensors per engine")
    print()

    domain = create_prism_domain(args.dataset)

    print()
    print(f"Domain created: {domain}")
    print()
    print("Next steps:")
    print(f"  python -m prism.entry_points.signal_vector --signal --domain {domain} --testing")
    print(f"  python -m prism.entry_points.cohort_geometry --domain {domain}")


if __name__ == '__main__':
    main()

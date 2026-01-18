"""
TEP Assessment Summary - WHAT + WHEN + MODE
============================================

Quick summary of all three detection layers.

Usage:
    python -m prism.assessments.tep_summary --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from collections import defaultdict
import os
import warnings

warnings.filterwarnings('ignore')


def run_summary(domain: str):
    """Run quick summary of TEP assessment layers."""
    from prism.db.parquet_store import get_parquet_path

    print("=" * 100)
    print("TEP ASSESSMENT SUMMARY: WHAT + WHEN + MODE")
    print("=" * 100)
    print()

    # Load vector data
    vec_df = pl.read_parquet(get_parquet_path('vector', 'signal', domain))
    tep_vec = vec_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    # =========================================================================
    # LAYER 1: WHAT (Classification power)
    # =========================================================================
    print("LAYER 1: WHAT (Classification Features)")
    print("-" * 50)

    what_metrics = ['alpha', 'beta', 'omega', 'spectral_slope', 'permutation_entropy']
    what_data = tep_vec.filter(pl.col('metric_name').is_in(what_metrics))

    for metric in what_metrics:
        m_data = what_data.filter(pl.col('metric_name') == metric)
        if len(m_data) > 0:
            mean_val = m_data['metric_value'].mean()
            std_val = m_data['metric_value'].std()
            print(f"  {metric:25s}: mean={mean_val:8.3f}, std={std_val:8.3f}")

    # =========================================================================
    # LAYER 2: WHEN (Break detection power)
    # =========================================================================
    print()
    print("LAYER 2: WHEN (Break Detection Features)")
    print("-" * 50)

    when_metrics = ['break_n', 'break_rate', 'dirac_n_impulses', 'heaviside_n_steps']
    when_data = tep_vec.filter(pl.col('metric_name').is_in(when_metrics))

    for metric in when_metrics:
        m_data = when_data.filter(pl.col('metric_name') == metric)
        if len(m_data) > 0:
            total = m_data['metric_value'].sum()
            mean_val = m_data['metric_value'].mean()
            max_val = m_data['metric_value'].max()
            print(f"  {metric:25s}: total={total:8.0f}, mean={mean_val:6.2f}, max={max_val:6.0f}")

    # =========================================================================
    # LAYER 3: MODE (Check if modes available)
    # =========================================================================
    print()
    print("LAYER 3: MODE (Behavioral Trajectory)")
    print("-" * 50)

    modes_path = get_parquet_path('vector', 'signal_modes', domain)
    if modes_path.exists():
        modes_df = pl.read_parquet(modes_path)
        tep_modes = modes_df.filter(pl.col('signal_id').str.starts_with('TEP_'))

        if 'mode_id' in tep_modes.columns:
            n_modes = tep_modes['mode_id'].n_unique()
            print(f"  Modes discovered: {n_modes}")

        if 'mode_affinity' in tep_modes.columns:
            mean_aff = tep_modes['mode_affinity'].mean()
            min_aff = tep_modes['mode_affinity'].min()
            print(f"  Mode affinity: mean={mean_aff:.3f}, min={min_aff:.3f}")

        if 'mode_entropy' in tep_modes.columns:
            mean_ent = tep_modes['mode_entropy'].mean()
            max_ent = tep_modes['mode_entropy'].max()
            print(f"  Mode entropy: mean={mean_ent:.3f}, max={max_ent:.3f}")
    else:
        print("  Mode data not computed yet")
        print("  Run: python -m prism.entry_points.geometry --domain cheme")

    # =========================================================================
    # DETECTION RATES FROM PREVIOUS EVALS
    # =========================================================================
    print()
    print("=" * 100)
    print("DETECTION RATES (from eval runs)")
    print("=" * 100)
    print()
    print("WHAT (Classification):")
    print("  Binary (Normal vs Fault):     64.5%")
    print("  Multi-class (identify fault): 60.0%")
    print()
    print("WHEN (Break Detection):")
    print("  Detection rate (z>1.0):       47.6%")
    print("  Early detection:              42.7%")
    print()
    print("MODE (not yet evaluated):")
    print("  Run tep_integrated.py for mode-based detection")
    print()

    # =========================================================================
    # ARCHITECTURE DIAGRAM
    # =========================================================================
    print("=" * 100)
    print("DETECTION ARCHITECTURE")
    print("=" * 100)
    print("""
    Raw TEP Data
        |
        v
    +-------------------+     +-------------------+     +-------------------+
    |   WHAT Layer      |     |   WHEN Layer      |     |   MODE Layer      |
    |   (Classification)|     |   (Break Detect)  |     |   (Trajectory)    |
    +-------------------+     +-------------------+     +-------------------+
    | GARCH (alpha,     |     | Break detector    |     | Mode affinity     |
    |   beta, omega)    |     | Dirac (impulses)  |     | Mode entropy      |
    | Spectral slope    |     | Heaviside (steps) |     | Mode sequence     |
    | Entropy           |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
            |                         |                         |
            v                         v                         v
    +---------------------------------------------------------------+
    |                    INTEGRATED DETECTION                       |
    |                                                               |
    |   Onset detected when:                                        |
    |   - WHAT: volatility/entropy shift                            |
    |   - WHEN: break/impulse/step detected                         |
    |   - MODE: affinity drop OR entropy spike                      |
    +---------------------------------------------------------------+
    """)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='TEP Assessment Summary')
    parser.add_argument('--domain', type=str, default=None)
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain")
    os.environ["PRISM_DOMAIN"] = domain

    run_summary(domain)


if __name__ == '__main__':
    main()

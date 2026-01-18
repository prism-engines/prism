#!/usr/bin/env python3
"""
PRISM Validation: MIMIC-IV ICU Sepsis Detection

Tests whether PRISM can detect sepsis regime transitions from ICU vitals.

Ground Truth:
    - Sepsis-3 criteria: SOFA >= 2 with suspected infection
    - Labels: septic vs stable

Hypothesis:
    - Septic patients should show different PRISM signatures
    - Entropy may increase (irregular vitals) or decrease (loss of variability)
    - Vitals may "decouple" before sepsis (cohort geometry change)

Reference:
    Johnson, A., et al. (2023). MIMIC-IV, a freely accessible EHR dataset.
    Scientific Data, 10(1), 1.
"""

import sys
from pathlib import Path

import polars as pl
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import f_oneway, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.engines.hurst import compute_hurst
from prism.engines.spectral import compute_spectral

try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False


def run_prism_engines(observations: pl.DataFrame) -> pl.DataFrame:
    """Run PRISM vector engines on ICU vital observations."""

    results = []
    signals = observations["signal_id"].unique().to_list()

    print(f"Processing {len(signals)} signals...")

    for i, signal_id in enumerate(signals):
        if i % 100 == 0:
            print(f"  {i}/{len(signals)}...")

        ts = observations.filter(pl.col("signal_id") == signal_id)
        values = ts["value"].to_numpy()

        if len(values) < 20:
            continue

        row = {"signal_id": signal_id}

        # Hurst
        try:
            hurst_result = compute_hurst(values)
            row["hurst"] = hurst_result.get("hurst_exponent")
        except:
            row["hurst"] = np.nan

        # Spectral
        try:
            spectral_result = compute_spectral(values)
            row["spectral_entropy"] = spectral_result.get("spectral_entropy")
        except:
            row["spectral_entropy"] = np.nan

        # Entropy
        if HAS_ANTROPY:
            try:
                row["sample_entropy"] = ant.sample_entropy(values, order=2)
            except:
                row["sample_entropy"] = np.nan
            try:
                row["permutation_entropy"] = ant.perm_entropy(values, order=3, normalize=True)
            except:
                row["permutation_entropy"] = np.nan
        else:
            row["sample_entropy"] = np.nan
            row["permutation_entropy"] = np.nan

        # Additional vital-specific metrics
        row["cv"] = values.std() / values.mean() if values.mean() != 0 else np.nan
        row["range"] = values.max() - values.min()

        results.append(row)

    return pl.DataFrame(results)


def main():
    data_dir = Path("data/mimic_demo")

    print("=" * 70)
    print("PRISM Validation: MIMIC-IV ICU Sepsis Detection")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    observations = pl.read_parquet(data_dir / "raw" / "observations.parquet")
    signals = pl.read_parquet(data_dir / "raw" / "signals.parquet")

    print(f"Observations: {len(observations)}")
    print(f"Signals: {len(signals)}")
    print()

    # Run PRISM
    print("Running PRISM engines...")
    metrics = run_prism_engines(observations)

    # Merge with signal metadata
    results = metrics.join(signals, on="signal_id", how="inner")

    print(f"\nResults: {len(results)} signals with metrics")
    print()

    # Save results
    (data_dir / "vector").mkdir(exist_ok=True)
    results.write_parquet(data_dir / "vector" / "signal.parquet")
    print(f"Saved to {data_dir / 'vector' / 'signal.parquet'}")
    print()

    # Analysis by regime
    print("=" * 70)
    print("RESULTS: Mean PRISM Metrics by Sepsis Regime")
    print("=" * 70)
    print()

    # Filter to valid results
    valid = results.filter(
        pl.col("sample_entropy").is_not_null() &
        pl.col("sample_entropy").is_finite()
    )

    summary = valid.group_by("regime").agg([
        pl.col("hurst").mean().round(3).alias("hurst"),
        pl.col("sample_entropy").mean().round(3).alias("sample_entropy"),
        pl.col("permutation_entropy").mean().round(3).alias("perm_entropy"),
        pl.col("spectral_entropy").mean().round(3).alias("spectral_entropy"),
        pl.col("cv").mean().round(3).alias("cv"),
        pl.len().alias("n"),
    ]).sort("regime")

    print(summary.to_pandas().to_string(index=False))
    print()

    # Analysis by vital sign
    print("=" * 70)
    print("RESULTS: PRISM Metrics by Vital Sign")
    print("=" * 70)
    print()

    vital_summary = valid.group_by("vital_name").agg([
        pl.col("hurst").mean().round(3).alias("hurst"),
        pl.col("sample_entropy").mean().round(3).alias("sample_entropy"),
        pl.len().alias("n"),
    ]).sort("vital_name")

    print(vital_summary.to_pandas().to_string(index=False))
    print()

    # ANOVA for regime discrimination
    print("=" * 70)
    print("ANOVA: Regime Discrimination by PRISM Metrics")
    print("=" * 70)
    print()

    septic = valid.filter(pl.col("regime") == "septic")
    stable = valid.filter(pl.col("regime") == "stable")

    for metric in ["sample_entropy", "permutation_entropy", "hurst", "cv"]:
        septic_vals = septic[metric].drop_nulls().to_numpy()
        stable_vals = stable[metric].drop_nulls().to_numpy()

        if len(septic_vals) > 5 and len(stable_vals) > 5:
            f_stat, p_val = f_oneway(septic_vals, stable_vals)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{metric:25s}: F = {f_stat:8.2f}, p = {p_val:.6f} {sig}")

    print()

    # Vital-specific regime analysis
    print("=" * 70)
    print("VITAL-SPECIFIC ANALYSIS: Septic vs Stable by Vital Type")
    print("=" * 70)
    print()

    for vital in ["heart_rate", "spo2", "respiratory_rate"]:
        vital_data = valid.filter(pl.col("vital_name") == vital)
        if len(vital_data) < 20:
            continue

        septic_v = vital_data.filter(pl.col("regime") == "septic")["sample_entropy"].drop_nulls().to_numpy()
        stable_v = vital_data.filter(pl.col("regime") == "stable")["sample_entropy"].drop_nulls().to_numpy()

        if len(septic_v) > 3 and len(stable_v) > 3:
            f_stat, p_val = f_oneway(septic_v, stable_v)
            print(f"{vital:25s}: Septic SampEn={septic_v.mean():.3f}, Stable SampEn={stable_v.mean():.3f}, F={f_stat:.2f}, p={p_val:.4f}")

    print()

    # 6-Axis Characterization
    print("=" * 70)
    print("6-AXIS CHARACTERIZATION: Signal Identification")
    print("=" * 70)
    print()

    print("Axis-based regime identification (without domain labels):")
    print()
    print(f"  Complexity (SampEn):  Septic = {septic['sample_entropy'].mean():.3f}, Stable = {stable['sample_entropy'].mean():.3f}")
    print(f"  Memory (Hurst):       Septic = {septic['hurst'].mean():.3f}, Stable = {stable['hurst'].mean():.3f}")
    print(f"  Variability (CV):     Septic = {septic['cv'].mean():.3f}, Stable = {stable['cv'].mean():.3f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("PRISM characterization of ICU vitals:")
    print("  - Entropy metrics may distinguish septic vs stable patients")
    print("  - Variability (CV) reflects vital sign stability")
    print("  - Multi-channel analysis needed for full regime detection")
    print()

    # Sample results
    print("Sample results (first 15):")
    print(valid.select([
        "signal_id", "vital_name", "regime",
        "hurst", "sample_entropy", "cv"
    ]).head(15).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()

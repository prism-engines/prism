#!/usr/bin/env python
"""
PRISM C-MAPSS Physics Validation
================================
Translates SQL validation queries to Polars for Parquet-based PRISM v2.0

Run after: characterize -> signal_vector -> laplace
"""

import polars as pl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/C_MAPSS_v2")


def parse_signal_id(signal_id: str) -> dict:
    """Parse CMAPSS_sensor_dataset_Uunit format."""
    parts = signal_id.split("_")
    if len(parts) >= 4 and parts[0] == "CMAPSS":
        sensor = parts[1]
        dataset = parts[2]
        unit = parts[3].replace("U", "") if parts[3].startswith("U") else parts[3]
        return {"sensor": sensor, "dataset": dataset, "unit": unit}
    return {"sensor": signal_id, "dataset": "unknown", "unit": "0"}


def load_data():
    """Load all required parquet files."""
    print("Loading data...")

    signal = pl.read_parquet(DATA_DIR / "vector/signal.parquet")
    field = pl.read_parquet(DATA_DIR / "vector/signal_field.parquet")
    char = pl.read_parquet(DATA_DIR / "raw/characterization.parquet")
    obs = pl.read_parquet(DATA_DIR / "raw/observations.parquet")

    # Parse signal_id components
    signal = signal.with_columns([
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["sensor"], return_dtype=pl.Utf8
        ).alias("sensor"),
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["dataset"], return_dtype=pl.Utf8
        ).alias("dataset"),
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["unit"], return_dtype=pl.Utf8
        ).alias("unit"),
    ])

    field = field.with_columns([
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["sensor"], return_dtype=pl.Utf8
        ).alias("sensor"),
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["dataset"], return_dtype=pl.Utf8
        ).alias("dataset"),
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["unit"], return_dtype=pl.Utf8
        ).alias("unit"),
    ])

    print(f"  signal: {signal.shape[0]:,} rows")
    print(f"  field: {field.shape[0]:,} rows")
    print(f"  characterization: {char.shape[0]:,} rows")
    print(f"  observations: {obs.shape[0]:,} rows")

    return signal, field, char, obs


def query_1_leading_signals(signal: pl.DataFrame) -> pl.DataFrame:
    """
    LEADING INDICATORS: Which sensors show behavioral shifts first?

    Uses gradient magnitude from Hurst/entropy metrics to detect early warning.
    """
    print("\n" + "="*70)
    print("QUERY 1: LEADING INDICATORS")
    print("Which sensors show behavioral shifts earliest?")
    print("="*70)

    # Focus on key degradation metrics
    key_metrics = ["hurst_exponent", "permutation_entropy", "sample_entropy"]

    # Get windowed metrics
    df = signal.filter(pl.col("metric_name").is_in(key_metrics))

    # Calculate shift detection per unit
    shifts = (
        df.sort(["signal_id", "metric_name", "obs_date"])
        .with_columns([
            pl.col("metric_value").shift(1).over(["signal_id", "metric_name"]).alias("prev_value"),
            pl.col("metric_value").std().over(["signal_id", "metric_name"]).alias("metric_std"),
        ])
        .with_columns([
            ((pl.col("metric_value") - pl.col("prev_value")).abs() /
             (pl.col("metric_std") + 0.001)).alias("z_change")
        ])
        .filter(pl.col("z_change") > 2.0)  # Significant shift threshold
        .group_by(["sensor", "dataset", "unit"])
        .agg([
            pl.col("obs_date").min().alias("first_shift_date"),
            pl.col("z_change").max().alias("max_z_change"),
        ])
    )

    # Get failure dates (max obs_date per unit)
    failure_dates = (
        signal
        .group_by(["dataset", "unit"])
        .agg(pl.col("obs_date").max().alias("failure_date"))
    )

    # Join and calculate lead time
    result = (
        shifts
        .join(failure_dates, on=["dataset", "unit"], how="left")
        .with_columns([
            (pl.col("failure_date") - pl.col("first_shift_date")).dt.total_days().alias("lead_days")
        ])
        .filter(pl.col("lead_days") > 0)
        .group_by("sensor")
        .agg([
            pl.col("lead_days").mean().alias("avg_lead_days"),
            pl.col("lead_days").std().alias("std_lead_days"),
            pl.col("max_z_change").mean().alias("avg_shift_magnitude"),
            pl.len().alias("n_units"),
        ])
        .sort("avg_lead_days", descending=True)
    )

    print(f"\nTop 15 Leading Sensors (by avg lead time):")
    print(result.head(15))

    return result


def query_5_degradation_signature(signal: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """
    DEGRADATION SIGNATURE: Metric profiles across RUL phases.

    Maps metrics to RUL bands: healthy (>100), degrading (50-100), critical (20-50), failing (<20)
    """
    print("\n" + "="*70)
    print("QUERY 5: DEGRADATION SIGNATURE")
    print("Metric profiles by RUL phase")
    print("="*70)

    # Get RUL observations
    rul_obs = obs.filter(pl.col("signal_id").str.contains("RUL"))

    # Parse RUL signal_ids to get dataset/unit
    rul_parsed = (
        rul_obs
        .with_columns([
            pl.col("signal_id").map_elements(
                lambda x: parse_signal_id(x)["dataset"], return_dtype=pl.Utf8
            ).alias("dataset"),
            pl.col("signal_id").map_elements(
                lambda x: parse_signal_id(x)["unit"], return_dtype=pl.Utf8
            ).alias("unit"),
            pl.col("value").alias("rul")
        ])
        .select(["dataset", "unit", "obs_date", "rul"])
    )

    # Assign RUL phases
    rul_phases = rul_parsed.with_columns([
        pl.when(pl.col("rul") > 100).then(pl.lit("1_healthy"))
        .when(pl.col("rul") > 50).then(pl.lit("2_degrading"))
        .when(pl.col("rul") > 20).then(pl.lit("3_critical"))
        .otherwise(pl.lit("4_failing"))
        .alias("phase")
    ])

    # Join signal metrics with RUL phases
    metrics_with_rul = (
        signal
        .join(
            rul_phases.select(["dataset", "unit", "obs_date", "phase"]),
            on=["dataset", "unit", "obs_date"],
            how="inner"
        )
    )

    # Compute signature by sensor, phase, metric
    signature = (
        metrics_with_rul
        .group_by(["sensor", "phase", "metric_name"])
        .agg([
            pl.col("metric_value").mean().alias("avg_value"),
            pl.col("metric_value").std().alias("std_value"),
            pl.len().alias("n_obs"),
        ])
        .filter(pl.col("n_obs") > 10)
        .sort(["sensor", "metric_name", "phase"])
    )

    # Show phase transitions for key metrics
    key_metrics = ["hurst_exponent", "permutation_entropy", "sample_entropy"]

    print(f"\nPhase transitions for key metrics:")
    for metric in key_metrics:
        metric_sig = (
            signature
            .filter(pl.col("metric_name") == metric)
            .pivot(index="sensor", on="phase", values="avg_value")
        )
        if metric_sig.shape[0] > 0:
            print(f"\n{metric}:")
            print(metric_sig.head(10))

    return signature


def query_6_stress_accumulation(field: pl.DataFrame) -> pl.DataFrame:
    """
    STRESS ACCUMULATION: Gradient magnitude and source/sink evolution.

    Tracks cumulative stress signals over time.
    """
    print("\n" + "="*70)
    print("QUERY 6: STRESS ACCUMULATION")
    print("Gradient magnitude and divergence trajectories")
    print("="*70)

    # Aggregate field metrics per unit over time
    stress = (
        field
        .filter(pl.col("gradient_magnitude").is_not_null())
        .group_by(["dataset", "unit", "window_end"])
        .agg([
            pl.col("gradient_magnitude").mean().alias("mean_gradient"),
            pl.col("gradient_magnitude").max().alias("max_gradient"),
            pl.col("divergence").mean().alias("mean_divergence"),
            pl.col("is_source").sum().alias("n_sources"),
            pl.col("is_sink").sum().alias("n_sinks"),
            pl.len().alias("n_metrics"),
        ])
        .sort(["dataset", "unit", "window_end"])
        .with_columns([
            pl.col("n_sources").cum_sum().over(["dataset", "unit"]).alias("cumulative_sources"),
            pl.col("n_sinks").cum_sum().over(["dataset", "unit"]).alias("cumulative_sinks"),
        ])
    )

    # Summary per dataset
    summary = (
        stress
        .group_by("dataset")
        .agg([
            pl.col("mean_gradient").mean().alias("avg_gradient"),
            pl.col("max_gradient").max().alias("peak_gradient"),
            pl.col("n_sources").sum().alias("total_sources"),
            pl.col("n_sinks").sum().alias("total_sinks"),
        ])
    )

    print(f"\nStress summary by dataset:")
    print(summary)

    # Show gradient trajectory for sample units
    sample_units = stress.filter(pl.col("dataset") == "FD001").select(["unit"]).unique().head(5)
    sample_trajectories = stress.filter(
        (pl.col("dataset") == "FD001") &
        (pl.col("unit").is_in(sample_units["unit"]))
    ).select(["unit", "window_end", "mean_gradient", "cumulative_sources", "cumulative_sinks"])

    print(f"\nSample trajectories (FD001, first 5 units):")
    print(sample_trajectories.head(20))

    return stress


def query_7_sensor_importance(char: pl.DataFrame) -> pl.DataFrame:
    """
    SENSOR IMPORTANCE: Based on characterization axes.

    Ranks sensors by discriminative properties (non-stationarity, complexity, memory).
    """
    print("\n" + "="*70)
    print("QUERY 7: SENSOR IMPORTANCE")
    print("Ranking by characterization axes")
    print("="*70)

    # Parse sensor from signal_id
    char_parsed = char.with_columns([
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["sensor"], return_dtype=pl.Utf8
        ).alias("sensor"),
        pl.col("signal_id").map_elements(
            lambda x: parse_signal_id(x)["dataset"], return_dtype=pl.Utf8
        ).alias("dataset"),
    ])

    # Compute importance score (high memory + low stationarity + high complexity = degradation signal)
    importance = (
        char_parsed
        .with_columns([
            (
                pl.col("ax_memory") * 0.4 +  # Persistence matters
                (1 - pl.col("ax_stationarity")) * 0.3 +  # Non-stationarity = trend
                pl.col("ax_complexity") * 0.2 +  # Information content
                pl.col("ax_determinism") * 0.1  # Predictability
            ).alias("importance_score")
        ])
        .group_by("sensor")
        .agg([
            pl.col("importance_score").mean().alias("avg_importance"),
            pl.col("ax_memory").mean().alias("avg_memory"),
            pl.col("ax_stationarity").mean().alias("avg_stationarity"),
            pl.col("ax_complexity").mean().alias("avg_complexity"),
            pl.col("ax_determinism").mean().alias("avg_determinism"),
            pl.col("ax_volatility").mean().alias("avg_volatility"),
            pl.len().alias("n_units"),
        ])
        .sort("avg_importance", descending=True)
    )

    print(f"\nTop 20 sensors by importance score:")
    print(importance.head(20))

    # Compare across datasets
    by_dataset = (
        char_parsed
        .with_columns([
            (
                pl.col("ax_memory") * 0.4 +
                (1 - pl.col("ax_stationarity")) * 0.3 +
                pl.col("ax_complexity") * 0.2 +
                pl.col("ax_determinism") * 0.1
            ).alias("importance_score")
        ])
        .group_by(["sensor", "dataset"])
        .agg([
            pl.col("importance_score").mean().alias("avg_importance"),
            pl.len().alias("n_units"),
        ])
        .pivot(index="sensor", on="dataset", values="avg_importance")
        .sort("FD001", descending=True, nulls_last=True)
    )

    print(f"\nImportance by dataset (top 15 from FD001):")
    print(by_dataset.head(15))

    return importance


def query_dynamical_class_distribution(char: pl.DataFrame) -> pl.DataFrame:
    """
    DYNAMICAL CLASS DISTRIBUTION: What types of dynamics do we see?
    """
    print("\n" + "="*70)
    print("BONUS: DYNAMICAL CLASS DISTRIBUTION")
    print("="*70)

    dist = (
        char
        .group_by("dynamical_class")
        .agg(pl.len().alias("count"))
        .with_columns([
            (pl.col("count") / pl.col("count").sum() * 100).alias("pct")
        ])
        .sort("count", descending=True)
    )

    print(f"\nTop 15 dynamical classes:")
    print(dist.head(15))

    return dist


def main():
    print("="*70)
    print("PRISM C-MAPSS PHYSICS VALIDATION")
    print("="*70)

    # Load data
    signal, field, char, obs = load_data()

    # Run queries
    q1 = query_1_leading_signals(signal)
    q5 = query_5_degradation_signature(signal, obs)
    q6 = query_6_stress_accumulation(field)
    q7 = query_7_sensor_importance(char)
    dyn = query_dynamical_class_distribution(char)

    # Summary of what's available vs needs more pipeline
    print("\n" + "="*70)
    print("PIPELINE STATUS")
    print("="*70)
    print("""
✓ Query 1: Leading Signals (from signal_vector)
✓ Query 5: Degradation Signature (from signal_vector + RUL)
✓ Query 6: Stress Accumulation (from laplace field)
✓ Query 7: Sensor Importance (from characterization)

✗ Query 2: Transfer Entropy → requires: state-level engines
✗ Query 3: Phase Transition → requires: cohort_geometry
✗ Query 4: Cohort Structure → requires: cohort_geometry (clustering)
✗ Query 8: Cointegration → requires: state-level engines
✗ Query 9: Summary Report → requires: above queries

Next steps to enable remaining queries:
  python -m prism.entry_points.cohort_geometry --domain C_MAPSS_v2
  python -m prism.entry_points.state --domain C_MAPSS_v2
    """)

    print("\nPhysics validation complete.")


if __name__ == "__main__":
    main()

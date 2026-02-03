"""
PRISM Signal Vector Temporal Engine

Creates signal_vector.parquet WITH temporal I column.
Each row is (unit_id, I, signal_id) with features computed
over a rolling window centered at index I.

This is the BRIDGE between observations and the geometry/dynamics pipeline.

The existing signal_vector.py aggregates per signal (one row per signal).
This produces per-index features (one row per signal per time point).

Pipeline:
    observations.parquet → signal_vector_temporal.py → signal_vector.parquet (with I)
                                                              ↓
                                                     state_vector.py
                                                              ↓
                                                     geometry/dynamics

Credit: Avery Rudder - Laplace transform insight
"""

import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from scipy import stats as scipy_stats


# ============================================================
# ENGINE REGISTRY
# ============================================================

def kurtosis_engine(y: np.ndarray) -> float:
    """Kurtosis (excess) - tail heaviness."""
    if len(y) < 4:
        return np.nan
    return float(scipy_stats.kurtosis(y, fisher=True, nan_policy='omit'))


def skewness_engine(y: np.ndarray) -> float:
    """Skewness - asymmetry."""
    if len(y) < 3:
        return np.nan
    return float(scipy_stats.skew(y, nan_policy='omit'))


def crest_factor_engine(y: np.ndarray) -> float:
    """Crest factor - peak / RMS (scale invariant)."""
    if len(y) < 1:
        return np.nan
    rms = np.sqrt(np.nanmean(y ** 2))
    if rms < 1e-10:
        return np.nan
    return float(np.nanmax(np.abs(y)) / rms)


def entropy_engine(y: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of distribution (normalized)."""
    if len(y) < bins:
        return np.nan

    # Remove NaN
    y_clean = y[~np.isnan(y)]
    if len(y_clean) < bins:
        return np.nan

    # Histogram
    counts, _ = np.histogram(y_clean, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    if len(probs) < 2:
        return 0.0

    # Normalized entropy
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(bins)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def hurst_engine(y: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (simplified)."""
    if len(y) < 20:
        return np.nan

    y_clean = y[~np.isnan(y)]
    if len(y_clean) < 20:
        return np.nan

    n = len(y_clean)

    # Simple R/S calculation
    mean_y = np.mean(y_clean)
    y_centered = y_clean - mean_y
    cumsum = np.cumsum(y_centered)

    R = np.max(cumsum) - np.min(cumsum)
    S = np.std(y_clean, ddof=1)

    if S < 1e-10 or R < 1e-10:
        return 0.5  # Random walk

    RS = R / S
    H = np.log(RS) / np.log(n)

    # Clamp to valid range
    return float(np.clip(H, 0.0, 1.0))


def autocorr_engine(y: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation at specified lag."""
    if len(y) < lag + 2:
        return np.nan

    y_clean = y[~np.isnan(y)]
    if len(y_clean) < lag + 2:
        return np.nan

    n = len(y_clean)
    mean_y = np.mean(y_clean)
    var_y = np.var(y_clean)

    if var_y < 1e-10:
        return 1.0  # Constant signal

    autocorr = np.sum((y_clean[:-lag] - mean_y) * (y_clean[lag:] - mean_y)) / ((n - lag) * var_y)

    return float(np.clip(autocorr, -1.0, 1.0))


def cv_engine(y: np.ndarray) -> float:
    """Coefficient of variation (std/mean) - scale invariant."""
    if len(y) < 2:
        return np.nan

    mean_y = np.nanmean(y)
    std_y = np.nanstd(y)

    if abs(mean_y) < 1e-10:
        return np.nan

    return float(std_y / abs(mean_y))


def range_ratio_engine(y: np.ndarray) -> float:
    """Range / mean - scale invariant."""
    if len(y) < 2:
        return np.nan

    mean_y = np.nanmean(y)
    range_y = np.nanmax(y) - np.nanmin(y)

    if abs(mean_y) < 1e-10:
        return np.nan

    return float(range_y / abs(mean_y))


def iqr_ratio_engine(y: np.ndarray) -> float:
    """IQR / median - scale invariant."""
    if len(y) < 4:
        return np.nan

    q75, q25 = np.nanpercentile(y, [75, 25])
    median = np.nanmedian(y)

    if abs(median) < 1e-10:
        return np.nan

    return float((q75 - q25) / abs(median))


# Engine registry
ENGINES = {
    'kurtosis': kurtosis_engine,
    'skewness': skewness_engine,
    'crest_factor': crest_factor_engine,
    'entropy': entropy_engine,
    'hurst': hurst_engine,
    'autocorr': lambda y: autocorr_engine(y, lag=1),
    'autocorr_10': lambda y: autocorr_engine(y, lag=10),
    'cv': cv_engine,
    'range_ratio': range_ratio_engine,
    'iqr_ratio': iqr_ratio_engine,
}

# Default engines to run
DEFAULT_ENGINES = ['kurtosis', 'skewness', 'crest_factor', 'entropy', 'hurst', 'autocorr']


# ============================================================
# TEMPORAL SIGNAL VECTOR COMPUTATION
# ============================================================

def compute_features_at_index(
    y: np.ndarray,
    engines: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute all features for a window of data.

    Args:
        y: Signal values in window
        engines: Dict of engine_name -> function

    Returns:
        Dict of feature_name -> value
    """
    results = {}

    for name, func in engines.items():
        try:
            results[name] = func(y)
        except Exception:
            results[name] = np.nan

    return results


def compute_signal_vector_temporal(
    observations_path: str,
    typology_path: str,
    output_path: str = "signal_vector.parquet",
    window_size: int = 100,
    stride: int = 1,
    engines: Optional[List[str]] = None,
    min_window_coverage: float = 0.5,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute temporal signal vector with I column.

    Each row is (unit_id, I, signal_id) with features computed
    over a rolling window.

    Args:
        observations_path: Path to observations.parquet
        typology_path: Path to typology.parquet
        output_path: Output path
        window_size: Rolling window size
        stride: Step between windows (1 = every index)
        engines: List of engine names to run (default: all)
        min_window_coverage: Minimum fraction of non-NaN values in window
        verbose: Print progress

    Returns:
        Signal vector DataFrame with I column
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL VECTOR TEMPORAL ENGINE")
        print(f"Window: {window_size}, Stride: {stride}")
        print("=" * 70)

    # Load data
    observations = pl.read_parquet(observations_path)
    typology = pl.read_parquet(typology_path)

    # Detect column names
    signal_col = 'signal_id' if 'signal_id' in observations.columns else 'signal_id'
    value_col = 'value' if 'value' in observations.columns else 'y'
    typology_signal_col = 'signal_id' if 'signal_id' in typology.columns else 'signal_id'

    if verbose:
        print(f"Signal column: {signal_col}")
        print(f"Value column: {value_col}")

    # Get active signals (not constant)
    # Check which column identifies constant signals
    if 'is_constant' in typology.columns:
        active_signals = typology.filter(
            ~pl.col('is_constant')
        )[typology_signal_col].unique().to_list()
    elif 'signal_std' in typology.columns:
        active_signals = typology.filter(
            pl.col('signal_std') > 1e-10
        )[typology_signal_col].unique().to_list()
    else:
        # No filter available, include all
        active_signals = typology[typology_signal_col].unique().to_list()

    if verbose:
        print(f"Active signals: {len(active_signals)}")

    # Filter observations
    observations = observations.filter(
        pl.col(signal_col).is_in(active_signals)
    )

    # Select engines
    if engines is None:
        engines = DEFAULT_ENGINES

    engine_funcs = {name: ENGINES[name] for name in engines if name in ENGINES}

    if verbose:
        print(f"Engines: {list(engine_funcs.keys())}")
        print()

    # Process each (unit_id, signal)
    results = []

    groups = observations.group_by(['unit_id', signal_col])
    n_groups = observations.select(['unit_id', signal_col]).unique().height

    if verbose:
        print(f"Processing {n_groups} signal groups...")

    processed = 0
    for (unit_id, signal_id), group in groups:
        # Skip null signal_id (unit_id can be null, signal_id cannot)
        if signal_id is None:
            continue

        # Sort by I
        group = group.sort('I')

        I_values = group['I'].to_numpy()
        y_values = group[value_col].to_numpy().astype(np.float64)
        n = len(y_values)

        if n < window_size:
            continue

        # Rolling window computation
        half_window = window_size // 2

        for i in range(0, n, stride):
            # Window centered at i
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Skip if window too small
            if end - start < window_size * min_window_coverage:
                continue

            window = y_values[start:end]

            # Skip if too many NaN
            nan_fraction = np.isnan(window).sum() / len(window)
            if nan_fraction > (1 - min_window_coverage):
                continue

            # Compute features
            features = compute_features_at_index(window, engine_funcs)

            # Build row
            row = {
                'unit_id': unit_id,
                'I': int(I_values[i]),
                'signal_id': signal_id,
                'window_start': int(I_values[start]),
                'window_end': int(I_values[end - 1]),
                'window_size': end - start,
            }
            row.update(features)

            results.append(row)

        processed += 1
        if verbose and processed % 10 == 0:
            print(f"  Processed {processed}/{n_groups} signals...")

    # Build DataFrame
    if not results:
        if verbose:
            print("WARNING: No results generated!")
        return pl.DataFrame()

    result = pl.DataFrame(results)

    # Sort by unit_id, signal_id, I
    result = result.sort(['unit_id', 'signal_id', 'I'])

    # Save
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")
        print(f"Columns: {result.columns}")
        print()
        print("Feature statistics:")
        for col in engine_funcs.keys():
            if col in result.columns:
                print(f"  {col}: mean={result[col].mean():.3f}, std={result[col].std():.3f}")

    return result


# ============================================================
# SQL ACCELERATION (for basic features)
# ============================================================

def compute_signal_vector_temporal_sql(
    observations_path: str,
    typology_path: str,
    output_path: str = "signal_vector.parquet",
    window_size: int = 100,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute temporal signal vector using SQL window functions.

    Faster for basic statistics but limited to what SQL can compute.
    Use Python version for entropy, hurst, etc.
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL VECTOR TEMPORAL (SQL)")
        print("=" * 70)

    con = duckdb.connect()

    # Load data
    con.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
    con.execute(f"CREATE TABLE typology AS SELECT * FROM read_parquet('{typology_path}')")

    # Detect columns
    cols = con.execute("SELECT * FROM observations LIMIT 1").pl().columns
    signal_col = 'signal_id' if 'signal_id' in cols else 'signal_id'
    value_col = 'value' if 'value' in cols else 'y'

    # Detect unit_id column (optional per CLAUDE.md - may be 'cohort' or absent)
    if 'unit_id' in cols:
        unit_col = 'unit_id'
    elif 'cohort' in cols:
        unit_col = 'cohort'
    else:
        unit_col = None

    # Check which column identifies constant signals
    typology_cols = con.execute("SELECT * FROM typology LIMIT 1").pl().columns

    # Build filter for active (non-constant) signals
    if 'is_constant' in typology_cols:
        constant_filter = "is_constant = FALSE"
    elif 'signal_std' in typology_cols:
        constant_filter = "signal_std > 1e-10"  # Non-zero variance = not constant
    else:
        constant_filter = "1=1"  # No filter available, include all

    # Build unit_id select and partition clauses
    unit_select = f"o.{unit_col} AS unit_id," if unit_col else "'default' AS unit_id,"
    unit_partition = f"o.{unit_col}, " if unit_col else ""

    # SQL with window functions
    sql = f"""
    WITH active_signals AS (
        SELECT DISTINCT signal_id
        FROM typology
        WHERE {constant_filter}
    ),

    windowed AS (
        SELECT
            {unit_select}
            o.I,
            o.{signal_col} AS signal_id,
            o.{value_col} AS value,

            -- Window aggregates
            AVG(o.{value_col}) OVER w AS window_mean,
            STDDEV(o.{value_col}) OVER w AS window_std,
            MIN(o.{value_col}) OVER w AS window_min,
            MAX(o.{value_col}) OVER w AS window_max,
            COUNT(*) OVER w AS window_size,
            -- Moments for manual kurtosis/skewness (avoids DuckDB KURTOSIS error on low-variance)
            SUM(o.{value_col}) OVER w AS m1_sum,
            SUM(o.{value_col} * o.{value_col}) OVER w AS m2_sum,
            SUM(o.{value_col} * o.{value_col} * o.{value_col}) OVER w AS m3_sum,
            SUM(o.{value_col} * o.{value_col} * o.{value_col} * o.{value_col}) OVER w AS m4_sum

        FROM observations o
        INNER JOIN active_signals a ON o.{signal_col} = a.signal_id
        WINDOW w AS (
            PARTITION BY {unit_partition}o.{signal_col}
            ORDER BY o.I
            ROWS BETWEEN {window_size // 2} PRECEDING AND {window_size // 2} FOLLOWING
        )
    )

    SELECT
        unit_id,
        I,
        signal_id,
        window_size,

        -- Kurtosis from moments (excess kurtosis = m4/m2^2 - 3)
        -- Guard against division by zero for low-variance windows
        CASE WHEN window_std > 1e-10 THEN
            ((m4_sum / window_size) - 4 * window_mean * (m3_sum / window_size) +
             6 * window_mean * window_mean * (m2_sum / window_size) -
             3 * window_mean * window_mean * window_mean * window_mean) /
            NULLIF(window_std * window_std * window_std * window_std, 0) - 3.0
        ELSE NULL END AS kurtosis,

        -- Skewness from moments (m3 / m2^1.5)
        CASE WHEN window_std > 1e-10 THEN
            ((m3_sum / window_size) - 3 * window_mean * (m2_sum / window_size) +
             2 * window_mean * window_mean * window_mean) /
            NULLIF(window_std * window_std * window_std, 0)
        ELSE NULL END AS skewness,

        -- Crest factor (max / rms approximation)
        GREATEST(ABS(window_max), ABS(window_min)) /
            NULLIF(SQRT(window_mean * window_mean + window_std * window_std), 0) AS crest_factor,

        -- Coefficient of variation
        window_std / NULLIF(ABS(window_mean), 0) AS cv,

        -- Range ratio
        (window_max - window_min) / NULLIF(ABS(window_mean), 0) AS range_ratio

    FROM windowed
    WHERE window_size >= {window_size // 2}
    ORDER BY unit_id, signal_id, I
    """

    result = con.execute(sql).pl()
    con.close()

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

    return result


# ============================================================
# COMBINED: SQL for basics, Python for complex
# ============================================================

def compute_signal_vector_temporal_hybrid(
    observations_path: str,
    typology_path: str,
    output_path: str = "signal_vector.parquet",
    window_size: int = 100,
    stride: int = 10,
    include_complex: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Hybrid approach: SQL for fast basics, Python for complex features.

    Args:
        observations_path: Path to observations.parquet
        typology_path: Path to typology.parquet
        output_path: Output path
        window_size: Rolling window size
        stride: Step between Python computations (SQL does all)
        include_complex: Include entropy, hurst (Python, slower)
        verbose: Print progress

    Returns:
        Combined signal vector DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL VECTOR TEMPORAL (HYBRID)")
        print("=" * 70)

    # Step 1: SQL for basic features (fast)
    if verbose:
        print("\n[1/2] SQL computation (kurtosis, skewness, cv)...")

    sql_result = compute_signal_vector_temporal_sql(
        observations_path, typology_path,
        output_path.replace('.parquet', '_sql.parquet'),
        window_size, verbose=False
    )

    if not include_complex:
        sql_result.write_parquet(output_path)
        return sql_result

    # Step 2: Python for complex features (slower, sampled)
    if verbose:
        print(f"\n[2/2] Python computation (entropy, hurst) with stride={stride}...")

    python_result = compute_signal_vector_temporal(
        observations_path, typology_path,
        output_path.replace('.parquet', '_python.parquet'),
        window_size=window_size,
        stride=stride,
        engines=['entropy', 'hurst', 'autocorr', 'autocorr_10'],
        verbose=False
    )

    # Step 3: Merge
    if verbose:
        print("\n[3/3] Merging results...")

    # Join on (unit_id, I, signal_id)
    if len(python_result) > 0:
        combined = sql_result.join(
            python_result.select(['unit_id', 'I', 'signal_id', 'entropy', 'hurst', 'autocorr', 'autocorr_10']),
            on=['unit_id', 'I', 'signal_id'],
            how='left'
        )
    else:
        combined = sql_result

    combined.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {combined.shape}")
        print(f"Columns: {combined.columns}")

    return combined


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Signal Vector Temporal Engine - Per-index features for dynamics

Usage:
    python signal_vector_temporal.py <observations.parquet> <typology.parquet> [output.parquet] [--window=100] [--stride=1]
    python signal_vector_temporal.py --sql <observations.parquet> <typology.parquet> [output.parquet]
    python signal_vector_temporal.py --hybrid <observations.parquet> <typology.parquet> [output.parquet]

Creates signal_vector.parquet WITH temporal I column:
- unit_id, I, signal_id, kurtosis, skewness, entropy, hurst, ...

This bridges observations to the geometry/dynamics pipeline.
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    # Parse arguments
    mode = 'python'
    args = sys.argv[1:]

    if args[0] == '--sql':
        mode = 'sql'
        args = args[1:]
    elif args[0] == '--hybrid':
        mode = 'hybrid'
        args = args[1:]

    observations_path = args[0]
    typology_path = args[1]
    output_path = args[2] if len(args) > 2 else "signal_vector.parquet"

    # Parse options
    window_size = 100
    stride = 1

    for arg in args[3:]:
        if arg.startswith('--window='):
            window_size = int(arg.split('=')[1])
        elif arg.startswith('--stride='):
            stride = int(arg.split('=')[1])

    # Run
    if mode == 'sql':
        compute_signal_vector_temporal_sql(observations_path, typology_path, output_path, window_size)
    elif mode == 'hybrid':
        compute_signal_vector_temporal_hybrid(observations_path, typology_path, output_path, window_size, stride)
    else:
        compute_signal_vector_temporal(observations_path, typology_path, output_path, window_size, stride)


if __name__ == "__main__":
    main()

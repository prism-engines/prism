"""
Stage 24: Gaussian Fingerprint Entry Point
==========================================

Pure orchestration - calls SQL engines for computation.

Inputs:
    - signal_vector.parquet

Outputs:
    - gaussian_fingerprint.parquet (per-signal Gaussian summary)
    - gaussian_similarity.parquet (pairwise Bhattacharyya distance)

Builds probabilistic fingerprints from windowed engine outputs,
then computes pairwise similarity between signals within each cohort.
"""

import argparse
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Optional


# Candidate fingerprint features (superset — only those present in signal_vector are used)
FINGERPRINT_FEATURES = [
    'spectral_entropy', 'hurst', 'sample_entropy', 'dfa', 'acf_decay',
    'kurtosis', 'skewness', 'trend_r2', 'recurrence_rate', 'determinism',
]

# Known cross-correlation pairs (only computed if both features exist)
CROSS_CORR_PAIRS = [
    ('spectral_entropy', 'hurst', 'corr_entropy_hurst'),
    ('spectral_entropy', 'sample_entropy', 'corr_spectral_sample_entropy'),
    ('hurst', 'dfa', 'corr_hurst_dfa'),
    ('kurtosis', 'skewness', 'corr_kurtosis_skewness'),
]


def _build_fingerprint_sql(features: List[str]) -> str:
    """Generate fingerprint SQL dynamically based on available columns."""
    # Mean and std lines (filter out NaN/Inf to prevent STDDEV_SAMP overflow)
    mean_lines = []
    std_lines = []
    for f in features:
        finite_filter = f"WHERE isfinite({f})"
        mean_lines.append(f"        AVG({f}) FILTER ({finite_filter}) AS mean_{f}")
        std_lines.append(f"        CASE WHEN COUNT({f}) FILTER ({finite_filter}) > 1 THEN STDDEV_SAMP({f}) FILTER ({finite_filter}) ELSE 0.0 END AS std_{f}")

    # Cross-correlations (only if both features present, filter NaN)
    corr_lines = []
    for fa, fb, alias in CROSS_CORR_PAIRS:
        if fa in features and fb in features:
            corr_lines.append(f"        CORR({fa}, {fb}) FILTER (WHERE isfinite({fa}) AND isfinite({fb})) AS {alias}")

    # Volatility: average of std columns
    vol_parts = [f"COALESCE(std_{f}, 0)" for f in features]
    vol_expr = " +\n            ".join(vol_parts)
    vol_line = f"    ({vol_expr}) / {len(features)}.0 AS fingerprint_volatility"

    # Assemble final SELECT from signal_stats
    select_cols = ["    cohort", "    signal_id", "    n_windows"]
    select_cols.extend([f"    mean_{f}" for f in features])
    select_cols.extend([f"    std_{f}" for f in features])
    for fa, fb, alias in CROSS_CORR_PAIRS:
        if fa in features and fb in features:
            select_cols.append(f"    {alias}")
    select_cols.append(vol_line)

    all_agg = ",\n".join(mean_lines + std_lines + corr_lines)
    select_block = ",\n".join(select_cols)

    sql = f"""
WITH signal_stats AS (
    SELECT
        COALESCE(cohort, '_default') AS cohort,
        signal_id,
        COUNT(*) AS n_windows,
{all_agg}
    FROM signal_vector
    GROUP BY COALESCE(cohort, '_default'), signal_id
)
SELECT
{select_block}
FROM signal_stats
ORDER BY cohort, signal_id;
"""
    return sql


def _build_similarity_sql(features: List[str]) -> str:
    """Generate similarity SQL dynamically based on available columns."""
    # Per-feature Bhattacharyya distance
    db_lines = []
    for f in features:
        db_lines.append(f"""        CASE WHEN a.std_{f} > 1e-10 AND b.std_{f} > 1e-10 THEN
            0.25 * POWER(a.mean_{f} - b.mean_{f}, 2)
                / (POWER(a.std_{f}, 2) + POWER(b.std_{f}, 2))
            + 0.5 * LN((POWER(a.std_{f}, 2) + POWER(b.std_{f}, 2))
                / (2.0 * a.std_{f} * b.std_{f}))
        ELSE NULL END AS db_{f}""")

    db_block = ",\n\n".join(db_lines)

    # Sum components
    coalesce_parts = [f"COALESCE(db_{f}, 0)" for f in features]
    sum_expr = " +\n        ".join(coalesce_parts)

    # Count non-null
    count_parts = [f"CASE WHEN db_{f} IS NOT NULL THEN 1 ELSE 0 END" for f in features]
    count_expr = " +\n        ".join(count_parts)

    # Per-feature diagnostic columns
    diag_cols = ",\n    ".join([f"db_{f}" for f in features])

    sql = f"""
WITH feature_distance AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,

{db_block},

        ABS(a.fingerprint_volatility - b.fingerprint_volatility) AS volatility_diff

    FROM gaussian_fingerprint a
    JOIN gaussian_fingerprint b
        ON a.cohort = b.cohort
        AND a.signal_id < b.signal_id
)
SELECT
    cohort,
    signal_a,
    signal_b,

    ({sum_expr}) AS bhattacharyya_distance,

    ({count_expr}) AS n_features,

    CASE WHEN ({count_expr}) > 0 THEN
        ({sum_expr}) / NULLIF(({count_expr}), 0)
    ELSE NULL END AS normalized_distance,

    EXP(-({sum_expr})) AS similarity,

    volatility_diff,

    {diag_cols}

FROM feature_distance
ORDER BY cohort, bhattacharyya_distance;
"""
    return sql


def run(
    signal_vector_path: str,
    fingerprint_output_path: str = "gaussian_fingerprint.parquet",
    similarity_output_path: str = "gaussian_similarity.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run Gaussian fingerprint and similarity computation.

    Dynamically discovers which feature columns exist in signal_vector
    and generates SQL accordingly — no hardcoded column assumptions.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        fingerprint_output_path: Output path for gaussian_fingerprint.parquet
        similarity_output_path: Output path for gaussian_similarity.parquet
        verbose: Print progress

    Returns:
        Fingerprint DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 24: GAUSSIAN FINGERPRINT")
        print("Probabilistic signal fingerprints + pairwise similarity")
        print("=" * 70)

    # Connect to DuckDB and load data
    con = duckdb.connect()
    con.execute(f"CREATE TABLE signal_vector AS SELECT * FROM read_parquet('{signal_vector_path}')")

    if verbose:
        n_rows = con.execute("SELECT COUNT(*) FROM signal_vector").fetchone()[0]
        n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM signal_vector").fetchone()[0]
        print(f"Loaded: {n_rows:,} windows, {n_signals} signals")

    # Discover available columns
    all_cols = [row[0] for row in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'signal_vector'"
    ).fetchall()]
    available_features = [f for f in FINGERPRINT_FEATURES if f in all_cols]

    if verbose:
        missing = [f for f in FINGERPRINT_FEATURES if f not in all_cols]
        print(f"Features available: {len(available_features)}/{len(FINGERPRINT_FEATURES)}")
        if missing:
            print(f"  Missing (skipped): {missing}")

    if len(available_features) == 0:
        if verbose:
            print("No fingerprint features found — skipping stage 24")
        con.close()
        result = pl.DataFrame()
        return result

    # Step 1: Compute fingerprints
    fingerprint_sql = _build_fingerprint_sql(available_features)

    if verbose:
        print("\nComputing Gaussian fingerprints...")

    fingerprint = con.execute(fingerprint_sql).pl()
    fingerprint.write_parquet(fingerprint_output_path)

    if verbose:
        print(f"  Fingerprints: {fingerprint.shape}")

    # Step 2: Load fingerprints and compute similarity
    if verbose:
        print("Computing pairwise similarity...")

    con.execute(f"CREATE TABLE gaussian_fingerprint AS SELECT * FROM read_parquet('{fingerprint_output_path}')")
    similarity_sql = _build_similarity_sql(available_features)
    similarity = con.execute(similarity_sql).pl()
    similarity.write_parquet(similarity_output_path)

    if verbose:
        print(f"  Similarity pairs: {similarity.shape}")
        print()
        print("=" * 50)
        print(f"  {Path(fingerprint_output_path).absolute()}")
        print(f"  {Path(similarity_output_path).absolute()}")
        print("=" * 50)

    con.close()

    return fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Stage 24: Gaussian Fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Builds probabilistic Gaussian fingerprints from signal_vector.parquet,
then computes pairwise Bhattacharyya similarity within each cohort.

Example:
  python -m engines.entry_points.stage_24_gaussian_fingerprint \\
      signal_vector.parquet \\
      -o gaussian_fingerprint.parquet \\
      --similarity gaussian_similarity.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='gaussian_fingerprint.parquet',
                        help='Output path for fingerprints (default: gaussian_fingerprint.parquet)')
    parser.add_argument('--similarity', default='gaussian_similarity.parquet',
                        help='Output path for similarity (default: gaussian_similarity.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        args.similarity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

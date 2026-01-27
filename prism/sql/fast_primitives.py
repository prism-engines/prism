"""
SQL Fast Primitives - DuckDB Vectorized Computations

Instant computations that would take minutes in Python loops.
DuckDB does them in milliseconds.

CANONICAL SPLIT:
    SQL (this file):  correlation, covariance, basic stats, derivatives, typology
    Python (engines): hurst, lyapunov, garch, entropy, fft, granger, dtw, rqa
"""

import duckdb
import pandas as pd
from pathlib import Path


def compute_all_fast(observations_path: str) -> dict:
    """
    Compute ALL SQL-able primitives in one shot.

    Returns dict with DataFrames:
        - 'signal_stats': Basic stats per signal
        - 'pairwise': Correlation, covariance per pair
        - 'pointwise': Derivatives per point
        - 'typology': Signal classification
    """
    con = duckdb.connect()

    # Load observations
    con.execute(f"""
        CREATE TABLE obs AS
        SELECT * FROM read_parquet('{observations_path}')
    """)

    results = {}

    # 1. SIGNAL STATS (per entity_id, signal_id)
    results['signal_stats'] = con.execute("""
        SELECT
            entity_id,
            signal_id,
            COUNT(*) AS n_points,
            AVG(y) AS y_mean,
            STDDEV_POP(y) AS y_std,
            MIN(y) AS y_min,
            MAX(y) AS y_max,
            MEDIAN(y) AS y_median,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_q25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS y_q75,
            SKEWNESS(y) AS y_skew,
            KURTOSIS(y) AS y_kurtosis,
            COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
            VAR_POP(y) AS variance
        FROM obs
        GROUP BY entity_id, signal_id
    """).df()

    # 2. TYPOLOGY (signal classification)
    results['typology'] = con.execute("""
        WITH stats AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_points,
                COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
                VAR_POP(y) AS variance,
                AVG(y) AS mean,
                STDDEV_POP(y) AS std,
                MIN(y) AS min,
                MAX(y) AS max
            FROM obs
            GROUP BY entity_id, signal_id
        )
        SELECT
            entity_id,
            signal_id,
            n_points,
            unique_ratio,
            variance,
            mean,
            std,
            min,
            max,
            CASE
                WHEN variance < 1e-10 THEN 'constant'
                WHEN unique_ratio < 0.01 THEN 'digital'
                WHEN unique_ratio < 0.05 THEN 'discrete'
                ELSE 'analog'
            END AS signal_class
        FROM stats
    """).df()

    # 3. PAIRWISE CORRELATIONS (the big one - instant in SQL)
    results['pairwise'] = con.execute("""
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y) AS correlation,
            COVAR_POP(a.y, b.y) AS covariance,
            COUNT(*) AS n_overlap
        FROM obs a
        JOIN obs b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        GROUP BY a.entity_id, a.signal_id, b.signal_id
        HAVING COUNT(*) >= 10
    """).df()

    # 4. LAG CORRELATIONS (lag-1 cross-correlation)
    results['lag_correlation'] = con.execute("""
        WITH lagged AS (
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS y_lag1
            FROM obs
        )
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y_lag1) AS lag1_corr_a_leads,
            CORR(a.y_lag1, b.y) AS lag1_corr_b_leads
        FROM lagged a
        JOIN lagged b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        WHERE a.y_lag1 IS NOT NULL AND b.y_lag1 IS NOT NULL
        GROUP BY a.entity_id, a.signal_id, b.signal_id
    """).df()

    # 5. POINTWISE DERIVATIVES (staged CTEs to avoid nested window functions)
    results['pointwise'] = con.execute("""
        WITH dy_calc AS (
            -- Stage 1: First derivative
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                y - LAG(y) OVER w AS dy,
                LEAD(y) OVER w - LAG(y) OVER w AS dy_central
            FROM obs
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        ),
        d2y_calc AS (
            -- Stage 2: Second derivative from first
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                dy,
                dy_central,
                dy - LAG(dy) OVER w AS d2y,
                LAG(dy) OVER w AS dy_prev
            FROM dy_calc
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        )
        SELECT
            entity_id,
            signal_id,
            I,
            y,
            COALESCE(dy, 0) AS dy,
            COALESCE(d2y, 0) AS d2y,
            COALESCE(dy_central, 0) / 2.0 AS dy_central,
            -- Curvature approximation: d2y / (1 + dy^2)^1.5
            CASE
                WHEN dy IS NOT NULL AND d2y IS NOT NULL
                THEN d2y / POWER(1 + dy*dy, 1.5)
                ELSE 0
            END AS curvature,
            -- Sign changes (regime boundaries)
            CASE
                WHEN dy > 0 AND dy_prev <= 0 THEN 1
                WHEN dy < 0 AND dy_prev >= 0 THEN -1
                ELSE 0
            END AS dy_sign_change
        FROM d2y_calc
    """).df()

    # 6. DERIVATIVE STATS (per signal) - staged CTEs
    results['derivative_stats'] = con.execute("""
        WITH dy_calc AS (
            -- Stage 1: First derivative
            SELECT
                entity_id,
                signal_id,
                I,
                y - LAG(y) OVER w AS dy
            FROM obs
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        ),
        d2y_calc AS (
            -- Stage 2: Second derivative + lag for sign changes
            SELECT
                entity_id,
                signal_id,
                dy,
                dy - LAG(dy) OVER w AS d2y,
                LAG(dy) OVER w AS dy_prev
            FROM dy_calc
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        )
        SELECT
            entity_id,
            signal_id,
            AVG(dy) AS dy_mean,
            STDDEV_POP(dy) AS dy_std,
            MAX(ABS(dy)) AS dy_max_abs,
            AVG(d2y) AS d2y_mean,
            STDDEV_POP(d2y) AS d2y_std,
            -- Count sign changes (volatility proxy)
            SUM(CASE WHEN dy > 0 AND dy_prev <= 0 THEN 1
                     WHEN dy < 0 AND dy_prev >= 0 THEN 1
                     ELSE 0 END) AS n_sign_changes
        FROM d2y_calc
        GROUP BY entity_id, signal_id
    """).df()

    con.close()
    return results


def compute_correlation_matrix(observations_path: str) -> pd.DataFrame:
    """Just the correlation matrix - fastest possible."""
    con = duckdb.connect()

    result = con.execute(f"""
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y) AS correlation
        FROM read_parquet('{observations_path}') a
        JOIN read_parquet('{observations_path}') b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        GROUP BY a.entity_id, a.signal_id, b.signal_id
    """).df()

    con.close()
    return result


def compute_typology(observations_path: str) -> pd.DataFrame:
    """Signal classification - instant."""
    con = duckdb.connect()

    result = con.execute(f"""
        WITH stats AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_points,
                COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
                VAR_POP(y) AS variance,
                AVG(y) AS mean,
                STDDEV_POP(y) AS std,
                MIN(y) AS min,
                MAX(y) AS max
            FROM read_parquet('{observations_path}')
            GROUP BY entity_id, signal_id
        )
        SELECT
            entity_id,
            signal_id,
            n_points,
            unique_ratio,
            variance,
            mean,
            std,
            min,
            max,
            CASE
                WHEN variance < 1e-10 THEN 'constant'
                WHEN unique_ratio < 0.01 THEN 'digital'
                WHEN unique_ratio < 0.05 THEN 'discrete'
                ELSE 'analog'
            END AS signal_class
        FROM stats
    """).df()

    con.close()
    return result


def compute_derivatives(observations_path: str) -> pd.DataFrame:
    """Pointwise derivatives - instant."""
    con = duckdb.connect()

    result = con.execute(f"""
        SELECT
            entity_id,
            signal_id,
            I,
            y,
            COALESCE(y - LAG(y) OVER w, 0) AS dy,
            COALESCE((y - LAG(y) OVER w) - LAG(y - LAG(y) OVER w) OVER w, 0) AS d2y
        FROM read_parquet('{observations_path}')
        WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
    """).df()

    con.close()
    return result


if __name__ == '__main__':
    import time

    obs_path = 'data/observations.parquet'

    print("Testing SQL fast primitives...")
    print("=" * 60)

    start = time.time()
    results = compute_all_fast(obs_path)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.3f}s")
    print("\nResults:")
    for name, df in results.items():
        print(f"  {name}: {len(df)} rows × {len(df.columns)} cols")

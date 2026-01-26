-- ============================================================================
-- ORTHON SQL ENGINES: STATISTICS
-- ============================================================================
-- Rolling and global statistics for signal characterization.
-- These form the basis for typology classification.
-- ============================================================================

-- ============================================================================
-- 001: GLOBAL STATISTICS (per signal)
-- ============================================================================

CREATE OR REPLACE VIEW v_stats_global AS
SELECT
    signal_id,
    index_dimension,
    signal_class,
    COUNT(*) AS n_points,
    MIN(y) AS y_min,
    MAX(y) AS y_max,
    MAX(y) - MIN(y) AS y_range,
    AVG(y) AS y_mean,
    STDDEV(y) AS y_std,
    VARIANCE(y) AS y_var,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY y) AS y_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS y_q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_iqr
FROM v_base
GROUP BY signal_id, index_dimension, signal_class;


-- ============================================================================
-- 002: ROLLING MEAN
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_mean AS
SELECT
    signal_id,
    I,
    y,
    AVG(y) OVER w_50 AS rolling_mean_50,
    AVG(y) OVER w_100 AS rolling_mean_100,
    AVG(y) OVER w_200 AS rolling_mean_200
FROM v_base
WINDOW 
    w_50 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING),
    w_100 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING),
    w_200 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 100 PRECEDING AND 100 FOLLOWING);


-- ============================================================================
-- 003: ROLLING STANDARD DEVIATION
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_std AS
SELECT
    signal_id,
    I,
    y,
    STDDEV(y) OVER w_50 AS rolling_std_50,
    STDDEV(y) OVER w_100 AS rolling_std_100,
    STDDEV(y) OVER w_200 AS rolling_std_200
FROM v_base
WINDOW 
    w_50 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING),
    w_100 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING),
    w_200 AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 100 PRECEDING AND 100 FOLLOWING);


-- ============================================================================
-- 004: Z-SCORE (standardized value)
-- ============================================================================

CREATE OR REPLACE VIEW v_zscore AS
SELECT
    b.signal_id,
    b.I,
    b.y,
    (b.y - s.y_mean) / NULLIF(s.y_std, 0) AS z_score,
    CASE
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 3 THEN 'extreme'
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 2 THEN 'outlier'
        ELSE 'normal'
    END AS z_category
FROM v_base b
JOIN v_stats_global s USING (signal_id);


-- ============================================================================
-- 005: ROLLING Z-SCORE (local anomaly detection)
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_zscore AS
SELECT
    signal_id,
    I,
    y,
    (y - AVG(y) OVER w) / NULLIF(STDDEV(y) OVER w, 0) AS rolling_zscore
FROM v_base
WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING);


-- ============================================================================
-- 006: SKEWNESS (distribution asymmetry)
-- ============================================================================
-- Skew = E[(X - μ)³] / σ³

CREATE OR REPLACE VIEW v_skewness AS
SELECT
    signal_id,
    AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 3)) AS skewness,
    CASE
        WHEN AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 3)) > 0.5 THEN 'right_skewed'
        WHEN AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 3)) < -0.5 THEN 'left_skewed'
        ELSE 'symmetric'
    END AS skew_category
FROM v_base
JOIN (SELECT signal_id, AVG(y) AS y_mean, STDDEV(y) AS y_std FROM v_base GROUP BY signal_id) sub 
    USING (signal_id)
GROUP BY signal_id, sub.y_mean, sub.y_std;


-- ============================================================================
-- 007: KURTOSIS (distribution tailedness)
-- ============================================================================
-- Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)

CREATE OR REPLACE VIEW v_kurtosis AS
SELECT
    signal_id,
    AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 4)) - 3 AS kurtosis,
    CASE
        WHEN AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 4)) - 3 > 1 THEN 'heavy_tailed'
        WHEN AVG(POWER((y - sub.y_mean) / NULLIF(sub.y_std, 0), 4)) - 3 < -1 THEN 'light_tailed'
        ELSE 'normal_tailed'
    END AS kurtosis_category
FROM v_base
JOIN (SELECT signal_id, AVG(y) AS y_mean, STDDEV(y) AS y_std FROM v_base GROUP BY signal_id) sub 
    USING (signal_id)
GROUP BY signal_id, sub.y_mean, sub.y_std;


-- ============================================================================
-- 008: COEFFICIENT OF VARIATION
-- ============================================================================
-- CV = std / |mean| - relative variability

CREATE OR REPLACE VIEW v_cv AS
SELECT
    signal_id,
    y_std / NULLIF(ABS(y_mean), 0) AS coefficient_of_variation,
    CASE
        WHEN y_std / NULLIF(ABS(y_mean), 0) > 1.0 THEN 'high_variability'
        WHEN y_std / NULLIF(ABS(y_mean), 0) > 0.3 THEN 'moderate_variability'
        ELSE 'low_variability'
    END AS variability_category
FROM v_stats_global;


-- ============================================================================
-- 009: PERCENTILE RANKS
-- ============================================================================

CREATE OR REPLACE VIEW v_percentile_rank AS
SELECT
    signal_id,
    I,
    y,
    PERCENT_RANK() OVER (PARTITION BY signal_id ORDER BY y) AS percentile_rank,
    NTILE(10) OVER (PARTITION BY signal_id ORDER BY y) AS decile,
    NTILE(100) OVER (PARTITION BY signal_id ORDER BY y) AS percentile_bucket
FROM v_base;


-- ============================================================================
-- 010: ROLLING MIN/MAX (local extrema detection)
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_extrema AS
SELECT
    signal_id,
    I,
    y,
    MIN(y) OVER w AS rolling_min,
    MAX(y) OVER w AS rolling_max,
    MAX(y) OVER w - MIN(y) OVER w AS rolling_range,
    (y - MIN(y) OVER w) / NULLIF(MAX(y) OVER w - MIN(y) OVER w, 0) AS normalized_position
FROM v_base
WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING);


-- ============================================================================
-- 011: LOCAL EXTREMA DETECTION (peaks and valleys)
-- ============================================================================

CREATE OR REPLACE VIEW v_local_extrema AS
SELECT
    signal_id,
    I,
    y,
    LAG(y) OVER w AS y_prev,
    LEAD(y) OVER w AS y_next,
    CASE
        WHEN y > LAG(y) OVER w AND y > LEAD(y) OVER w THEN 'peak'
        WHEN y < LAG(y) OVER w AND y < LEAD(y) OVER w THEN 'valley'
        ELSE 'neither'
    END AS extrema_type
FROM v_base
WINDOW w AS (PARTITION BY signal_id ORDER BY I);


-- ============================================================================
-- 012: MEAN ABSOLUTE DEVIATION (MAD)
-- ============================================================================

CREATE OR REPLACE VIEW v_mad AS
SELECT
    signal_id,
    AVG(ABS(y - sub.y_median)) AS mad
FROM v_base
JOIN (
    SELECT signal_id, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY y) AS y_median 
    FROM v_base GROUP BY signal_id
) sub USING (signal_id)
GROUP BY signal_id;


-- ============================================================================
-- 013: AUTOCORRELATION (lag-1)
-- ============================================================================
-- Correlation with self at lag 1

CREATE OR REPLACE VIEW v_autocorr_lag1 AS
SELECT
    a.signal_id,
    CORR(a.y, b.y) AS autocorr_lag1
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 1
GROUP BY a.signal_id;


-- ============================================================================
-- 014: AUTOCORRELATION (multiple lags)
-- ============================================================================

CREATE OR REPLACE VIEW v_autocorr_multi AS
SELECT
    signal_id,
    1 AS lag,
    CORR(a.y, b.y) AS autocorrelation
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 1
GROUP BY a.signal_id
UNION ALL
SELECT
    signal_id,
    5 AS lag,
    CORR(a.y, b.y) AS autocorrelation
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 5
GROUP BY a.signal_id
UNION ALL
SELECT
    signal_id,
    10 AS lag,
    CORR(a.y, b.y) AS autocorrelation
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 10
GROUP BY a.signal_id
UNION ALL
SELECT
    signal_id,
    20 AS lag,
    CORR(a.y, b.y) AS autocorrelation
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 20
GROUP BY a.signal_id;


-- ============================================================================
-- 015: RUNS TEST COMPONENTS (for randomness)
-- ============================================================================
-- Count runs above/below median

CREATE OR REPLACE VIEW v_runs AS
WITH median_calc AS (
    SELECT signal_id, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY y) AS y_median
    FROM v_base GROUP BY signal_id
),
above_below AS (
    SELECT 
        b.signal_id,
        b.I,
        CASE WHEN b.y > m.y_median THEN 1 ELSE 0 END AS above_median
    FROM v_base b
    JOIN median_calc m USING (signal_id)
),
run_changes AS (
    SELECT
        signal_id,
        I,
        above_median,
        CASE WHEN above_median != LAG(above_median) OVER (PARTITION BY signal_id ORDER BY I) THEN 1 ELSE 0 END AS new_run
    FROM above_below
)
SELECT
    signal_id,
    SUM(new_run) + 1 AS n_runs,
    COUNT(*) AS n_total,
    SUM(above_median) AS n_above,
    COUNT(*) - SUM(above_median) AS n_below
FROM run_changes
GROUP BY signal_id;


-- ============================================================================
-- STATISTICS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_stats_complete AS
SELECT
    g.signal_id,
    g.index_dimension,
    g.signal_class,
    g.n_points,
    g.y_min,
    g.y_max,
    g.y_range,
    g.y_mean,
    g.y_std,
    g.y_var,
    g.y_q1,
    g.y_median,
    g.y_q3,
    g.y_iqr,
    s.skewness,
    s.skew_category,
    k.kurtosis,
    k.kurtosis_category,
    cv.coefficient_of_variation,
    cv.variability_category,
    m.mad,
    ac.autocorr_lag1,
    r.n_runs
FROM v_stats_global g
LEFT JOIN v_skewness s USING (signal_id)
LEFT JOIN v_kurtosis k USING (signal_id)
LEFT JOIN v_cv cv USING (signal_id)
LEFT JOIN v_mad m USING (signal_id)
LEFT JOIN v_autocorr_lag1 ac USING (signal_id)
LEFT JOIN v_runs r USING (signal_id);

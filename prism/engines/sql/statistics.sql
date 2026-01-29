-- =============================================================================
-- Statistics Engine (SQL)
-- =============================================================================
-- Computes basic statistics for each signal.
-- Input: observations table with (entity_id, signal_id, I, y)
-- Output: signal-level statistics
-- =============================================================================

SELECT
    entity_id,
    signal_id,
    COUNT(*) AS n_points,
    AVG(y) AS mean,
    STDDEV_SAMP(y) AS std,
    MIN(y) AS min,
    MAX(y) AS max,
    MAX(y) - MIN(y) AS range,
    MEDIAN(y) AS median,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS iqr,
    VARIANCE(y) AS variance,
    STDDEV_SAMP(y) / NULLIF(ABS(AVG(y)), 0) AS cv
FROM observations
GROUP BY entity_id, signal_id
ORDER BY entity_id, signal_id;

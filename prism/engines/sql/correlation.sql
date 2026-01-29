-- =============================================================================
-- Correlation Engine (SQL)
-- =============================================================================
-- Computes pairwise Pearson correlation between all signal pairs per entity.
-- Input: observations table with (entity_id, signal_id, I, y)
-- Output: correlation matrix as (entity_id, signal_a, signal_b, correlation)
-- =============================================================================

WITH signal_pairs AS (
    SELECT DISTINCT
        a.entity_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b
    FROM observations a
    INNER JOIN observations b
        ON a.entity_id = b.entity_id
        AND a.signal_id < b.signal_id
),
aligned AS (
    SELECT
        sp.entity_id,
        sp.signal_a,
        sp.signal_b,
        a.I,
        a.y AS y_a,
        b.y AS y_b
    FROM signal_pairs sp
    INNER JOIN observations a
        ON sp.entity_id = a.entity_id
        AND sp.signal_a = a.signal_id
    INNER JOIN observations b
        ON sp.entity_id = b.entity_id
        AND sp.signal_b = b.signal_id
        AND a.I = b.I
)
SELECT
    entity_id,
    signal_a,
    signal_b,
    CORR(y_a, y_b) AS correlation,
    COUNT(*) AS n_points
FROM aligned
GROUP BY entity_id, signal_a, signal_b
HAVING COUNT(*) >= 10
ORDER BY entity_id, signal_a, signal_b;

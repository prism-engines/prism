-- =============================================================================
-- Regime Assignment Engine (SQL)
-- =============================================================================
-- Assigns observations to regimes based on percentile bins.
-- Input: observations table with (entity_id, signal_id, I, value)
-- Output: observations with regime_id based on quartiles
-- =============================================================================

WITH signal_quartiles AS (
    SELECT
        entity_id,
        signal_id,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS q2,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS q3
    FROM observations
    GROUP BY entity_id, signal_id
)
SELECT
    o.entity_id,
    o.signal_id,
    o.I,
    o.value,
    CASE
        WHEN o.value <= q.q1 THEN 0
        WHEN o.value <= q.q2 THEN 1
        WHEN o.value <= q.q3 THEN 2
        ELSE 3
    END AS regime_id,
    CASE
        WHEN o.value <= q.q1 THEN 'low'
        WHEN o.value <= q.q2 THEN 'mid_low'
        WHEN o.value <= q.q3 THEN 'mid_high'
        ELSE 'high'
    END AS regime_name
FROM observations o
INNER JOIN signal_quartiles q
    ON o.entity_id = q.entity_id
    AND o.signal_id = q.signal_id
ORDER BY o.entity_id, o.signal_id, o.I;

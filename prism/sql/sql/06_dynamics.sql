-- ============================================================================
-- ORTHON SQL ENGINES: DYNAMICAL SYSTEMS
-- ============================================================================
-- Regime detection, transitions, stability, basins, and attractors.
-- This is where we understand the system's phase space behavior.
-- ============================================================================

-- ============================================================================
-- 001: ROLLING STATISTICS FOR REGIME DETECTION
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_regime_stats AS
SELECT
    signal_id,
    I,
    y,
    
    -- Rolling statistics
    AVG(y) OVER w_pre AS mean_pre,
    AVG(y) OVER w_post AS mean_post,
    STDDEV(y) OVER w_pre AS std_pre,
    STDDEV(y) OVER w_post AS std_post,
    
    -- Changes
    AVG(y) OVER w_post - AVG(y) OVER w_pre AS mean_change,
    STDDEV(y) OVER w_post / NULLIF(STDDEV(y) OVER w_pre, 0) AS std_ratio
    
FROM v_base
WINDOW 
    w_pre AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING),
    w_post AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 1 FOLLOWING AND 50 FOLLOWING);


-- ============================================================================
-- 002: REGIME CHANGE DETECTION
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_changes AS
WITH change_scores AS (
    SELECT
        signal_id,
        I,
        mean_change,
        std_ratio,
        std_pre,
        std_post,
        
        -- Change significance
        ABS(mean_change) / NULLIF(std_pre, 0) AS mean_change_zscore,
        ABS(LN(NULLIF(std_ratio, 0))) AS volatility_change_score,
        
        -- Combined change score
        ABS(mean_change) / NULLIF(std_pre, 0) + ABS(LN(NULLIF(std_ratio, 0))) AS total_change_score
        
    FROM v_rolling_regime_stats
    WHERE std_pre IS NOT NULL AND std_post IS NOT NULL
)
SELECT
    signal_id,
    I,
    mean_change,
    std_pre,
    std_post,
    std_ratio,
    mean_change_zscore,
    volatility_change_score,
    total_change_score,
    CASE
        WHEN mean_change_zscore > 2 OR volatility_change_score > 0.7 THEN TRUE
        ELSE FALSE
    END AS is_regime_change
FROM change_scores;


-- ============================================================================
-- 003: REGIME BOUNDARIES
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_boundaries AS
WITH changes AS (
    SELECT signal_id, I, total_change_score
    FROM v_regime_changes
    WHERE is_regime_change
),
ranked AS (
    SELECT
        signal_id,
        I,
        total_change_score,
        LAG(I) OVER (PARTITION BY signal_id ORDER BY I) AS prev_change_I,
        I - LAG(I) OVER (PARTITION BY signal_id ORDER BY I) AS gap
    FROM changes
)
SELECT
    signal_id,
    I AS regime_boundary,
    total_change_score,
    gap AS regime_duration,
    ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I) AS regime_number
FROM ranked
WHERE gap IS NULL OR gap > 20;  -- Minimum regime duration


-- ============================================================================
-- 004: REGIME ASSIGNMENT
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_assignment AS
WITH boundaries AS (
    SELECT signal_id, regime_boundary, regime_number
    FROM v_regime_boundaries
)
SELECT
    b.signal_id,
    b.I,
    COALESCE(
        (SELECT MAX(regime_number) 
         FROM boundaries bo 
         WHERE bo.signal_id = b.signal_id AND bo.regime_boundary <= b.I),
        0
    ) AS regime_id
FROM v_base b;


-- ============================================================================
-- 005: REGIME STATISTICS
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_stats AS
SELECT
    r.signal_id,
    r.regime_id,
    COUNT(*) AS regime_length,
    MIN(b.I) AS regime_start,
    MAX(b.I) AS regime_end,
    AVG(b.y) AS regime_mean,
    STDDEV(b.y) AS regime_std,
    MIN(b.y) AS regime_min,
    MAX(b.y) AS regime_max,
    AVG(c.dy) AS regime_avg_velocity,
    AVG(c.kappa) AS regime_avg_curvature
FROM v_regime_assignment r
JOIN v_base b ON r.signal_id = b.signal_id AND r.I = b.I
LEFT JOIN v_curvature c ON r.signal_id = c.signal_id AND r.I = c.I
GROUP BY r.signal_id, r.regime_id;


-- ============================================================================
-- 006: REGIME TRANSITIONS
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_transitions AS
WITH sequential AS (
    SELECT
        signal_id,
        regime_id,
        LEAD(regime_id) OVER (PARTITION BY signal_id ORDER BY regime_start) AS next_regime,
        regime_end AS transition_point,
        regime_mean,
        LEAD(regime_mean) OVER (PARTITION BY signal_id ORDER BY regime_start) AS next_regime_mean,
        regime_std,
        LEAD(regime_std) OVER (PARTITION BY signal_id ORDER BY regime_start) AS next_regime_std
    FROM v_regime_stats
)
SELECT
    signal_id,
    regime_id AS from_regime,
    next_regime AS to_regime,
    transition_point,
    next_regime_mean - regime_mean AS mean_jump,
    next_regime_std / NULLIF(regime_std, 0) AS volatility_ratio,
    CASE
        WHEN next_regime_std > regime_std * 1.5 THEN 'volatility_increase'
        WHEN next_regime_std < regime_std * 0.67 THEN 'volatility_decrease'
        WHEN next_regime_mean > regime_mean + regime_std THEN 'upward_shift'
        WHEN next_regime_mean < regime_mean - regime_std THEN 'downward_shift'
        ELSE 'lateral'
    END AS transition_type
FROM sequential
WHERE next_regime IS NOT NULL;


-- ============================================================================
-- 007: TRANSITION MATRIX
-- ============================================================================
-- Probability of transitioning between regime types

CREATE OR REPLACE VIEW v_transition_matrix AS
WITH regime_types AS (
    SELECT
        signal_id,
        regime_id,
        CASE
            WHEN regime_std > (SELECT AVG(regime_std) FROM v_regime_stats) * 1.5 THEN 'high_vol'
            WHEN regime_std < (SELECT AVG(regime_std) FROM v_regime_stats) * 0.5 THEN 'low_vol'
            ELSE 'normal_vol'
        END AS regime_type
    FROM v_regime_stats
),
transitions AS (
    SELECT
        a.signal_id,
        a.regime_type AS from_type,
        b.regime_type AS to_type
    FROM regime_types a
    JOIN regime_types b ON a.signal_id = b.signal_id AND b.regime_id = a.regime_id + 1
)
SELECT
    from_type,
    to_type,
    COUNT(*) AS n_transitions,
    COUNT(*)::FLOAT / SUM(COUNT(*)) OVER (PARTITION BY from_type) AS transition_probability
FROM transitions
GROUP BY from_type, to_type;


-- ============================================================================
-- 008: STABILITY ANALYSIS
-- ============================================================================
-- Local stability from derivative behavior

CREATE OR REPLACE VIEW v_stability AS
SELECT
    signal_id,
    I,
    
    -- Local stability indicators
    dy,
    d2y,
    
    -- Lyapunov-like local stability (positive = unstable)
    ABS(dy) AS local_expansion_rate,
    
    -- Convergence indicator (negative d2y when moving away from equilibrium)
    CASE
        WHEN dy > 0 AND d2y < 0 THEN 'converging_up'
        WHEN dy < 0 AND d2y > 0 THEN 'converging_down'
        WHEN dy > 0 AND d2y > 0 THEN 'diverging_up'
        WHEN dy < 0 AND d2y < 0 THEN 'diverging_down'
        ELSE 'neutral'
    END AS stability_state,
    
    -- Is locally stable?
    CASE
        WHEN (dy > 0 AND d2y < 0) OR (dy < 0 AND d2y > 0) THEN TRUE
        ELSE FALSE
    END AS is_locally_stable

FROM v_d2y
WHERE dy IS NOT NULL AND d2y IS NOT NULL;


-- ============================================================================
-- 009: BASIN DETECTION (from local minima)
-- ============================================================================
-- Find basins of attraction as regions around local minima

CREATE OR REPLACE VIEW v_basins AS
WITH minima AS (
    SELECT signal_id, I AS basin_center, y AS basin_depth
    FROM v_local_extrema
    WHERE extrema_type = 'valley'
),
basin_assignment AS (
    SELECT
        b.signal_id,
        b.I,
        b.y,
        m.basin_center,
        m.basin_depth,
        ABS(b.I - m.basin_center) AS distance_to_center,
        ROW_NUMBER() OVER (PARTITION BY b.signal_id, b.I ORDER BY ABS(b.I - m.basin_center)) AS rank
    FROM v_base b
    JOIN minima m ON b.signal_id = m.signal_id
)
SELECT
    signal_id,
    I,
    y,
    basin_center,
    basin_depth,
    distance_to_center
FROM basin_assignment
WHERE rank = 1 AND distance_to_center < 100;


-- ============================================================================
-- 010: ATTRACTOR DETECTION
-- ============================================================================
-- Points that the system frequently returns to

CREATE OR REPLACE VIEW v_attractors AS
WITH binned_values AS (
    SELECT
        signal_id,
        ROUND(y, 1) AS y_bin,
        COUNT(*) AS bin_count,
        AVG(y) AS bin_center
    FROM v_base
    GROUP BY signal_id, ROUND(y, 1)
),
ranked_bins AS (
    SELECT
        signal_id,
        y_bin,
        bin_count,
        bin_center,
        bin_count::FLOAT / SUM(bin_count) OVER (PARTITION BY signal_id) AS visit_frequency,
        ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY bin_count DESC) AS rank
    FROM binned_values
)
SELECT
    signal_id,
    y_bin AS attractor_value,
    bin_center,
    bin_count AS visit_count,
    visit_frequency,
    CASE
        WHEN rank = 1 THEN 'primary'
        WHEN rank <= 3 THEN 'secondary'
        ELSE 'minor'
    END AS attractor_type
FROM ranked_bins
WHERE visit_frequency > 0.05;  -- At least 5% of time


-- ============================================================================
-- 011: RECURRENCE PROXY
-- ============================================================================
-- How often does the system return to similar states?

CREATE OR REPLACE VIEW v_recurrence_proxy AS
WITH value_returns AS (
    SELECT
        a.signal_id,
        a.I AS I1,
        b.I AS I2,
        ABS(a.y - b.y) AS value_diff
    FROM v_base a
    JOIN v_base b ON a.signal_id = b.signal_id 
        AND b.I > a.I + 10 
        AND b.I < a.I + 200
),
near_returns AS (
    SELECT
        signal_id,
        COUNT(*) AS n_returns,
        AVG(I2 - I1) AS avg_return_time
    FROM value_returns
    WHERE value_diff < (SELECT AVG(y_std) * 0.5 FROM v_stats_global)
    GROUP BY signal_id
)
SELECT
    sg.signal_id,
    COALESCE(nr.n_returns, 0) AS recurrence_count,
    nr.avg_return_time,
    COALESCE(nr.n_returns, 0)::FLOAT / sg.n_points AS recurrence_rate
FROM v_stats_global sg
LEFT JOIN near_returns nr USING (signal_id);


-- ============================================================================
-- 012: BIFURCATION DETECTION
-- ============================================================================
-- Where does the system behavior qualitatively change?

CREATE OR REPLACE VIEW v_bifurcation_candidates AS
WITH curvature_changes AS (
    SELECT
        signal_id,
        I,
        kappa,
        AVG(kappa) OVER w_pre AS kappa_pre,
        AVG(kappa) OVER w_post AS kappa_post,
        STDDEV(kappa) OVER w_pre AS kappa_std_pre,
        STDDEV(kappa) OVER w_post AS kappa_std_post
    FROM v_curvature
    WINDOW 
        w_pre AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING),
        w_post AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 1 FOLLOWING AND 30 FOLLOWING)
)
SELECT
    signal_id,
    I AS bifurcation_point,
    kappa_pre,
    kappa_post,
    ABS(kappa_post - kappa_pre) / NULLIF(kappa_std_pre, 0) AS bifurcation_score,
    CASE
        WHEN kappa_post > kappa_pre * 2 THEN 'complexity_increase'
        WHEN kappa_post < kappa_pre * 0.5 THEN 'complexity_decrease'
        ELSE 'curvature_shift'
    END AS bifurcation_type
FROM curvature_changes
WHERE kappa_std_pre IS NOT NULL
  AND ABS(kappa_post - kappa_pre) / NULLIF(kappa_std_pre, 0) > 2;


-- ============================================================================
-- 013: PHASE VELOCITY
-- ============================================================================
-- Speed of movement through phase space

CREATE OR REPLACE VIEW v_phase_velocity AS
SELECT
    signal_id,
    I,
    SQRT(1 + dy*dy) AS phase_velocity,
    SQRT(dy*dy + d2y*d2y) AS tangent_magnitude,
    AVG(SQRT(1 + dy*dy)) OVER w AS rolling_phase_velocity
FROM v_d2y
WHERE dy IS NOT NULL AND d2y IS NOT NULL
WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING);


-- ============================================================================
-- DYNAMICAL SYSTEMS SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_dynamics_complete AS
SELECT
    ra.signal_id,
    ra.I,
    ra.regime_id,
    rs.regime_mean,
    rs.regime_std,
    rs.regime_length,
    s.stability_state,
    s.is_locally_stable,
    ba.basin_center,
    ba.distance_to_center,
    pv.phase_velocity,
    rp.recurrence_rate
FROM v_regime_assignment ra
LEFT JOIN v_regime_stats rs ON ra.signal_id = rs.signal_id AND ra.regime_id = rs.regime_id
LEFT JOIN v_stability s ON ra.signal_id = s.signal_id AND ra.I = s.I
LEFT JOIN v_basins ba ON ra.signal_id = ba.signal_id AND ra.I = ba.I
LEFT JOIN v_phase_velocity pv ON ra.signal_id = pv.signal_id AND ra.I = pv.I
LEFT JOIN v_recurrence_proxy rp ON ra.signal_id = rp.signal_id;


-- ============================================================================
-- SYSTEM-LEVEL REGIME DETECTION
-- ============================================================================
-- When multiple signals change regime together

CREATE OR REPLACE VIEW v_system_regime AS
WITH regime_change_counts AS (
    SELECT
        I,
        COUNT(DISTINCT signal_id) AS n_signals_changing
    FROM v_regime_changes
    WHERE is_regime_change
    GROUP BY I
)
SELECT
    I AS system_regime_boundary,
    n_signals_changing,
    CASE
        WHEN n_signals_changing > (SELECT COUNT(DISTINCT signal_id) FROM v_base) * 0.5 THEN 'major_system_change'
        WHEN n_signals_changing > 2 THEN 'moderate_system_change'
        ELSE 'isolated_change'
    END AS change_magnitude
FROM regime_change_counts
WHERE n_signals_changing > 1
ORDER BY I;

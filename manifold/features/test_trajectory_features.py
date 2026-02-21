"""
Tests for trajectory feature extraction.

Validates:
    1. Feature extraction runs without error
    2. Output shape and column structure correct
    3. Cross-axis features computed correctly
    4. Known engines produce expected collapse patterns
    5. No unexpected all-NaN columns
"""

import polars as pl
import numpy as np
import pytest
from pathlib import Path

from trajectory_features import (
    extract_single_axis_features,
    extract_cross_axis_features,
    build_feature_matrix,
    TRAJECTORY_METRICS,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures: synthetic trajectory data
# ─────────────────────────────────────────────────────────────────────

def _make_synthetic_sigs(n_engines=10, n_windows=8, collapse_engines=None):
    """Create synthetic trajectory signatures for testing."""
    collapse_engines = collapse_engines or []
    rows = []
    for i in range(n_engines):
        eng = f'engine_{i+1}'
        collapsing = eng in collapse_engines
        for w in range(n_windows):
            t = w / (n_windows - 1)  # 0 to 1
            
            if collapsing:
                eff_dim = 2.8 - 0.5 * t  # 2.8 → 2.3
                cond = 2.0 + 6.0 * t      # 2.0 → 8.0
            else:
                eff_dim = 2.6 + 0.05 * np.sin(t * 3)  # stable oscillation
                cond = 2.5 + 0.3 * np.sin(t * 2)
            
            speed = 0.5 + 0.3 * np.random.randn()
            curv = 1.0 + 0.5 * t + 0.2 * np.random.randn()
            
            rows.append({
                'cohort': eng,
                'signal_0_end': float(w * 24 + 24),
                'eigenvalue_1': 1.5 + 0.2 * np.random.randn(),
                'eigenvalue_2': 1.0 + 0.1 * np.random.randn(),
                'eigenvalue_3': 0.5 + 0.1 * np.random.randn(),
                'effective_dim': eff_dim,
                'total_variance': 3.0,
                'condition_number': cond,
                'effective_dim_velocity': -0.05 if collapsing else 0.01,
                'effective_dim_acceleration': 0.0,
                'effective_dim_curvature': 0.0,
                'speed': max(0.1, speed),
                'curvature': max(0.1, curv),
                'acceleration_magnitude': 0.5,
                'torsion': float('nan'),
                'arc_length': float(w) * 1.2,
            })
    
    return pl.DataFrame(rows)


def _make_synthetic_match(n_engines=10, collapse_engines=None):
    """Create synthetic trajectory match for testing."""
    collapse_engines = collapse_engines or []
    rows = []
    for i in range(n_engines):
        eng = f'engine_{i+1}'
        tid = 1 if eng in collapse_engines else 0
        rows.append({
            'cohort': eng,
            'trajectory_id': tid,
            'match_distance': 0.2 + 0.1 * np.random.rand(),
            'second_distance': 0.35,
            'match_confidence': 0.3 if eng not in collapse_engines else 0.6,
            'trajectory_position': 1.0,
            'n_windows': 8,
        })
    return pl.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestSingleAxisFeatures:

    def test_output_shape(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'test')
        assert len(feats) == 10
        assert 'cohort' in feats.columns
        assert f'test_n_windows' in feats.columns

    def test_all_metrics_present(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'ax')
        for metric in TRAJECTORY_METRICS:
            assert f'ax_{metric}_mean' in feats.columns, f"Missing {metric}_mean"
            assert f'ax_{metric}_delta' in feats.columns, f"Missing {metric}_delta"

    def test_collapse_detected_in_delta(self):
        collapse = ['engine_1', 'engine_2']
        sigs = _make_synthetic_sigs(collapse_engines=collapse)
        match = _make_synthetic_match(collapse_engines=collapse)
        feats = extract_single_axis_features(sigs, match, 'ax')
        
        for eng in collapse:
            row = feats.filter(pl.col('cohort') == eng)
            delta = row['ax_effective_dim_delta'][0]
            assert delta < -0.2, f"{eng} should collapse: delta={delta}"
        
        stable = feats.filter(~pl.col('cohort').is_in(collapse))
        for row in stable.iter_rows(named=True):
            assert abs(row['ax_effective_dim_delta']) < 0.2, \
                f"{row['cohort']} should be stable"

    def test_position_weighted_emphasizes_late(self):
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(n_engines=3, collapse_engines=collapse)
        match = _make_synthetic_match(n_engines=3, collapse_engines=collapse)
        feats = extract_single_axis_features(sigs, match, 'ax')
        
        row = feats.filter(pl.col('cohort') == 'engine_1')
        pw = row['ax_eff_dim_pos_weighted'][0]
        mean = row['ax_effective_dim_mean'][0]
        # Position-weighted should be lower than mean for collapsers
        # (late values weighted more, late values are lower)
        assert pw < mean, "Position-weighted should emphasize late (low) values"

    def test_quartile_ordering(self):
        sigs = _make_synthetic_sigs(n_engines=3, collapse_engines=['engine_1'])
        match = _make_synthetic_match(n_engines=3, collapse_engines=['engine_1'])
        feats = extract_single_axis_features(sigs, match, 'ax')
        
        row = feats.filter(pl.col('cohort') == 'engine_1')
        q1 = row['ax_effective_dim_q1'][0]
        q3 = row['ax_effective_dim_q3'][0]
        # For collapsing engine, q3 < q1 (dimension decreases)
        assert q3 < q1, f"Collapser q3={q3} should be < q1={q1}"

    def test_match_metadata_joined(self):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        feats = extract_single_axis_features(sigs, match, 'ax')
        assert 'ax_trajectory_id' in feats.columns
        assert 'ax_match_confidence' in feats.columns
        assert feats['ax_trajectory_id'].drop_nulls().len() == 10


class TestCrossAxisFeatures:

    def test_collapse_count(self):
        # Two axes: engine_1 collapses on both, engine_2 on one
        axis_a = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=5, collapse_engines=['engine_1', 'engine_2']),
            _make_synthetic_match(n_engines=5, collapse_engines=['engine_1', 'engine_2']),
            'axis_a',
        )
        axis_b = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=5, collapse_engines=['engine_1']),
            _make_synthetic_match(n_engines=5, collapse_engines=['engine_1']),
            'axis_b',
        )
        
        cross = extract_cross_axis_features({'axis_a': axis_a, 'axis_b': axis_b})
        
        eng1 = cross.filter(pl.col('cohort') == 'engine_1')
        assert eng1['cross_n_axes_collapsing'][0] == 2
        assert eng1['cross_collapse_on_all'][0] == 1
        
        eng2 = cross.filter(pl.col('cohort') == 'engine_2')
        assert eng2['cross_n_axes_collapsing'][0] == 1
        assert eng2['cross_collapse_on_all'][0] == 0

    def test_stable_engine_no_collapse(self):
        axis_a = extract_single_axis_features(
            _make_synthetic_sigs(n_engines=3),
            _make_synthetic_match(n_engines=3),
            'axis_a',
        )
        cross = extract_cross_axis_features({'axis_a': axis_a})
        
        for row in cross.iter_rows(named=True):
            assert row['cross_n_axes_collapsing'] == 0


class TestBuildFeatureMatrix:

    def test_full_pipeline(self, tmp_path):
        collapse = ['engine_1']
        sigs = _make_synthetic_sigs(collapse_engines=collapse)
        match = _make_synthetic_match(collapse_engines=collapse)
        
        sigs_path = tmp_path / 'sigs.parquet'
        match_path = tmp_path / 'match.parquet'
        sigs.write_parquet(str(sigs_path))
        match.write_parquet(str(match_path))
        
        features = build_feature_matrix(
            {'test_axis': {'sigs': str(sigs_path), 'match': str(match_path)}},
            verbose=False,
        )
        
        assert len(features) == 10
        assert 'cohort' in features.columns
        assert 'cross_n_axes_collapsing' in features.columns

    def test_zero_variance_dropped(self, tmp_path):
        sigs = _make_synthetic_sigs()
        match = _make_synthetic_match()
        sigs_path = tmp_path / 'sigs.parquet'
        match_path = tmp_path / 'match.parquet'
        sigs.write_parquet(str(sigs_path))
        match.write_parquet(str(match_path))
        
        features = build_feature_matrix(
            {'test': {'sigs': str(sigs_path), 'match': str(match_path)}},
            drop_zero_variance=True,
            verbose=False,
        )
        
        # trajectory_position should be dropped (always 1.0)
        assert 'test_trajectory_position' not in features.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

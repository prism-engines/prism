#!/usr/bin/env python3
"""
C-MAPSS: Evaluate RUL with PRISM vectors + geometry + modes + affinity + wavelet.

PR #6 Implementation: Affinity-Weighted Modes and Wavelet Microscope

Combines:
1. Signal vector metrics (Hurst, entropy, etc.)
2. Cohort geometry (PCA, distance, clustering)
3. Pairwise geometry summary (correlation, mutual information)
4. Mode discovery features (n_modes, affinity, entropy)
5. NEW: Affinity-weighted mode features (weighted means, variance, contrasts)
6. NEW: Wavelet microscope features (frequency-band degradation)

Target: RMSE < 6.2 (improvement from v3's 6.47)
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import sys

warnings.filterwarnings('ignore')

# Add prism to path
sys.path.insert(0, '/Users/jasonrudder/prism-mac')

from prism.modules.modes import compute_affinity_weighted_features
from prism.modules.wavelet_microscope import run_wavelet_microscope, extract_wavelet_features

# =============================================================================
# CONFIG
# =============================================================================

DOMAIN = "cmapss_fd001"
DATA_DIR = Path("/Users/jasonrudder/prism-mac/data") / DOMAIN
SOURCE_DIR = Path("/Users/jasonrudder/prism-mac/data/CMAPSSData")

RUL_CAP = 125

BENCHMARKS = {
    'PHM08_Winner': 12.4,
    'DCNN': 12.61,
    'Bi-LSTM': 17.60,
    'LightGBM': 6.62,
    'PRISM_v1': 9.47,
    'PRISM_v2_geo': 7.01,
    'PRISM_v3_modes': 6.47,
}

KEY_METRICS = [
    'hurst_exponent', 'sample_entropy', 'permutation_entropy',
    'realized_vol', 'skewness', 'kurtosis', 'signal_to_noise',
    'rqa_determinism', 'rqa_recurrence_rate', 'rqa_laminarity',
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_prism_vectors() -> pl.DataFrame:
    """Load PRISM signal vectors."""
    vectors = pl.read_parquet(DATA_DIR / 'vector' / 'signal.parquet')
    vectors = vectors.with_columns([
        pl.col('signal_id').str.extract(r'u(\d+)_').cast(pl.Int64).alias('unit'),
        pl.col('signal_id').str.extract(r'_(s\d+)$').alias('sensor'),
    ])
    return vectors


def load_observations() -> pl.DataFrame:
    """Load raw observations for wavelet analysis."""
    return pl.read_parquet(DATA_DIR / 'raw' / 'observations.parquet')


def load_cohort_geometry() -> pl.DataFrame:
    """Load cohort geometry."""
    return pl.read_parquet(DATA_DIR / 'geometry' / 'cohort.parquet')


def load_pairwise_summary() -> pl.DataFrame:
    """Load pairwise geometry summary."""
    return pl.read_parquet(DATA_DIR / 'geometry' / 'cohort_pairwise_summary.parquet')


def load_cohort_modes() -> pl.DataFrame:
    """Load mode discovery results."""
    return pl.read_parquet(DATA_DIR / 'geometry' / 'cohort_modes.parquet')


def load_ground_truth() -> pd.DataFrame:
    """Load C-MAPSS with ground truth RUL."""
    ALL_COLUMNS = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
    train_path = SOURCE_DIR / "train_FD001.txt"
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=ALL_COLUMNS)
    max_cycles = df.groupby('unit')['cycle'].max()
    df = df.merge(max_cycles.rename('max_cycle'), left_on='unit', right_index=True)
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=RUL_CAP)
    return df


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_vector_features(vectors: pl.DataFrame) -> pd.DataFrame:
    """Extract per-engine features from PRISM vectors."""
    print("  Extracting vector features...")
    features_list = []
    units = sorted(vectors['unit'].drop_nulls().unique().to_list())

    for unit in units:
        unit_data = vectors.filter(pl.col('unit') == unit)
        obs_dates = unit_data['obs_date'].unique().sort().to_list()

        for obs_date in obs_dates:
            cycle_data = unit_data.filter(pl.col('obs_date') == obs_date)
            feat = {'unit': unit, 'obs_date': obs_date}

            for metric in KEY_METRICS:
                metric_vals = cycle_data.filter(
                    pl.col('metric_name') == metric
                )['metric_value'].drop_nulls().to_list()

                if metric_vals:
                    feat[f'{metric}_mean'] = np.mean(metric_vals)
                    feat[f'{metric}_std'] = np.std(metric_vals)
                    feat[f'{metric}_min'] = np.min(metric_vals)
                    feat[f'{metric}_max'] = np.max(metric_vals)
                    feat[f'{metric}_range'] = np.max(metric_vals) - np.min(metric_vals)

            features_list.append(feat)

    return pd.DataFrame(features_list)


def add_geometry_features(features_df: pd.DataFrame, cohort_geo: pl.DataFrame, pairwise_summary: pl.DataFrame) -> pd.DataFrame:
    """Add cohort geometry and pairwise summary features."""
    print("  Adding geometry features...")

    cohort_geo_pd = cohort_geo.to_pandas()
    cohort_geo_pd['unit'] = cohort_geo_pd['cohort_id'].str.extract(r'u(\d+)').astype(int)

    geo_cols = [
        'pca_var_pc1', 'pca_var_pc2', 'pca_var_pc3', 'pca_cumulative_3',
        'pca_effective_dim', 'pca_n_components_90',
        'distance_mean', 'distance_std', 'distance_cohesion',
        'clustering_silhouette', 'mst_total_weight', 'mst_avg_degree',
        'lof_mean_score', 'lof_n_outliers', 'lof_outlier_ratio',
    ]

    geo_to_merge = cohort_geo_pd[['unit'] + [c for c in geo_cols if c in cohort_geo_pd.columns]]
    features_df = features_df.merge(geo_to_merge, on='unit', how='left')

    pairwise_pd = pairwise_summary.to_pandas()
    pairwise_pd['unit'] = pairwise_pd['cohort_id'].str.extract(r'u(\d+)').astype(int)

    pairwise_cols = {
        'mean_correlation': 'pairwise_mean_corr',
        'std_correlation': 'pairwise_std_corr',
        'mean_distance': 'pairwise_mean_dist',
        'mean_mi': 'pairwise_mean_mi',
        'mean_kendall': 'pairwise_mean_kendall',
    }
    pairwise_pd = pairwise_pd.rename(columns=pairwise_cols)

    pairwise_to_merge = pairwise_pd[['unit'] + list(pairwise_cols.values())]
    features_df = features_df.merge(pairwise_to_merge, on='unit', how='left')

    return features_df


def add_mode_features(features_df: pd.DataFrame, modes_df: pl.DataFrame) -> pd.DataFrame:
    """Add basic mode discovery features (v3)."""
    print("  Adding mode features...")

    modes_pd = modes_df.to_pandas()
    modes_pd['unit'] = modes_pd['cohort_id'].str.extract(r'u(\d+)').astype(int)

    mode_stats = modes_pd.groupby('unit').agg({
        'n_modes': 'first',
        'mode_affinity': ['mean', 'std', 'min'],
        'mode_entropy': ['mean', 'std', 'max'],
        'mode_id': 'nunique',
    }).reset_index()

    mode_stats.columns = ['unit', 'n_modes', 'mode_affinity_mean', 'mode_affinity_std',
                          'mode_affinity_min', 'mode_entropy_mean', 'mode_entropy_std',
                          'mode_entropy_max', 'n_modes_used']

    mode_stats['mode_concentration'] = mode_stats['n_modes_used'] / mode_stats['n_modes']

    fingerprint_cols = [c for c in modes_pd.columns if c.startswith('fingerprint_')]
    for col in fingerprint_cols:
        short_name = col.replace('fingerprint_', 'fp_')
        mode_stats[f'{short_name}_cohort_mean'] = modes_pd.groupby('unit')[col].mean().values
        mode_stats[f'{short_name}_cohort_std'] = modes_pd.groupby('unit')[col].std().values

    features_df = features_df.merge(mode_stats, on='unit', how='left')

    return features_df


def add_affinity_weighted_features(features_df: pd.DataFrame, modes_df: pl.DataFrame) -> pd.DataFrame:
    """Add affinity-weighted mode features (PR #6)."""
    print("  Adding affinity-weighted features...")

    cohorts = modes_df['cohort_id'].unique().to_list()

    affinity_features = []
    for cohort_id in cohorts:
        af = compute_affinity_weighted_features(modes_df, None, cohort_id)
        affinity_features.append(af)

    affinity_df = pd.DataFrame(affinity_features)
    affinity_df['unit'] = affinity_df['cohort_id'].str.extract(r'u(\d+)').astype(int)

    # Drop cohort_id to avoid conflict
    affinity_df = affinity_df.drop(columns=['cohort_id'])

    # Select only numeric columns and unit
    numeric_cols = affinity_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'unit' not in numeric_cols:
        numeric_cols = ['unit'] + numeric_cols

    affinity_df = affinity_df[numeric_cols]

    features_df = features_df.merge(affinity_df, on='unit', how='left')

    return features_df


def add_wavelet_features(features_df: pd.DataFrame, observations: pl.DataFrame) -> pd.DataFrame:
    """Add wavelet microscope features (PR #6)."""
    print("  Adding wavelet features...")

    # Get unique cohorts
    units = features_df['unit'].unique()

    wavelet_features = []
    for unit in units:
        cohort_id = f'u{unit:03d}'

        # Run wavelet microscope
        wavelet_results = run_wavelet_microscope(
            observations, cohort_id,
            top_n_snr_variance=5,
            window_size=21
        )

        # Extract features
        wf = extract_wavelet_features(wavelet_results, cohort_id)
        wf['unit'] = unit
        wavelet_features.append(wf)

    if not wavelet_features:
        return features_df

    wavelet_df = pd.DataFrame(wavelet_features)

    # Drop cohort_id to avoid conflict
    if 'cohort_id' in wavelet_df.columns:
        wavelet_df = wavelet_df.drop(columns=['cohort_id'])

    # Select only numeric columns and unit
    numeric_cols = wavelet_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'unit' not in numeric_cols:
        numeric_cols = ['unit'] + numeric_cols

    wavelet_df = wavelet_df[numeric_cols]

    features_df = features_df.merge(wavelet_df, on='unit', how='left')

    return features_df


def merge_with_rul(features_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PRISM features with ground truth RUL."""
    base_date = datetime(2000, 1, 1)
    features_df['cycle'] = features_df['obs_date'].apply(
        lambda x: (pd.Timestamp(x) - pd.Timestamp(base_date)).days
    )

    merged = features_df.merge(
        truth_df[['unit', 'cycle', 'RUL']],
        on=['unit', 'cycle'],
        how='left'
    )
    merged = merged.dropna(subset=['RUL'])
    return merged


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rul(merged: pd.DataFrame) -> dict:
    """Train and evaluate RUL prediction."""
    print("\nTraining RUL predictor...")

    feature_cols = [c for c in merged.columns if c not in ['unit', 'cycle', 'obs_date', 'RUL']]
    feature_cols = [c for c in feature_cols if merged[c].notna().sum() > 0]

    # Remove any string columns
    numeric_cols = []
    for c in feature_cols:
        if merged[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
            numeric_cols.append(c)

    feature_cols = numeric_cols

    print(f"  Feature columns: {len(feature_cols)}")

    # Split
    test_idx = merged.groupby('unit')['cycle'].idxmax()
    test_df = merged.loc[test_idx]
    train_df = merged.drop(test_idx)

    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['RUL'].values
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['RUL'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled).clip(0, RUL_CAP)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'rmse': rmse,
        'importance': importance,
        'test_df': test_df,
        'y_pred': y_pred,
        'y_test': y_test,
        'n_features': len(feature_cols),
    }


def print_comparison(rmse: float):
    """Print benchmark comparison."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"\n  PRISM v4 (Affinity+Wavelet) RMSE: {rmse:.2f}")
    print()

    for name, bench_rmse in sorted(BENCHMARKS.items(), key=lambda x: x[1]):
        diff = ((bench_rmse - rmse) / bench_rmse) * 100
        status = "BEAT" if rmse < bench_rmse else "    "
        print(f"  vs {name:<18}: {bench_rmse:>6.2f}  {status} ({diff:+.1f}%)")


def print_feature_category_importance(importance: pd.DataFrame):
    """Print importance by feature category."""
    print("\n" + "=" * 60)
    print("FEATURE CATEGORY IMPORTANCE")
    print("=" * 60)

    categories = {
        'Vector': lambda x: any(m in x for m in KEY_METRICS) and not x.startswith('fp_') and not x.startswith('trans_') and not x.startswith('m'),
        'Geometry': lambda x: any(g in x for g in ['pca_', 'distance_', 'clustering_', 'mst_', 'lof_', 'pairwise_']),
        'Mode (v3)': lambda x: x.startswith('fp_') or x in ['n_modes', 'mode_affinity_mean', 'mode_entropy_mean', 'mode_concentration'],
        'Affinity (v4)': lambda x: x.startswith('aff_') or x.startswith('m0_') or x.startswith('m1_') or x.startswith('m2_') or x.startswith('m3_') or x.startswith('contrast_') or x.startswith('trans_') or x.startswith('stable_') or 'transitioning' in x,
        'Wavelet (v4)': lambda x: 'wavelet' in x.lower(),
    }

    totals = {}
    for cat, pred in categories.items():
        cat_features = importance[importance['feature'].apply(pred)]
        totals[cat] = cat_features['importance'].sum()

    for cat, total in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20}: {total*100:>6.2f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("C-MAPSS FD001 - PRISM v4 RUL EVALUATION")
    print("(Affinity-Weighted Modes + Wavelet Microscope)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    vectors = load_prism_vectors()
    observations = load_observations()
    cohort_geo = load_cohort_geometry()
    pairwise_summary = load_pairwise_summary()
    cohort_modes = load_cohort_modes()
    truth = load_ground_truth()

    print(f"  Vector rows: {len(vectors):,}")
    print(f"  Observation rows: {len(observations):,}")
    print(f"  Cohort geometry rows: {len(cohort_geo)}")
    print(f"  Pairwise summary rows: {len(pairwise_summary)}")
    print(f"  Mode assignments: {len(cohort_modes)}")

    # Extract features
    print("\n[2] Extracting features...")
    features = extract_vector_features(vectors)
    features = add_geometry_features(features, cohort_geo, pairwise_summary)
    features = add_mode_features(features, cohort_modes)
    features = add_affinity_weighted_features(features, cohort_modes)
    features = add_wavelet_features(features, observations)

    print(f"  Total features: {len(features.columns) - 2}")

    # Merge with RUL
    print("\n[3] Merging with RUL...")
    merged = merge_with_rul(features, truth)
    print(f"  Merged rows: {len(merged):,}")

    # Evaluate
    print("\n[4] Evaluating...")
    results = evaluate_rul(merged)

    # Results
    print("\n" + "=" * 60)
    print("TOP 25 FEATURES")
    print("=" * 60)
    print(results['importance'].head(25).to_string(index=False))

    # New v4 features specifically
    print("\n" + "=" * 60)
    print("NEW v4 FEATURES (Affinity + Wavelet)")
    print("=" * 60)
    v4_features = results['importance'][
        results['importance']['feature'].apply(
            lambda x: x.startswith('aff_') or x.startswith('m0_') or x.startswith('m1_') or
                     x.startswith('m2_') or x.startswith('m3_') or x.startswith('contrast_') or
                     x.startswith('trans_') or x.startswith('stable_') or 'transitioning' in x or
                     'wavelet' in x.lower()
        )
    ]
    if len(v4_features) > 0:
        print(v4_features.head(20).to_string(index=False))
    else:
        print("  No v4 features in importance (all zero)")

    print_feature_category_importance(results['importance'])
    print_comparison(results['rmse'])

    # Save
    output_path = DATA_DIR / 'rul_results_v4.parquet'
    results_df = pd.DataFrame({
        'unit': results['test_df']['unit'].values,
        'y_true': results['y_test'],
        'y_pred': results['y_pred'],
    })
    results_df.to_parquet(output_path)
    print(f"\nSaved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features: {results['n_features']}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  vs v3 (6.47): {((6.47 - results['rmse']) / 6.47) * 100:+.1f}%")
    print(f"  vs LightGBM (6.62): {((6.62 - results['rmse']) / 6.62) * 100:+.1f}%")


if __name__ == '__main__':
    main()

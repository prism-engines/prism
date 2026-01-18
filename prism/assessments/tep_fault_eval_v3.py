"""
TEP Fault Classification V3 - Focused Improvements
===================================================

Changes from V1:
1. Add Laplace field features (proven signal)
2. Better tuned classifiers
3. NO rolling features (added noise in V2)
4. Feature selection to reduce overfitting

Usage:
    python -m prism.assessments.tep_fault_eval_v3 --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    IsolationForest,
    ExtraTreesClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
import os

warnings.filterwarnings('ignore')


def load_tep_data(domain: str):
    """Load TEP vector, field, and observation data."""
    from prism.db.parquet_store import get_parquet_path

    vec_path = get_parquet_path('vector', 'signal', domain)
    field_path = get_parquet_path('vector', 'signal_field', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    vec_df = pl.read_parquet(vec_path)
    obs_df = pl.read_parquet(obs_path)

    field_df = None
    if field_path.exists():
        field_df = pl.read_parquet(field_path)

    return vec_df, field_df, obs_df


def build_feature_matrix(vec_df: pl.DataFrame, key_metrics: list = None):
    """Build feature matrix from PRISM vector data (V1 approach)."""

    if key_metrics is None:
        key_metrics = [
            'permutation_entropy', 'sample_entropy',
            'hurst_exponent', 'persistence',
            'lyapunov_exponent', 'is_chaotic',
            'realized_vol', 'skewness', 'kurtosis', 'max_drawdown',
            'determinism', 'laminarity', 'recurrence_rate', 'entropy',
            'alpha', 'beta', 'omega', 'unconditional_vol',
            'dominant_period', 'spectral_entropy', 'spectral_slope',
            'hilbert_amp_cv', 'hilbert_inst_freq_mean',
            'break_n', 'break_rate', 'break_is_accelerating',
            'dirac_n_impulses', 'heaviside_n_steps',
            'vector_score', 'score_entropy', 'score_hurst',
        ]

    # Filter to TEP process signals (exclude FAULT)
    process_df = vec_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    # Filter to key metrics
    filtered = process_df.filter(pl.col('metric_name').is_in(key_metrics))

    # Aggregate by date - mean and std across all signals
    agg_df = filtered.group_by(['obs_date', 'engine', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('mean'),
        pl.col('metric_value').std().alias('std'),
        pl.col('metric_value').min().alias('min'),
        pl.col('metric_value').max().alias('max'),
    ])

    # Create feature names
    agg_df = agg_df.with_columns([
        (pl.col('engine') + '_' + pl.col('metric_name')).alias('base_name')
    ])

    # Pivot each statistic
    features = None
    for stat in ['mean', 'std', 'min', 'max']:
        pivot = agg_df.select([
            'obs_date',
            (pl.col('base_name') + f'_{stat}').alias('feature'),
            pl.col(stat).alias('value')
        ]).pivot(on='feature', index='obs_date', values='value')

        if features is None:
            features = pivot
        else:
            features = features.join(pivot, on='obs_date')

    return features


def build_field_features(field_df: pl.DataFrame) -> pl.DataFrame:
    """Build features from Laplace field data."""
    if field_df is None:
        return None

    # Filter to TEP
    tep_field = field_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    if len(tep_field) == 0:
        return None

    # Key Laplace field metrics
    field_metrics = [
        'gradient_mean', 'gradient_std', 'gradient_magnitude',
        'laplacian_mean', 'laplacian_std',
        'divergence', 'total_potential', 'n_inflections'
    ]

    available = [c for c in field_metrics if c in tep_field.columns]
    if not available:
        return None

    date_col = 'window_end' if 'window_end' in tep_field.columns else 'obs_date'

    agg_exprs = []
    for metric in available:
        agg_exprs.extend([
            pl.col(metric).mean().alias(f'field_{metric}_mean'),
            pl.col(metric).std().alias(f'field_{metric}_std'),
            pl.col(metric).max().alias(f'field_{metric}_max'),
        ])

    field_agg = tep_field.group_by(date_col).agg(agg_exprs)
    return field_agg.rename({date_col: 'obs_date'})


def get_fault_labels(obs_df: pl.DataFrame):
    """Extract fault labels per date."""
    fault_raw = obs_df.filter(pl.col('signal_id') == 'TEP_FAULT')
    labels = fault_raw.group_by('obs_date').agg([
        pl.col('value').mode().first().alias('fault_code'),
    ])
    return labels


def run_evaluation(domain: str):
    """Run focused TEP fault classification evaluation."""

    print("=" * 100)
    print("TEP FAULT CLASSIFICATION V3 - FOCUSED IMPROVEMENTS")
    print("=" * 100)
    print()
    print("Changes from V1:")
    print("  1. Add Laplace field features")
    print("  2. Tuned classifiers (more trees, better params)")
    print("  3. Feature selection (top K features)")
    print("  4. NO rolling features (caused overfitting)")
    print()

    # Load data
    print("Loading data...")
    vec_df, field_df, obs_df = load_tep_data(domain)

    # Build features (V1 approach)
    print("Building vector features (V1 approach)...")
    features = build_feature_matrix(vec_df)
    n_vec = len(features.columns) - 1
    print(f"  Vector features: {n_vec}")

    # Add field features
    print("Building Laplace field features...")
    field_features = build_field_features(field_df)
    if field_features is not None:
        # Cast dates to match
        if features['obs_date'].dtype != field_features['obs_date'].dtype:
            field_features = field_features.with_columns(
                pl.col('obs_date').cast(features['obs_date'].dtype)
            )
        features = features.join(field_features, on='obs_date', how='left')
        n_field = len(field_features.columns) - 1
        print(f"  Added field features: {n_field}")
    else:
        print("  No field features available")

    # Get labels
    labels = get_fault_labels(obs_df)

    # Join
    data = features.join(labels, on='obs_date', how='inner')
    print(f"\nSamples: {len(data)}")

    # Convert to numpy
    pdf = data.to_pandas()
    feature_cols = [c for c in pdf.columns if c not in ['obs_date', 'fault_code']]
    X = pdf[feature_cols].values
    y = pdf['fault_code'].values.astype(int)

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Feature selection - keep top 80 features
    print("\nSelecting top 80 features...")
    selector = SelectKBest(f_classif, k=min(80, X.shape[1]))
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    print(f"  Selected {len(selected_features)} features")

    # =========================================================================
    # EVALUATION 1: BINARY CLASSIFICATION
    # =========================================================================
    print()
    print("=" * 100)
    print("EVALUATION 1: BINARY CLASSIFICATION (Normal vs Fault)")
    print("=" * 100)

    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    # Tuned Gradient Boosting
    clf_bin = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    clf_bin.fit(X_train_sel, y_train_bin)
    y_pred_bin = clf_bin.predict(X_test_sel)
    y_prob_bin = clf_bin.predict_proba(X_test_sel)[:, 1]

    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    f1_bin = f1_score(y_test_bin, y_pred_bin)
    auc_bin = roc_auc_score(y_test_bin, y_prob_bin)

    print(f"\nAccuracy: {acc_bin:.1%}")
    print(f"F1 Score: {f1_bin:.3f}")
    print(f"AUC-ROC:  {auc_bin:.3f}")
    print()
    print(classification_report(y_test_bin, y_pred_bin, target_names=['Normal', 'Fault']))

    # =========================================================================
    # EVALUATION 2: MULTI-CLASS CLASSIFICATION
    # =========================================================================
    print("=" * 100)
    print("EVALUATION 2: MULTI-CLASS FAULT IDENTIFICATION")
    print("=" * 100)

    # Extra Trees with tuned params
    clf_multi = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    clf_multi.fit(X_train_sel, y_train)
    y_pred_multi = clf_multi.predict(X_test_sel)

    acc_multi = accuracy_score(y_test, y_pred_multi)

    print(f"\nOverall Accuracy: {acc_multi:.1%}")
    print("\nPer-Fault Accuracy:")
    print("-" * 50)

    for fid in sorted(np.unique(y_test)):
        mask = y_test == fid
        if mask.sum() >= 1:
            class_acc = (y_pred_multi[mask] == y_test[mask]).mean()
            label = "Normal" if fid == 0 else f"IDV{fid:02d}"
            n = mask.sum()
            status = "+" if class_acc >= 0.8 else "~" if class_acc >= 0.5 else "-"
            print(f"  {status} {label:8s}: {class_acc:5.1%} (n={n})")

    # =========================================================================
    # EVALUATION 3: ANOMALY DETECTION
    # =========================================================================
    print()
    print("=" * 100)
    print("EVALUATION 3: ANOMALY DETECTION (Train on Normal Only)")
    print("=" * 100)

    normal_mask = y_train == 0
    X_train_normal = X_train_sel[normal_mask]

    clf_anom = IsolationForest(
        n_estimators=300,
        contamination=0.08,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    clf_anom.fit(X_train_normal)

    y_pred_anom = clf_anom.predict(X_test_sel)
    y_pred_anom_bin = (y_pred_anom == -1).astype(int)
    y_test_anom_bin = (y_test > 0).astype(int)

    acc_anom = accuracy_score(y_test_anom_bin, y_pred_anom_bin)
    f1_anom = f1_score(y_test_anom_bin, y_pred_anom_bin)

    print(f"\nAccuracy: {acc_anom:.1%}")
    print(f"F1 Score: {f1_anom:.3f}")

    # =========================================================================
    # TOP FEATURES
    # =========================================================================
    print()
    print("=" * 100)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 100)
    print()

    imp = clf_multi.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]

    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {selected_features[idx]:50s}: {imp[idx]:.4f}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print()
    print("=" * 100)
    print("VERSION COMPARISON")
    print("=" * 100)
    print()
    print("                        V1          V2          V3")
    print("-" * 60)
    print(f"Binary Accuracy:      64.5%       60.0%      {acc_bin:.1%}")
    print(f"Multi-class Accuracy: 60.0%       57.3%      {acc_multi:.1%}")
    print(f"Anomaly Accuracy:     55.5%       50.0%      {acc_anom:.1%}")
    print()
    print("=" * 100)

    return {
        'binary_acc': acc_bin,
        'multiclass_acc': acc_multi,
        'anomaly_acc': acc_anom
    }


def main():
    parser = argparse.ArgumentParser(description='TEP Fault Classification V3')
    parser.add_argument('--domain', type=str, default=None)
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for TEP evaluation")
    os.environ["PRISM_DOMAIN"] = domain

    run_evaluation(domain)


if __name__ == '__main__':
    main()

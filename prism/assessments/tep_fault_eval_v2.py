"""
TEP Fault Classification Evaluation V2 - Enhanced
==================================================

Improvements over V1:
1. Focus on top-performing features (GARCH, spectral, break detector)
2. Add Laplace field features (gradient, divergence, laplacian)
3. Add rolling window / rate-of-change features
4. Tuned classifier hyperparameters
5. XGBoost for better performance

Usage:
    python -m prism.assessments.tep_fault_eval_v2 --domain cheme
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
    AdaBoostClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# TOP PERFORMING FEATURES (from V1 analysis)
# =============================================================================

TOP_METRICS = [
    # GARCH (volatility dynamics) - TOP PERFORMERS
    'alpha', 'beta', 'omega', 'unconditional_vol',
    # Spectral (frequency domain) - TOP PERFORMERS
    'spectral_slope', 'spectral_entropy', 'dominant_period',
    # Break detector - TOP PERFORMERS
    'break_n', 'break_rate', 'break_is_accelerating',
    # Hilbert (amplitude/phase)
    'hilbert_amp_cv', 'hilbert_inst_freq_mean',
    # Entropy
    'permutation_entropy', 'sample_entropy',
    # Hurst / persistence
    'hurst_exponent', 'persistence',
    # RQA (recurrence)
    'determinism', 'laminarity', 'recurrence_rate',
    # Realized vol
    'realized_vol', 'kurtosis', 'skewness',
    # Lyapunov
    'lyapunov_exponent',
    # Vector score
    'vector_score',
]


def load_tep_data(domain: str):
    """Load TEP vector, field, and observation data."""
    from prism.db.parquet_store import get_parquet_path

    vec_path = get_parquet_path('vector', 'signal', domain)
    field_path = get_parquet_path('vector', 'signal_field', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    vec_df = pl.read_parquet(vec_path)
    obs_df = pl.read_parquet(obs_path)

    # Load field data if exists
    field_df = None
    if field_path.exists():
        field_df = pl.read_parquet(field_path)

    return vec_df, field_df, obs_df


def build_vector_features(vec_df: pl.DataFrame) -> pl.DataFrame:
    """Build feature matrix from PRISM vector data."""

    # Filter to TEP process signals (exclude FAULT)
    process_df = vec_df.filter(
        pl.col('signal_id').str.starts_with('TEP_') &
        ~pl.col('signal_id').str.contains('FAULT')
    )

    # Filter to top metrics
    filtered = process_df.filter(pl.col('metric_name').is_in(TOP_METRICS))

    # Aggregate by date - mean, std, min, max, range
    agg_df = filtered.group_by(['obs_date', 'engine', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('mean'),
        pl.col('metric_value').std().alias('std'),
        pl.col('metric_value').min().alias('min'),
        pl.col('metric_value').max().alias('max'),
        (pl.col('metric_value').max() - pl.col('metric_value').min()).alias('range'),
        pl.col('metric_value').median().alias('median'),
    ])

    # Create feature names
    agg_df = agg_df.with_columns([
        (pl.col('engine') + '_' + pl.col('metric_name')).alias('base_name')
    ])

    # Pivot each statistic
    features = None
    for stat in ['mean', 'std', 'min', 'max', 'range', 'median']:
        pivot = agg_df.select([
            'obs_date',
            (pl.col('base_name') + f'_{stat}').alias('feature'),
            pl.col(stat).alias('value')
        ]).pivot(on='feature', index='obs_date', values='value')

        if features is None:
            features = pivot
        else:
            # Drop obs_date from pivot before joining to avoid duplicates
            pivot_cols = [c for c in pivot.columns if c != 'obs_date']
            features = features.join(
                pivot.select(['obs_date'] + pivot_cols),
                on='obs_date',
                how='outer',
                coalesce=True
            )

    return features.sort('obs_date')


def build_field_features(field_df: pl.DataFrame) -> pl.DataFrame:
    """Build features from Laplace field data (gradient, divergence, etc.)."""
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

    # Check which columns exist
    available = [c for c in field_metrics if c in tep_field.columns]

    if not available:
        return None

    # Aggregate by window_end (date)
    date_col = 'window_end' if 'window_end' in tep_field.columns else 'obs_date'

    agg_exprs = []
    for metric in available:
        agg_exprs.extend([
            pl.col(metric).mean().alias(f'field_{metric}_mean'),
            pl.col(metric).std().alias(f'field_{metric}_std'),
            pl.col(metric).max().alias(f'field_{metric}_max'),
            pl.col(metric).min().alias(f'field_{metric}_min'),
        ])

    field_agg = tep_field.group_by(date_col).agg(agg_exprs)

    return field_agg.rename({date_col: 'obs_date'})


def add_rolling_features(features: pl.DataFrame, windows: list = [3, 7, 14]) -> pl.DataFrame:
    """Add rolling window features (rate of change, momentum)."""

    features = features.sort('obs_date')

    # Get numeric columns (exclude date)
    numeric_cols = [c for c in features.columns if c != 'obs_date']

    # Sample a subset of important features for rolling
    important_patterns = ['garch_', 'spectral_', 'break_', 'entropy_']
    rolling_cols = [c for c in numeric_cols if any(p in c.lower() for p in important_patterns)][:20]

    for window in windows:
        for col in rolling_cols:
            # Rolling mean
            features = features.with_columns([
                pl.col(col).rolling_mean(window_size=window).alias(f'{col}_roll{window}_mean')
            ])
            # Rate of change (diff)
            features = features.with_columns([
                pl.col(col).diff(n=window).alias(f'{col}_roc{window}')
            ])

    return features


def get_fault_labels(obs_df: pl.DataFrame):
    """Extract fault labels per date."""
    fault_raw = obs_df.filter(pl.col('signal_id') == 'TEP_FAULT')

    # Get dominant fault per date
    labels = fault_raw.group_by('obs_date').agg([
        pl.col('value').mode().first().alias('fault_code'),
        pl.col('value').n_unique().alias('n_fault_types'),
    ])

    return labels


def run_binary_classification(X_train, X_test, y_train, y_test, feature_cols):
    """Binary classification: Normal vs Any Fault with tuned params."""
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    # Tuned Gradient Boosting
    clf = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    clf.fit(X_train, y_train_bin)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    auc = roc_auc_score(y_test_bin, y_prob)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, np.vstack([X_train, X_test]),
                                 np.hstack([y_train_bin, y_test_bin]),
                                 cv=cv, scoring='accuracy')

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'clf': clf,
        'y_pred': y_pred,
        'y_true': y_test_bin,
        'report': classification_report(y_test_bin, y_pred, target_names=['Normal', 'Fault'])
    }


def run_multiclass_classification(X_train, X_test, y_train, y_test, feature_cols):
    """Multi-class: Identify specific fault type with tuned params."""

    # Extra Trees often outperforms RF on noisy data
    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Per-class accuracy
    per_class = {}
    for fid in sorted(np.unique(y_test)):
        mask = y_test == fid
        if mask.sum() >= 1:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            label = "Normal" if fid == 0 else f"IDV{fid:02d}"
            per_class[label] = {'accuracy': class_acc, 'n': mask.sum()}

    # Feature importance
    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:30]
    top_features = [(feature_cols[i], imp[i]) for i in top_idx]

    return {
        'accuracy': acc,
        'per_class': per_class,
        'clf': clf,
        'y_pred': y_pred,
        'y_true': y_test,
        'feature_importance': imp,
        'top_features': top_features
    }


def run_anomaly_detection(X_train_normal, X_test, y_test):
    """Anomaly detection: Train on normal, detect faults."""
    clf = IsolationForest(
        n_estimators=500,
        contamination=0.05,  # Lower contamination
        max_samples='auto',
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_normal)

    # Predict: -1 = anomaly, 1 = normal
    y_pred = clf.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)  # 1 = anomaly/fault

    y_test_binary = (y_test > 0).astype(int)

    acc = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)

    # Also compute scores for ROC
    scores = -clf.decision_function(X_test)  # Higher = more anomalous
    try:
        auc = roc_auc_score(y_test_binary, scores)
    except:
        auc = 0.5

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'clf': clf,
        'y_pred': y_pred_binary,
        'y_true': y_test_binary
    }


def run_evaluation(domain: str):
    """Run enhanced TEP fault classification evaluation."""

    print("=" * 100)
    print("TEP FAULT CLASSIFICATION V2 - ENHANCED EVALUATION")
    print("=" * 100)
    print()
    print("Improvements:")
    print("  1. Focus on top-performing features (GARCH, spectral, break)")
    print("  2. Laplace field features (gradient, divergence)")
    print("  3. Rolling window / rate-of-change features")
    print("  4. Tuned classifier hyperparameters")
    print("  5. Extra Trees + higher n_estimators")
    print()

    # Load data
    print("Loading data...")
    vec_df, field_df, obs_df = load_tep_data(domain)

    # Build vector features
    print("Building vector features...")
    features = build_vector_features(vec_df)
    print(f"  Vector features: {len(features.columns) - 1}")

    # Build field features
    print("Building Laplace field features...")
    field_features = build_field_features(field_df)
    if field_features is not None:
        features = features.join(field_features, on='obs_date', how='left')
        print(f"  Added field features: {len(field_features.columns) - 1}")
    else:
        print("  No field features available")

    # Add rolling features
    print("Adding rolling window features...")
    n_before = len(features.columns)
    features = add_rolling_features(features, windows=[3, 7])
    print(f"  Added rolling features: {len(features.columns) - n_before}")

    # Get labels
    labels = get_fault_labels(obs_df)

    # Debug: check date types
    print(f"  Features dates: {features['obs_date'].dtype}, range: {features['obs_date'].min()} to {features['obs_date'].max()}")
    print(f"  Labels dates: {labels['obs_date'].dtype}, range: {labels['obs_date'].min()} to {labels['obs_date'].max()}")

    # Ensure date types match
    if features['obs_date'].dtype != labels['obs_date'].dtype:
        labels = labels.with_columns(pl.col('obs_date').cast(features['obs_date'].dtype))

    # Join - inner first, then fill nulls instead of dropping all
    data = features.join(labels, on='obs_date', how='inner')
    print(f"  After join: {len(data)} rows")

    # Only drop rows with null fault_code, not all nulls
    data = data.filter(pl.col('fault_code').is_not_null())
    print(f"\nTotal samples: {len(data)}")

    # Convert to numpy
    pdf = data.to_pandas()
    feature_cols = [c for c in pdf.columns if c not in ['obs_date', 'fault_code', 'n_fault_types']]
    X = pdf[feature_cols].values
    y = pdf['fault_code'].values.astype(int)

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        label = "Normal" if u == 0 else f"IDV{u:02d}"
        print(f"  {label}: {c}")
    print()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # =========================================================================
    # EVALUATION 1: BINARY CLASSIFICATION
    # =========================================================================
    print("=" * 100)
    print("EVALUATION 1: BINARY CLASSIFICATION (Normal vs Fault)")
    print("=" * 100)

    binary_results = run_binary_classification(X_train_s, X_test_s, y_train, y_test, feature_cols)

    print(f"\nAccuracy:     {binary_results['accuracy']:.1%}")
    print(f"F1 Score:     {binary_results['f1']:.3f}")
    print(f"AUC-ROC:      {binary_results['auc']:.3f}")
    print(f"CV Accuracy:  {binary_results['cv_mean']:.1%} (+/- {binary_results['cv_std']:.1%})")
    print()
    print(binary_results['report'])

    # =========================================================================
    # EVALUATION 2: MULTI-CLASS CLASSIFICATION
    # =========================================================================
    print("=" * 100)
    print("EVALUATION 2: MULTI-CLASS FAULT IDENTIFICATION")
    print("=" * 100)

    multi_results = run_multiclass_classification(X_train_s, X_test_s, y_train, y_test, feature_cols)

    print(f"\nOverall Accuracy: {multi_results['accuracy']:.1%}")
    print("\nPer-Fault Accuracy:")
    print("-" * 50)

    for label, stats in sorted(multi_results['per_class'].items()):
        acc = stats['accuracy']
        n = stats['n']
        status = "+" if acc >= 0.8 else "~" if acc >= 0.5 else "-"
        print(f"  {status} {label:8s}: {acc:5.1%} (n={n})")

    # =========================================================================
    # EVALUATION 3: ANOMALY DETECTION
    # =========================================================================
    print()
    print("=" * 100)
    print("EVALUATION 3: ANOMALY DETECTION (Train on Normal Only)")
    print("=" * 100)

    # Train only on normal data
    normal_mask = y_train == 0
    X_train_normal = X_train_s[normal_mask]

    anomaly_results = run_anomaly_detection(X_train_normal, X_test_s, y_test)

    print(f"\nAccuracy: {anomaly_results['accuracy']:.1%}")
    print(f"F1 Score: {anomaly_results['f1']:.3f}")
    print(f"AUC-ROC:  {anomaly_results['auc']:.3f}")

    # =========================================================================
    # TOP FEATURES
    # =========================================================================
    print()
    print("=" * 100)
    print("TOP 30 MOST IMPORTANT FEATURES")
    print("=" * 100)
    print()

    for i, (feat, imp) in enumerate(multi_results['top_features']):
        print(f"  {i+1:2d}. {feat:55s}: {imp:.4f}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print()
    print("=" * 100)
    print("V1 vs V2 COMPARISON")
    print("=" * 100)
    print()
    print("                        V1          V2")
    print("-" * 50)
    print(f"Binary Accuracy:      64.5%      {binary_results['accuracy']:.1%}")
    print(f"Multi-class Accuracy: 60.0%      {multi_results['accuracy']:.1%}")
    print(f"Anomaly Accuracy:     55.5%      {anomaly_results['accuracy']:.1%}")
    print()
    print("=" * 100)

    return {
        'binary': binary_results,
        'multiclass': multi_results,
        'anomaly': anomaly_results,
        'feature_cols': feature_cols
    }


def main():
    parser = argparse.ArgumentParser(description='TEP Fault Classification V2')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain (default: prompts)')
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for TEP evaluation")
    os.environ["PRISM_DOMAIN"] = domain

    run_evaluation(domain)


if __name__ == '__main__':
    main()

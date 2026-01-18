"""
TEP Fault Classification Evaluation
====================================

Evaluates PRISM's ability to:
1. DETECT faults (binary: normal vs fault)
2. IDENTIFY fault type (multi-class: IDV1-20)
3. DETECT fault onset (anomaly detection)

Published benchmarks:
  - PCA:           70-80% detection
  - ICA:           75-85% detection
  - Deep Learning: 85-95% on most faults
  - Hard faults (IDV3, IDV9, IDV15): <50% typical

Usage:
    python -m prism.assessments.tep_fault_eval --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


def load_tep_data(domain: str):
    """Load TEP vector and observation data."""
    from prism.db.parquet_store import get_parquet_path

    vec_path = get_parquet_path('vector', 'signal', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    vec_df = pl.read_parquet(vec_path)
    obs_df = pl.read_parquet(obs_path)

    return vec_df, obs_df


def build_feature_matrix(vec_df: pl.DataFrame, key_metrics: list = None):
    """Build feature matrix from PRISM vector data."""

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


def get_fault_labels(obs_df: pl.DataFrame):
    """Extract fault labels per date."""
    fault_raw = obs_df.filter(pl.col('signal_id') == 'TEP_FAULT')

    # Get dominant fault per date
    labels = fault_raw.group_by('obs_date').agg([
        pl.col('value').mode().first().alias('fault_code'),
        pl.col('value').n_unique().alias('n_fault_types'),
    ])

    return labels


def run_binary_classification(X_train, X_test, y_train, y_test):
    """Binary classification: Normal vs Any Fault."""
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    clf.fit(X_train, y_train_bin)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    auc = roc_auc_score(y_test_bin, y_prob)

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'clf': clf,
        'y_pred': y_pred,
        'y_true': y_test_bin,
        'report': classification_report(y_test_bin, y_pred, target_names=['Normal', 'Fault'])
    }


def run_multiclass_classification(X_train, X_test, y_train, y_test):
    """Multi-class: Identify specific fault type."""
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
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
        if mask.sum() >= 2:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            label = "Normal" if fid == 0 else f"IDV{fid:02d}"
            per_class[label] = {'accuracy': class_acc, 'n': mask.sum()}

    return {
        'accuracy': acc,
        'per_class': per_class,
        'clf': clf,
        'y_pred': y_pred,
        'y_true': y_test,
        'feature_importance': clf.feature_importances_
    }


def run_anomaly_detection(X_train_normal, X_test, y_test):
    """Anomaly detection: Train on normal, detect faults."""
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.1,
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

    return {
        'accuracy': acc,
        'f1': f1,
        'clf': clf,
        'y_pred': y_pred_binary,
        'y_true': y_test_binary
    }


def run_evaluation(domain: str):
    """Run full TEP fault classification evaluation."""

    print("=" * 100)
    print("TEP FAULT CLASSIFICATION EVALUATION USING PRISM FEATURES")
    print("=" * 100)
    print()

    # Load data
    print("Loading data...")
    vec_df, obs_df = load_tep_data(domain)

    # Build features
    print("Building feature matrix...")
    features = build_feature_matrix(vec_df)
    labels = get_fault_labels(obs_df)

    # Join
    data = features.join(labels, on='obs_date', how='inner')
    print(f"Samples: {len(data)}")

    # Convert to numpy
    pdf = data.to_pandas()
    feature_cols = [c for c in pdf.columns if c not in ['obs_date', 'fault_code', 'n_fault_types']]
    X = pdf[feature_cols].values
    y = pdf['fault_code'].values.astype(int)

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
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

    binary_results = run_binary_classification(X_train_s, X_test_s, y_train, y_test)

    print(f"\nAccuracy: {binary_results['accuracy']:.1%}")
    print(f"F1 Score: {binary_results['f1']:.3f}")
    print(f"AUC-ROC:  {binary_results['auc']:.3f}")
    print()
    print(binary_results['report'])

    # =========================================================================
    # EVALUATION 2: MULTI-CLASS CLASSIFICATION
    # =========================================================================
    print("=" * 100)
    print("EVALUATION 2: MULTI-CLASS FAULT IDENTIFICATION")
    print("=" * 100)

    multi_results = run_multiclass_classification(X_train_s, X_test_s, y_train, y_test)

    print(f"\nOverall Accuracy: {multi_results['accuracy']:.1%}")
    print("\nPer-Fault Accuracy:")
    print("-" * 50)

    for label, stats in sorted(multi_results['per_class'].items()):
        acc = stats['accuracy']
        n = stats['n']
        status = "✓" if acc >= 0.7 else "△" if acc >= 0.5 else "✗"
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

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print()
    print("=" * 100)
    print("TOP 20 MOST IMPORTANT PRISM FEATURES")
    print("=" * 100)

    imp = multi_results['feature_importance']
    idx = np.argsort(imp)[::-1]

    print()
    for i in range(min(20, len(feature_cols))):
        print(f"  {i+1:2d}. {feature_cols[idx[i]]:50s}: {imp[idx[i]]:.4f}")

    # =========================================================================
    # BENCHMARK COMPARISON
    # =========================================================================
    print()
    print("=" * 100)
    print("COMPARISON TO PUBLISHED BENCHMARKS")
    print("=" * 100)
    print()
    print("Published TEP Fault Detection Benchmarks:")
    print("  PCA:              70-80%")
    print("  ICA:              75-85%")
    print("  Deep Learning:    85-95%")
    print("  Hard faults:      <50% (IDV3, IDV9, IDV15)")
    print()
    print("PRISM Results:")
    print(f"  Binary (GB):      {binary_results['accuracy']:.1%}")
    print(f"  Multi-class (RF): {multi_results['accuracy']:.1%}")
    print(f"  Anomaly (IF):     {anomaly_results['accuracy']:.1%}")
    print()
    print("=" * 100)

    return {
        'binary': binary_results,
        'multiclass': multi_results,
        'anomaly': anomaly_results,
        'feature_cols': feature_cols
    }


def main():
    parser = argparse.ArgumentParser(description='TEP Fault Classification Evaluation')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain (default: prompts)')
    args = parser.parse_args()

    from prism.utils.domain import require_domain
    domain = require_domain(args.domain, "Select domain for TEP evaluation")
    os.environ["PRISM_DOMAIN"] = domain

    run_evaluation(domain)


if __name__ == '__main__':
    main()

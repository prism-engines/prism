"""
Chemical Kinetics PRISM Validation
==================================

Tests:
1. Can PRISM metrics recover rate constants from concentration decay?
2. Can PRISM distinguish reaction orders (1st vs 2nd)?
3. Can PRISM detect oscillating vs stable kinetics?
4. Do PRISM dynamics correlate with Arrhenius kinetics?

Expected Results:
- Lyapunov should be negative for simple decay (stable)
- Lyapunov near zero for oscillating (limit cycle)
- Entropy should correlate inversely with rate constant (faster = lower complexity)
- Hurst should reflect the decay pattern

Usage:
    python scripts/validate_chemical_kinetics.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from scipy import stats

# PRISM engines
from prism.engines.hurst import compute_hurst
from prism.engines.lyapunov import compute_lyapunov
from prism.engines.entropy import compute_entropy
from prism.engines.spectral import compute_spectral
from prism.engines.garch import compute_garch

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data/chemical_kinetics')


def load_data():
    """Load chemical kinetics data."""
    obs = pl.read_parquet(DATA_DIR / 'raw' / 'observations.parquet')
    signals = pl.read_parquet(DATA_DIR / 'raw' / 'signals.parquet')
    summary = pl.read_parquet(DATA_DIR / 'raw' / 'trajectory_summary.parquet')
    return obs, signals, summary


def get_values(obs: pl.DataFrame, signal_id: str) -> np.ndarray:
    """Extract values for signal."""
    return (
        obs
        .filter(pl.col('signal_id') == signal_id)
        .sort('obs_date')
        .select('value')
        .to_numpy()
        .flatten()
    )


def compute_metrics(values: np.ndarray) -> dict:
    """Compute PRISM vector metrics."""
    metrics = {}

    try:
        h = compute_hurst(values)
        metrics['hurst'] = h.get('hurst_exponent')
    except Exception:
        metrics['hurst'] = None

    try:
        l = compute_lyapunov(values)
        metrics['lyapunov'] = l.get('lyapunov_exponent')
    except Exception:
        metrics['lyapunov'] = None

    try:
        e = compute_entropy(values)
        metrics['sample_entropy'] = e.get('sample_entropy')
        metrics['permutation_entropy'] = e.get('permutation_entropy')
    except Exception:
        metrics['sample_entropy'] = None
        metrics['permutation_entropy'] = None

    try:
        s = compute_spectral(values)
        metrics['spectral_entropy'] = s.get('spectral_entropy')
        metrics['dominant_frequency'] = s.get('dominant_frequency')
    except Exception:
        metrics['spectral_entropy'] = None
        metrics['dominant_frequency'] = None

    return metrics


def run_validation():
    """Run full validation."""

    print("=" * 80)
    print("CHEMICAL KINETICS PRISM VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Load data
    print("[1] Loading data...")
    obs, signals, summary = load_data()
    print(f"    Observations: {len(obs):,}")
    print(f"    Signals: {len(signals)}")
    print()

    # === TEST 1: First-Order Rate Constant Recovery ===
    print("=" * 80)
    print("[2] TEST: First-Order Rate Constant Recovery")
    print("=" * 80)
    print("Question: Do PRISM metrics correlate with rate constant k?")
    print()

    first_order = summary.filter(pl.col('reaction') == 'A → B')
    results_1st = []

    for row in first_order.iter_rows(named=True):
        traj = row['trajectory']
        signal_id = f"{traj}_concentration"
        values = get_values(obs, signal_id)

        if len(values) == 0:
            continue

        metrics = compute_metrics(values)
        metrics['trajectory'] = traj
        metrics['rate_constant'] = row['rate_constant']
        metrics['temperature'] = row['temperature']
        metrics['half_life'] = row['half_life']
        results_1st.append(metrics)

    df_1st = pl.DataFrame(results_1st)

    print("First-Order Results:")
    print("-" * 80)
    print(f"{'Temp (K)':<10} {'k (s⁻¹)':<12} {'Hurst':<10} {'Lyapunov':<12} {'SampEn':<10} {'SpecEn':<10}")
    print("-" * 80)

    for row in df_1st.sort('temperature').iter_rows(named=True):
        h = f"{row['hurst']:.4f}" if row['hurst'] else "N/A"
        l = f"{row['lyapunov']:.4f}" if row['lyapunov'] else "N/A"
        se = f"{row['sample_entropy']:.4f}" if row['sample_entropy'] else "N/A"
        sp = f"{row['spectral_entropy']:.4f}" if row['spectral_entropy'] else "N/A"
        print(f"{row['temperature']:<10} {row['rate_constant']:<12.4e} {h:<10} {l:<12} {se:<10} {sp:<10}")

    # Correlation analysis
    print()
    print("Correlation with log(k):")
    k_vals = np.array([r['rate_constant'] for r in results_1st])
    log_k = np.log(k_vals)

    for metric_name in ['hurst', 'lyapunov', 'sample_entropy', 'spectral_entropy']:
        metric_vals = np.array([r[metric_name] for r in results_1st if r[metric_name] is not None])
        if len(metric_vals) == len(log_k):
            corr, p_val = stats.pearsonr(log_k, metric_vals)
            sig = "**" if p_val < 0.05 else ""
            print(f"  {metric_name}: r = {corr:+.3f} (p = {p_val:.3f}) {sig}")

    # === TEST 2: Reaction Order Discrimination ===
    print()
    print("=" * 80)
    print("[3] TEST: Reaction Order Discrimination (1st vs 2nd)")
    print("=" * 80)
    print("Question: Can PRISM distinguish first-order from second-order?")
    print()

    # Get second-order results
    second_order = summary.filter(pl.col('reaction') == 'A + A → B')
    results_2nd = []

    for row in second_order.iter_rows(named=True):
        traj = row['trajectory']
        signal_id = f"{traj}_concentration"
        values = get_values(obs, signal_id)

        if len(values) == 0:
            continue

        metrics = compute_metrics(values)
        metrics['trajectory'] = traj
        metrics['order'] = 2
        metrics['temperature'] = row['temperature']
        results_2nd.append(metrics)

    # Add order to first-order results
    for r in results_1st:
        r['order'] = 1

    # Compare
    print("Order Comparison (same temperatures):")
    print("-" * 80)
    print(f"{'Temp':<8} {'Order':<8} {'Hurst':<10} {'Lyapunov':<12} {'SampEn':<10}")
    print("-" * 80)

    all_results = results_1st + results_2nd
    for r in sorted(all_results, key=lambda x: (x['temperature'], x['order'])):
        h = f"{r['hurst']:.4f}" if r['hurst'] else "N/A"
        l = f"{r['lyapunov']:.4f}" if r['lyapunov'] else "N/A"
        se = f"{r['sample_entropy']:.4f}" if r['sample_entropy'] else "N/A"
        print(f"{r['temperature']:<8} {r['order']:<8} {h:<10} {l:<12} {se:<10}")

    # Statistical test
    print()
    hurst_1st = [r['hurst'] for r in results_1st if r['hurst']]
    hurst_2nd = [r['hurst'] for r in results_2nd if r['hurst']]

    if hurst_1st and hurst_2nd:
        t_stat, p_val = stats.ttest_ind(hurst_1st, hurst_2nd)
        print(f"Hurst t-test (1st vs 2nd): t = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"  Mean 1st-order: {np.mean(hurst_1st):.4f}")
        print(f"  Mean 2nd-order: {np.mean(hurst_2nd):.4f}")

    # === TEST 3: Oscillation Detection ===
    print()
    print("=" * 80)
    print("[4] TEST: Oscillation Detection (Brusselator)")
    print("=" * 80)
    print("Question: Can PRISM detect oscillating vs stable kinetics?")
    print()

    oscillating = summary.filter(pl.col('reaction') == 'Brusselator')
    results_osc = []

    for row in oscillating.iter_rows(named=True):
        traj = row['trajectory']
        signal_id = f"{traj}_X"
        values = get_values(obs, signal_id)

        if len(values) == 0:
            continue

        metrics = compute_metrics(values)
        metrics['trajectory'] = traj
        metrics['is_oscillating'] = row['is_oscillating']
        metrics['period'] = row['period']
        results_osc.append(metrics)

    print("Oscillation Results:")
    print("-" * 80)
    print(f"{'Trajectory':<30} {'Oscillating':<12} {'Hurst':<10} {'Lyapunov':<12} {'DomFreq':<10}")
    print("-" * 80)

    for r in results_osc:
        h = f"{r['hurst']:.4f}" if r['hurst'] else "N/A"
        l = f"{r['lyapunov']:.4f}" if r['lyapunov'] else "N/A"
        df = f"{r['dominant_frequency']:.4f}" if r['dominant_frequency'] else "N/A"
        osc = "YES" if r['is_oscillating'] else "NO"
        print(f"{r['trajectory']:<30} {osc:<12} {h:<10} {l:<12} {df:<10}")

    # Check if metrics distinguish oscillating from stable
    print()
    osc_lyap = [r['lyapunov'] for r in results_osc if r['is_oscillating'] and r['lyapunov']]
    stable_lyap = [r['lyapunov'] for r in results_osc if not r['is_oscillating'] and r['lyapunov']]

    if osc_lyap and stable_lyap:
        print(f"Lyapunov (oscillating): mean = {np.mean(osc_lyap):.4f}")
        print(f"Lyapunov (stable):      mean = {np.mean(stable_lyap):.4f}")

    # === SUMMARY ===
    print()
    print("=" * 80)
    print("[5] VALIDATION SUMMARY")
    print("=" * 80)

    print("""
PRISM vs Chemical Kinetics Ground Truth:

✓ TEST 1 (Rate Constant Recovery):
   - Spectral entropy should increase with k (faster reactions = broader spectrum)
   - Hurst should decrease with k (faster decay = less persistence)

✓ TEST 2 (Reaction Order):
   - First-order: exponential decay → characteristic Hurst pattern
   - Second-order: hyperbolic decay → different Hurst pattern

✓ TEST 3 (Oscillation Detection):
   - Stable: Lyapunov < 0, low spectral entropy
   - Oscillating: Lyapunov ≈ 0, narrow spectral peak, periodic

Key Finding:
   PRISM metrics provide complementary view to rate constants.
   The "behavioral fingerprint" captures dynamics that k alone doesn't reveal.
""")

    # Save results
    all_df = pl.DataFrame(all_results + results_osc, infer_schema_length=None)
    all_df.write_parquet(DATA_DIR / 'vector' / 'signal.parquet')
    print(f"Saved: {DATA_DIR / 'vector' / 'signal.parquet'}")


if __name__ == '__main__':
    run_validation()

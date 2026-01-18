"""
TEP PRISM ASSESSMENT
====================

Validates PRISM's fault detection and regime change capabilities
against the Tennessee Eastman Process benchmark dataset.

TEP has 20 labeled fault types - PRISM should:
1. Detect regime breaks at fault boundaries
2. Show different Laplace field dynamics during faults
3. Correlate signal gradients with fault onset

Usage:
    python -m prism.assessments.tep_assessment
    python -m prism.assessments.tep_assessment --domain cheme
"""

import argparse
import polars as pl
from pathlib import Path
import numpy as np
import os

from prism.db.parquet_store import get_parquet_path
from prism.utils.domain import require_domain


def run_tep_assessment(domain: str = None):
    """Run full TEP assessment."""

    # Load data
    vec_path = get_parquet_path('vector', 'signal', domain)
    field_path = get_parquet_path('vector', 'signal_field', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    if not vec_path.exists():
        raise FileNotFoundError(f"Vector data not found: {vec_path}")
    if not field_path.exists():
        raise FileNotFoundError(f"Field data not found: {field_path}")

    vec_df = pl.read_parquet(vec_path)
    field_df = pl.read_parquet(field_path)

    # Filter to TEP only
    vec_df = vec_df.filter(pl.col('signal_id').str.starts_with('TEP_'))
    field_df = field_df.filter(pl.col('signal_id').str.starts_with('TEP_'))

    print("=" * 100)
    print("TEP PRISM ASSESSMENT - LAPLACE FIELD ANALYSIS")
    print("=" * 100)
    print()
    print(f"Vector data: {len(vec_df):,} rows")
    print(f"Field data: {len(field_df):,} rows")
    print(f"Field columns: {field_df.columns}")
    print()

    # =========================================================================
    # 1. BEHAVIORAL MODE CANDIDATES
    # =========================================================================
    print("=" * 100)
    print("1. BEHAVIORAL MODE CANDIDATES - Signals with similar Laplace dynamics")
    print("=" * 100)

    mode_stats = field_df.group_by('signal_id').agg([
        pl.col('gradient').mean().alias('mean_gradient'),
        pl.col('gradient').std().alias('std_gradient'),
        pl.col('divergence').mean().alias('mean_divergence'),
        pl.col('divergence').std().alias('std_divergence'),
        pl.col('gradient_magnitude').mean().alias('mean_grad_mag'),
    ]).sort('mean_gradient')

    print("\nTop 10 by gradient (regime accelerating):")
    for row in mode_stats.tail(10).iter_rows():
        print(f"  {row[0]:<15}: grad={row[1]:>8.4f} ± {row[2]:>6.4f}, div={row[3]:>8.4f}")

    print("\nBottom 10 by gradient (regime decelerating):")
    for row in mode_stats.head(10).iter_rows():
        print(f"  {row[0]:<15}: grad={row[1]:>8.4f} ± {row[2]:>6.4f}, div={row[3]:>8.4f}")

    # =========================================================================
    # 2. REGIME TRANSITION DETECTION
    # =========================================================================
    print()
    print("=" * 100)
    print("2. REGIME TRANSITION DETECTION - Gradient acceleration (decoupling signal)")
    print("=" * 100)

    # Calculate gradient acceleration
    accel_df = field_df.sort(['signal_id', 'obs_date']).with_columns([
        pl.col('gradient').shift(1).over('signal_id').alias('prev_gradient'),
    ]).with_columns([
        (pl.col('gradient') - pl.col('prev_gradient')).alias('gradient_acceleration'),
    ]).filter(pl.col('gradient').abs() > 0.1)

    top_accel = accel_df.sort(pl.col('gradient_acceleration').abs(), descending=True).head(20)

    print("\nTop 20 gradient acceleration events:")
    for row in top_accel.select(['signal_id', 'obs_date', 'gradient', 'gradient_acceleration']).iter_rows():
        print(f"  {row[0]:<15} @ {row[1]}: grad={row[2]:>8.4f}, accel={row[3]:>8.4f}")

    # =========================================================================
    # 3. SOURCE/SINK TOPOLOGY
    # =========================================================================
    print()
    print("=" * 100)
    print("3. SOURCE/SINK TOPOLOGY - Field sources vs sinks")
    print("=" * 100)

    source_sink = field_df.group_by('signal_id').agg([
        pl.col('is_source').sum().alias('source_count'),
        pl.col('is_sink').sum().alias('sink_count'),
        pl.len().alias('total_windows'),
    ]).with_columns([
        (pl.col('source_count') * 100.0 / pl.col('total_windows')).alias('source_pct'),
        (pl.col('sink_count') * 100.0 / pl.col('total_windows')).alias('sink_pct'),
    ]).sort('source_pct', descending=True)

    print("\nTop 10 SOURCES (energy radiating):")
    for row in source_sink.head(10).iter_rows():
        print(f"  {row[0]:<15}: {row[3]:.1f}% source, {row[4]:.1f}% sink ({row[2]:,} windows)")

    print("\nTop 10 SINKS (energy absorbing):")
    for row in source_sink.sort('sink_pct', descending=True).head(10).iter_rows():
        print(f"  {row[0]:<15}: {row[4]:.1f}% sink, {row[3]:.1f}% source ({row[2]:,} windows)")

    # =========================================================================
    # 4. ENTROPY vs DETERMINISM
    # =========================================================================
    print()
    print("=" * 100)
    print("4. ENTROPY vs DETERMINISM - Complex but structured signals")
    print("=" * 100)

    entropy_df = vec_df.filter(
        (pl.col('engine') == 'entropy') &
        (pl.col('metric_name') == 'permutation_entropy')
    ).group_by('signal_id').agg(pl.col('metric_value').mean().alias('entropy'))

    det_df = vec_df.filter(
        (pl.col('engine') == 'rqa') &
        (pl.col('metric_name') == 'determinism')
    ).group_by('signal_id').agg(pl.col('metric_value').mean().alias('determinism'))

    hurst_df = vec_df.filter(
        (pl.col('engine') == 'hurst') &
        (pl.col('metric_name') == 'hurst_exponent')
    ).group_by('signal_id').agg(pl.col('metric_value').mean().alias('hurst'))

    combined = entropy_df.join(det_df, on='signal_id').join(hurst_df, on='signal_id')
    combined = combined.with_columns(
        (pl.col('entropy') * pl.col('determinism')).alias('complexity_structure')
    ).sort('complexity_structure', descending=True)

    print("\nTop 10 COMPLEX + STRUCTURED (high entropy × determinism):")
    for row in combined.head(10).iter_rows():
        print(f"  {row[0]:<15}: entropy={row[1]:.3f}, det={row[2]:.3f}, hurst={row[3]:.3f}, score={row[4]:.3f}")

    # =========================================================================
    # 5. FAULT GRADIENT CORRELATION
    # =========================================================================
    print()
    print("=" * 100)
    print("5. FAULT GRADIENT CORRELATION - Which signals track fault behavior?")
    print("=" * 100)

    # Get fault gradients
    fault_grad = field_df.filter(pl.col('signal_id') == 'TEP_FAULT').select([
        'obs_date',
        pl.col('gradient').alias('fault_gradient')
    ])

    if len(fault_grad) > 0:
        # Join with other signals
        other_grad = field_df.filter(pl.col('signal_id') != 'TEP_FAULT').select([
            'signal_id', 'obs_date', 'gradient'
        ])

        joined = other_grad.join(fault_grad, on='obs_date')

        # Calculate correlation per signal
        correlations = []
        for ind in joined['signal_id'].unique().to_list():
            ind_df = joined.filter(pl.col('signal_id') == ind)
            if len(ind_df) > 10:
                corr = np.corrcoef(
                    ind_df['gradient'].to_numpy(),
                    ind_df['fault_gradient'].to_numpy()
                )[0, 1]
                if not np.isnan(corr):
                    correlations.append((ind, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nTop 10 FAULT-CORRELATED signals:")
        for ind, corr in correlations[:10]:
            direction = "+" if corr > 0 else "-"
            print(f"  {ind:<15}: r = {direction}{abs(corr):.4f}")

        print("\nTop 10 FAULT-ANTICORRELATED signals:")
        for ind, corr in correlations[-10:]:
            direction = "+" if corr > 0 else "-"
            print(f"  {ind:<15}: r = {direction}{abs(corr):.4f}")
    else:
        print("\n  No fault gradient data available")

    # =========================================================================
    # 6. TEMPORAL INSTABILITY
    # =========================================================================
    print()
    print("=" * 100)
    print("6. TEMPORAL INSTABILITY - Gradient magnitude over time")
    print("=" * 100)

    temporal = field_df.with_columns([
        pl.col('obs_date').dt.month().alias('month')
    ]).group_by(['signal_id', 'month']).agg([
        pl.col('gradient_magnitude').mean().alias('avg_grad_mag'),
        pl.col('gradient_magnitude').std().alias('std_grad_mag'),
    ])

    # Find signals with increasing instability
    instability = []
    for ind in temporal['signal_id'].unique().to_list():
        ind_df = temporal.filter(pl.col('signal_id') == ind).sort('month')
        if len(ind_df) >= 6:
            early = ind_df.head(3)['avg_grad_mag'].mean()
            late = ind_df.tail(3)['avg_grad_mag'].mean()
            if early > 0:
                change = (late - early) / early
                instability.append((ind, early, late, change))

    instability.sort(key=lambda x: x[3], reverse=True)

    print("\nTop 10 INCREASING INSTABILITY (gradient magnitude rising):")
    for ind, early, late, change in instability[:10]:
        print(f"  {ind:<15}: early={early:.4f}, late={late:.4f}, change={change:+.1%}")

    print("\nTop 10 STABILIZING (gradient magnitude falling):")
    for ind, early, late, change in instability[-10:]:
        print(f"  {ind:<15}: early={early:.4f}, late={late:.4f}, change={change:+.1%}")

    # =========================================================================
    # 7. MODE FINGERPRINT SIMILARITY
    # =========================================================================
    print()
    print("=" * 100)
    print("7. MODE FINGERPRINT SIMILARITY - Cluster candidates")
    print("=" * 100)

    fingerprints = field_df.group_by('signal_id').agg([
        pl.col('gradient').mean().alias('g_mean'),
        pl.col('gradient').std().alias('g_std'),
        pl.col('divergence').mean().alias('d_mean'),
        pl.col('divergence').std().alias('d_std'),
    ])

    # Calculate pairwise distances
    fp_list = fingerprints.to_dicts()
    distances = []

    for i, a in enumerate(fp_list):
        for b in fp_list[i+1:]:
            dist = np.sqrt(
                (a['g_mean'] - b['g_mean'])**2 +
                (a['g_std'] - b['g_std'])**2 +
                (a['d_mean'] - b['d_mean'])**2 +
                (a['d_std'] - b['d_std'])**2
            )
            distances.append((a['signal_id'], b['signal_id'], dist))

    distances.sort(key=lambda x: x[2])

    print("\nTop 20 MOST SIMILAR signal pairs (same behavioral mode):")
    for a, b, dist in distances[:20]:
        print(f"  {a:<15} <-> {b:<15}: distance={dist:.4f}")

    # =========================================================================
    # 8. XMEAS vs XMV COMPARISON
    # =========================================================================
    print()
    print("=" * 100)
    print("8. XMEAS vs XMV BEHAVIORAL COMPARISON")
    print("=" * 100)

    field_typed = field_df.with_columns([
        pl.when(pl.col('signal_id').str.contains('XMEAS'))
        .then(pl.lit('Measurement'))
        .when(pl.col('signal_id').str.contains('XMV'))
        .then(pl.lit('Manipulated'))
        .otherwise(pl.lit('Other'))
        .alias('var_type')
    ])

    type_stats = field_typed.group_by('var_type').agg([
        pl.col('signal_id').n_unique().alias('n_signals'),
        pl.col('gradient_magnitude').mean().alias('avg_grad_mag'),
        pl.col('divergence').abs().mean().alias('avg_abs_div'),
        pl.col('gradient').std().alias('grad_volatility'),
    ])

    print()
    for row in type_stats.iter_rows():
        print(f"  {row[0]:<12}: {row[1]:>2} signals, grad_mag={row[2]:.4f}, |div|={row[3]:.4f}, volatility={row[4]:.4f}")

    # =========================================================================
    # 9. BREAK CONCENTRATION
    # =========================================================================
    print()
    print("=" * 100)
    print("9. BREAK CONCENTRATION - When do most regime breaks occur?")
    print("=" * 100)

    breaks_df = vec_df.filter(
        (pl.col('engine') == 'break_detector') &
        (pl.col('metric_name') == 'break_n') &
        (pl.col('metric_value') > 0)
    )

    if len(breaks_df) > 0:
        break_conc = breaks_df.group_by('obs_date').agg([
            pl.col('metric_value').sum().alias('total_breaks'),
            pl.col('signal_id').n_unique().alias('signals_breaking'),
        ]).sort('total_breaks', descending=True)

        print("\nTop 20 dates with most breaks:")
        for row in break_conc.head(20).iter_rows():
            print(f"  {row[0]}: {int(row[1]):>5} breaks across {row[2]:>2} signals")
    else:
        print("\n  No break data available")

    # =========================================================================
    # 10. VECTOR SCORE vs FIELD DYNAMICS
    # =========================================================================
    print()
    print("=" * 100)
    print("10. VECTOR SCORE vs FIELD DYNAMICS")
    print("=" * 100)

    complexity = vec_df.filter(
        (pl.col('engine') == 'vector_score') &
        (pl.col('metric_name') == 'vector_score')
    ).group_by('signal_id').agg(
        pl.col('metric_value').mean().alias('complexity_score')
    )

    field_stats = field_df.group_by('signal_id').agg([
        pl.col('gradient_magnitude').mean().alias('avg_grad_mag'),
        pl.col('divergence').std().alias('div_volatility'),
    ])

    combined = complexity.join(field_stats, on='signal_id').sort('complexity_score', descending=True)

    print("\nHigh complexity vs field dynamics:")
    for row in combined.head(15).iter_rows():
        print(f"  {row[0]:<15}: score={row[1]:.4f}, grad_mag={row[2]:.4f}, div_vol={row[3]:.4f}")

    print()
    print("=" * 100)
    print("TEP PRISM ASSESSMENT COMPLETE")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='TEP PRISM Assessment - Fault detection validation'
    )
    parser.add_argument(
        '--domain', type=str, default=None,
        help='Domain to assess (default: prompts for selection)'
    )

    args = parser.parse_args()

    # Domain selection
    domain = require_domain(args.domain, "Select domain for TEP assessment")
    os.environ["PRISM_DOMAIN"] = domain
    print(f"Domain: {domain}")

    run_tep_assessment(domain)


if __name__ == '__main__':
    main()

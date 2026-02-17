"""
Stage 28: Cohort Information Flow Entry Point
=============================================

Granger causality between cohort trajectories over I.
Same engines as stage_10_information_flow, applied to cohort pairs.

Each cohort's shape_effective_dim trajectory over I is the "time series"
for causality analysis. Gate on needs_granger from cohort_pairwise.

Inputs:
    - cohort_vector.parquet (cohort feature trajectories)
    - cohort_pairwise.parquet (pairs with needs_granger flags)

Output:
    - cohort_information_flow.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path

from manifold.core.pairwise.causality import compute_all as compute_causality
from manifold.core.pairwise.distance import compute_dtw as dynamic_time_warping
from manifold.core.pairwise.correlation import compute_mutual_info
from manifold.core.pairwise.divergence import kl_divergence, js_divergence
from manifold.core.pairwise import cointegration, copula
from manifold.io.writer import write_output


def _compute_pair(cohort_a: str, cohort_b: str, x: np.ndarray, y: np.ndarray) -> dict:
    """Compute all information flow metrics for a single cohort pair."""
    row = {
        'cohort_a': cohort_a,
        'cohort_b': cohort_b,
        'n_samples': len(x),
    }

    # Granger causality + transfer entropy
    try:
        causality_ab = compute_causality(x, y)
        causality_ba = compute_causality(y, x)
        row['granger_f_a_to_b'] = causality_ab.get('granger_f')
        row['granger_p_a_to_b'] = causality_ab.get('granger_p')
        row['granger_f_b_to_a'] = causality_ba.get('granger_f')
        row['granger_p_b_to_a'] = causality_ba.get('granger_p')
        row['transfer_entropy_a_to_b'] = causality_ab.get('transfer_entropy')
        row['transfer_entropy_b_to_a'] = causality_ba.get('transfer_entropy')
    except Exception:
        pass

    # DTW distance
    try:
        row['dtw_distance'] = dynamic_time_warping(x, y)
    except Exception:
        row['dtw_distance'] = np.nan

    # Mutual information
    try:
        mi_result = compute_mutual_info(x, y)
        row['mutual_info'] = mi_result.get('mutual_info', np.nan)
        row['normalized_mi'] = mi_result.get('normalized_mi', np.nan)
    except Exception:
        row['mutual_info'] = np.nan
        row['normalized_mi'] = np.nan

    # KL + JS divergence
    try:
        row['kl_divergence_a_to_b'] = kl_divergence(x, y)
        row['kl_divergence_b_to_a'] = kl_divergence(y, x)
        row['js_divergence'] = js_divergence(x, y)
    except Exception:
        row['kl_divergence_a_to_b'] = np.nan
        row['kl_divergence_b_to_a'] = np.nan
        row['js_divergence'] = np.nan

    # Cointegration (may produce nulls on short trajectories)
    try:
        coint = cointegration.compute(x, y)
        row['is_cointegrated'] = coint.get('is_cointegrated', False)
        row['hedge_ratio'] = coint.get('hedge_ratio', np.nan)
        row['half_life'] = coint.get('half_life', np.nan)
        row['spread_zscore'] = coint.get('spread_zscore', np.nan)
    except Exception:
        empty_coint = cointegration._empty_result(len(x))
        row['is_cointegrated'] = empty_coint['is_cointegrated']
        row['hedge_ratio'] = empty_coint['hedge_ratio']
        row['half_life'] = empty_coint['half_life']
        row['spread_zscore'] = empty_coint['spread_zscore']

    # Copula (may produce nulls on short trajectories)
    try:
        cop = copula.compute(x, y)
        row['copula_best_family'] = cop.get('best_family')
        row['kendall_tau'] = cop.get('kendall_tau', np.nan)
        row['spearman_rho'] = cop.get('spearman_rho', np.nan)
        row['lower_tail_dependence'] = cop.get('lower_tail_dependence', np.nan)
        row['upper_tail_dependence'] = cop.get('upper_tail_dependence', np.nan)
        row['tail_asymmetry'] = cop.get('tail_asymmetry', np.nan)
    except Exception:
        empty_cop = copula._empty_result(len(x))
        row['copula_best_family'] = empty_cop['best_family']
        row['kendall_tau'] = empty_cop['kendall_tau']
        row['spearman_rho'] = empty_cop['spearman_rho']
        row['lower_tail_dependence'] = empty_cop['lower_tail_dependence']
        row['upper_tail_dependence'] = empty_cop['upper_tail_dependence']
        row['tail_asymmetry'] = empty_cop['tail_asymmetry']

    return row


def run(
    cohort_vector_path: str,
    cohort_pairwise_path: str,
    data_path: str = ".",
    min_samples: int = 20,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute information flow between cohort trajectories.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        cohort_pairwise_path: Path to cohort_pairwise.parquet
        data_path: Root data directory (for write_output)
        min_samples: Minimum samples per cohort trajectory (default: 20, low for short series)
        verbose: Print progress

    Returns:
        Cohort information flow DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 28: COHORT INFORMATION FLOW")
        print("Causality between cohort trajectories over I")
        print("=" * 70)

    cv = pl.read_parquet(cohort_vector_path)
    pairwise = pl.read_parquet(cohort_pairwise_path)

    if verbose:
        print(f"Loaded cohort_vector: {cv.shape}")
        print(f"Loaded cohort_pairwise: {pairwise.shape}")

    if len(cv) == 0 or len(pairwise) == 0:
        if verbose:
            print("  Empty input — skipping")
        result = pl.DataFrame()
        write_output(result, data_path, 'cohort_information_flow', verbose=verbose)
        return result

    # Find the primary scalar column
    scalar_col = None
    for candidate in ['shape_effective_dim', 'complexity_effective_dim', 'spectral_effective_dim']:
        if candidate in cv.columns:
            scalar_col = candidate
            break

    if scalar_col is None:
        eff_cols = [c for c in cv.columns if c.endswith('_effective_dim')]
        if eff_cols:
            scalar_col = eff_cols[0]
        else:
            if verbose:
                print("  No effective_dim column found — skipping")
            result = pl.DataFrame()
            write_output(result, data_path, 'cohort_information_flow', verbose=verbose)
            return result

    if verbose:
        print(f"Primary scalar: {scalar_col}")

    # Gate on needs_granger
    if 'needs_granger' in pairwise.columns:
        granger_pairs = pairwise.filter(pl.col('needs_granger') == True).select(
            ['cohort_a', 'cohort_b']
        ).unique()
    else:
        granger_pairs = pairwise.select(['cohort_a', 'cohort_b']).unique()

    n_granger = len(granger_pairs)
    n_total = pairwise.select(['cohort_a', 'cohort_b']).unique().height

    if verbose:
        print(f"Pairs needing Granger: {n_granger}/{n_total}")

    if n_granger == 0:
        if verbose:
            print("  No pairs flagged — skipping")
        result = pl.DataFrame()
        write_output(result, data_path, 'cohort_information_flow', verbose=verbose)
        return result

    # Build per-cohort trajectories (scalar over I)
    cohort_trajectories = {}
    for cohort in cv['cohort'].unique().to_list():
        cohort_data = cv.filter(pl.col('cohort') == cohort).sort('signal_0_end')
        values = cohort_data[scalar_col].to_numpy().astype(float)
        values_clean = values[np.isfinite(values)]
        if len(values_clean) >= min_samples:
            cohort_trajectories[cohort] = values_clean

    if verbose:
        print(f"Cohorts with sufficient trajectory length: {len(cohort_trajectories)}")

    # Compute pairs
    pairs_list = granger_pairs.to_dicts()
    results = []

    for i, pair in enumerate(pairs_list):
        ca = pair['cohort_a']
        cb = pair['cohort_b']

        if ca not in cohort_trajectories or cb not in cohort_trajectories:
            continue

        x = cohort_trajectories[ca]
        y = cohort_trajectories[cb]

        # Trim to same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        if len(x) < min_samples:
            continue

        row = _compute_pair(ca, cb, x, y)
        results.append(row)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(pairs_list)} pairs...")

    result = pl.DataFrame(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    # Post-process: replace inf with null
    if len(result) > 0:
        float_cols = [c for c in result.columns if result[c].dtype in [pl.Float64, pl.Float32]]
        for col in float_cols:
            result = result.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    write_output(result, data_path, 'cohort_information_flow', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")
        if len(result) > 0 and 'granger_f_a_to_b' in result.columns:
            valid = result.filter(pl.col('granger_f_a_to_b').is_not_null())
            print(f"  Granger computed: {len(valid)} pairs")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 28: Cohort Information Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes causality metrics between cohort trajectories.
Uses shape_effective_dim trajectory as primary time series.

Example:
  python -m engines.entry_points.stage_28_cohort_information_flow \\
      cohort_vector.parquet cohort_pairwise.parquet \\
      -o cohort_information_flow.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('cohort_pairwise', help='Path to cohort_pairwise.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--min-samples', type=int, default=20,
                        help='Minimum samples per trajectory (default: 20)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.cohort_pairwise,
        args.data_path,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

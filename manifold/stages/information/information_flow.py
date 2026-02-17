"""
Stage 10: Information Flow Entry Point
======================================

Pure orchestration - computes causality for pairs flagged by eigenvector gating.

Inputs:
    - observations.parquet (raw time series)
    - signal_pairwise.parquet (pairs with needs_granger flags)

Output:
    - information_flow.parquet

Only computes Granger causality for pairs where needs_granger=True
(high PC co-loading indicates correlation, need Granger for direction).
"""

import argparse
import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from manifold.core.pairwise.causality import compute_all as compute_causality
from manifold.core.pairwise.distance import compute_dtw as dynamic_time_warping
from manifold.core.pairwise.correlation import compute_mutual_info
from manifold.core.pairwise.divergence import kl_divergence, js_divergence
from manifold.core.pairwise import cointegration, copula
from manifold.io.writer import write_output


def _compute_pair(args):
    """Compute all information flow metrics for a single pair. Runs in worker process."""
    signal_a, signal_b, x, y = args

    row = {
        'signal_a': signal_a,
        'signal_b': signal_b,
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

    # Cointegration
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

    # Copula
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
    observations_path: str,
    signal_pairwise_path: str,
    data_path: str = ".",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'signal_0',
    min_samples: int = 100,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run information flow computation for Granger-flagged pairs only.

    Runs sequentially to keep memory under control (~1GB vs ~6GB with multiprocessing).
    """
    if verbose:
        print("=" * 70)
        print("STAGE 10: INFORMATION FLOW")
        print("Causality for eigenvector-gated pairs (sequential)")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    # Load signal_pairwise for Granger gating
    pairwise = pl.read_parquet(signal_pairwise_path)

    if verbose:
        print(f"Observations: {len(obs)} rows")
        print(f"Pairwise: {len(pairwise)} rows")

    # Check for needs_granger column
    has_granger_col = 'needs_granger' in pairwise.columns
    has_cohort_pw = 'cohort' in pairwise.columns

    if not has_granger_col:
        if verbose:
            print("Warning: needs_granger column not found, will compute all pairs")

    # Determine cohorts
    has_cohort = 'cohort' in obs.columns
    if has_cohort:
        cohorts = obs['cohort'].unique().sort().to_list()
    else:
        cohorts = [None]

    signals = obs[signal_column].unique().to_list()

    if verbose:
        print(f"Signals: {len(signals)}")
        print(f"Cohorts: {len(cohorts)}")

    # Sequential per-cohort computation
    results = []
    total_items = 0

    for ci, cohort in enumerate(cohorts):
        # Filter observations to this cohort
        if cohort is not None:
            cohort_obs = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_obs = obs

        # Get granger pairs for THIS cohort (not globally)
        if has_granger_col:
            if has_cohort_pw and cohort is not None:
                cohort_pw = pairwise.filter(pl.col('cohort') == cohort)
            else:
                cohort_pw = pairwise
            pairs_list = (
                cohort_pw.filter(pl.col('needs_granger') == True)
                .select(['signal_a', 'signal_b']).unique().to_dicts()
            )
            # Fallback: if no pairs flagged AND pc1_coloading is all zero
            if len(pairs_list) == 0 and 'pc1_coloading' in cohort_pw.columns:
                if (cohort_pw['pc1_coloading'] == 0.0).all():
                    pairs_list = cohort_pw.select(['signal_a', 'signal_b']).unique().to_dicts()
        else:
            pairs_list = pairwise.select(['signal_a', 'signal_b']).unique().to_dicts()

        # Build signal lookup for this cohort only
        signal_data = {}
        for signal in signals:
            sig = cohort_obs.filter(pl.col(signal_column) == signal).sort(index_column)
            values = sig[value_column].to_numpy()
            values = values[~np.isnan(values)]
            if len(values) >= min_samples:
                signal_data[signal] = values

        # Compute pairs within this cohort
        for pair in pairs_list:
            signal_a = pair['signal_a']
            signal_b = pair['signal_b']

            if signal_a not in signal_data or signal_b not in signal_data:
                continue

            x = signal_data[signal_a]
            y = signal_data[signal_b]

            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            if len(x) < min_samples:
                continue

            row = _compute_pair((signal_a, signal_b, x, y))
            if cohort is not None:
                row['cohort'] = cohort
            results.append(row)
            total_items += 1

        if verbose and (ci + 1) % 10 == 0:
            print(f"  Processed {ci + 1}/{len(cohorts)} cohorts ({total_items} pairs so far)...")

    # Build DataFrame (high infer_schema_length because copula_best_family is str
    # that may be null in early rows)
    result = pl.DataFrame(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    # Post-process: replace inf with null in float columns EXCEPT sentinel columns.
    # half_life uses inf as sentinel for "no decay detected" (not mean-reverting).
    if len(result) > 0:
        sentinel_cols = {'half_life'}  # inf is meaningful for these columns
        float_cols = [
            c for c in result.columns
            if result[c].dtype in [pl.Float64, pl.Float32]
            and c not in sentinel_cols
        ]
        for col in float_cols:
            result = result.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
        write_output(result, data_path, 'information_flow', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0 and 'granger_f_a_to_b' in result.columns:
            valid_granger = result.filter(pl.col('granger_f_a_to_b').is_not_null())
            if len(valid_granger) > 0:
                print(f"\nGranger causality computed: {len(valid_granger)} pairs")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Information Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes causality metrics for eigenvector-gated pairs.

Uses needs_granger flag from signal_pairwise to avoid computing
causality for all O(n²) pairs. Only pairs with high PC co-loading
(indicating correlation) get Granger analysis to determine direction.

Example:
  python -m engines.entry_points.stage_10_information_flow \\
      observations.parquet signal_pairwise.parquet \\
      -o information_flow.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('signal_pairwise', help='Path to signal_pairwise.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum samples per signal (default: 100)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.signal_pairwise,
        args.data_path,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

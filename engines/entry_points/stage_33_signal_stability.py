"""
Stage 33: Rolling Signal Stability
===================================

Per-signal Hilbert and Wavelet stability metrics, aggregated across
signals per cohort per rolling window.

Replaces FTLE as the stability indicator:
- FTLE: 200 sample minimum, 7 windows, 32% test coverage
- Hilbert+Wavelet: 8 sample minimum, full windows, ~100% coverage

Uses same rolling architecture as Stage 20 (sensor_eigendecomp):
- Level 1: Raw observations in rolling windows
- Level 2: Compute Hilbert + Wavelet per signal per window
- Aggregate across signals

Input:
    observations.parquet (cohort, signal_id, I, value)

Output:
    signal_stability.parquet (one row per cohort per window)
    - 10 Hilbert aggregate columns (h_*)
    - 10 Wavelet aggregate columns (w_*)
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path

from engines.manifold.signal.hilbert_stability import compute as compute_hilbert
from engines.manifold.signal.wavelet_stability import compute as compute_wavelet


# ── Hilbert aggregation columns ──
HILBERT_AGG_COLS = [
    'h_freq_stability_mean',
    'h_freq_stability_min',
    'h_freq_std_max',
    'h_freq_drift_mean',
    'h_freq_drift_spread',
    'h_amp_cv_mean',
    'h_amp_trend_mean',
    'h_phase_coherence_mean',
    'h_phase_coherence_min',
    'h_am_fm_ratio_mean',
]

# ── Wavelet aggregation columns ──
WAVELET_AGG_COLS = [
    'w_energy_low_mean',
    'w_energy_high_mean',
    'w_energy_ratio_mean',
    'w_entropy_mean',
    'w_concentration_mean',
    'w_energy_drift_mean',
    'w_energy_drift_max',
    'w_temporal_std_mean',
    'w_intermittency_mean',
    'w_intermittency_max',
]


def _safe_mean(vals):
    """Mean ignoring NaN, returns NaN if all NaN or empty."""
    valid = [v for v in vals if np.isfinite(v)]
    return float(np.mean(valid)) if valid else np.nan


def _safe_min(vals):
    """Min ignoring NaN."""
    valid = [v for v in vals if np.isfinite(v)]
    return float(np.min(valid)) if valid else np.nan


def _safe_max(vals):
    """Max ignoring NaN."""
    valid = [v for v in vals if np.isfinite(v)]
    return float(np.max(valid)) if valid else np.nan


def _safe_std(vals):
    """Std ignoring NaN."""
    valid = [v for v in vals if np.isfinite(v)]
    return float(np.std(valid)) if len(valid) > 1 else np.nan


def aggregate_hilbert(metrics_list):
    """Aggregate per-signal Hilbert metrics across signals."""
    if not metrics_list:
        return {k: np.nan for k in HILBERT_AGG_COLS}

    return {
        'h_freq_stability_mean': _safe_mean([m['inst_freq_stability'] for m in metrics_list]),
        'h_freq_stability_min': _safe_min([m['inst_freq_stability'] for m in metrics_list]),
        'h_freq_std_max': _safe_max([m['inst_freq_std'] for m in metrics_list]),
        'h_freq_drift_mean': _safe_mean([m['inst_freq_drift'] for m in metrics_list]),
        'h_freq_drift_spread': _safe_std([m['inst_freq_drift'] for m in metrics_list]),
        'h_amp_cv_mean': _safe_mean([m['inst_amp_cv'] for m in metrics_list]),
        'h_amp_trend_mean': _safe_mean([m['inst_amp_trend'] for m in metrics_list]),
        'h_phase_coherence_mean': _safe_mean([m['phase_coherence'] for m in metrics_list]),
        'h_phase_coherence_min': _safe_min([m['phase_coherence'] for m in metrics_list]),
        'h_am_fm_ratio_mean': _safe_mean([m['am_fm_ratio'] for m in metrics_list]),
    }


def aggregate_wavelet(metrics_list):
    """Aggregate per-signal wavelet metrics across signals."""
    if not metrics_list:
        return {k: np.nan for k in WAVELET_AGG_COLS}

    return {
        'w_energy_low_mean': _safe_mean([m['wavelet_energy_low'] for m in metrics_list]),
        'w_energy_high_mean': _safe_mean([m['wavelet_energy_high'] for m in metrics_list]),
        'w_energy_ratio_mean': _safe_mean([m['wavelet_energy_ratio'] for m in metrics_list]),
        'w_entropy_mean': _safe_mean([m['wavelet_entropy'] for m in metrics_list]),
        'w_concentration_mean': _safe_mean([m['wavelet_concentration'] for m in metrics_list]),
        'w_energy_drift_mean': _safe_mean([m['wavelet_energy_drift'] for m in metrics_list]),
        'w_energy_drift_max': _safe_max([m['wavelet_energy_drift'] for m in metrics_list]),
        'w_temporal_std_mean': _safe_mean([m['wavelet_temporal_std'] for m in metrics_list]),
        'w_intermittency_mean': _safe_mean([m['wavelet_intermittency'] for m in metrics_list]),
        'w_intermittency_max': _safe_max([m['wavelet_intermittency'] for m in metrics_list]),
    }


def run(
    observations_path: str,
    output_path: str = "signal_stability.parquet",
    window_size: int = 30,
    stride: int = 5,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute rolling signal stability metrics per cohort.

    For each cohort, slides a window over the observation index,
    computes per-signal Hilbert and Wavelet metrics at each window,
    then aggregates across signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for signal_stability.parquet
        window_size: Number of observations per window
        stride: Stride between windows
        verbose: Print progress

    Returns:
        DataFrame with per-cohort, per-window stability metrics
    """
    if verbose:
        print("=" * 70)
        print("STAGE 33: ROLLING SIGNAL STABILITY")
        print("Per-signal Hilbert + Wavelet, aggregated across signals")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)

    has_cohort = 'cohort' in obs.columns
    cohorts = sorted(obs['cohort'].unique().to_list()) if has_cohort else ['all']
    signal_ids = sorted(obs['signal_id'].unique().to_list())

    if verbose:
        print(f"\nLoaded observations: {obs.shape}")
        print(f"Cohorts: {len(cohorts)}, Signals: {len(signal_ids)}")
        print(f"Window: {window_size}, Stride: {stride}")

    all_results = []

    for ci, cohort in enumerate(cohorts):
        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        # Pivot to wide: rows=I, cols=signal_id
        try:
            wide = cohort_data.pivot(
                values='value', index='I', on='signal_id',
            ).sort('I')
        except Exception:
            continue

        available_signals = sorted([c for c in wide.columns if c != 'I'])
        if not available_signals:
            continue

        I_vals = wide['I'].to_numpy()
        n_obs = len(I_vals)

        effective_window = min(window_size, n_obs)
        if effective_window < 8:
            effective_window = n_obs

        # Get signal arrays once
        signal_arrays = {}
        for sig in available_signals:
            signal_arrays[sig] = wide[sig].to_numpy().astype(float)

        # Rolling windows
        for start in range(0, n_obs - effective_window + 1, stride):
            end = start + effective_window
            window_I = int(I_vals[end - 1])

            hilbert_metrics = []
            wavelet_metrics = []

            for sig in available_signals:
                sig_vals = signal_arrays[sig][start:end]
                valid = sig_vals[np.isfinite(sig_vals)]

                if len(valid) >= 4:
                    h = compute_hilbert(valid)
                    hilbert_metrics.append(h)

                if len(valid) >= 8:
                    w = compute_wavelet(valid)
                    wavelet_metrics.append(w)

            row = {
                'cohort': cohort,
                'I': window_I,
                'n_signals': len(available_signals),
                'n_hilbert_signals': len(hilbert_metrics),
                'n_wavelet_signals': len(wavelet_metrics),
            }
            row.update(aggregate_hilbert(hilbert_metrics))
            row.update(aggregate_wavelet(wavelet_metrics))
            all_results.append(row)

        if verbose and (ci + 1) % 50 == 0:
            print(f"  Processed {ci + 1}/{len(cohorts)} cohorts ({len(all_results)} windows)")

    # Build output
    if all_results:
        df = pl.DataFrame(all_results)
    else:
        schema = {
            'cohort': pl.Utf8,
            'I': pl.Int64,
            'n_signals': pl.Int64,
            'n_hilbert_signals': pl.Int64,
            'n_wavelet_signals': pl.Int64,
        }
        for col in HILBERT_AGG_COLS + WAVELET_AGG_COLS:
            schema[col] = pl.Float64
        df = pl.DataFrame(schema=schema)

    df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")

        if len(df) > 0:
            print(f"Cohorts with data: {df['cohort'].n_unique()}")

            # Hilbert summary
            valid_h = df.filter(
                pl.col('h_freq_stability_mean').is_not_null() &
                pl.col('h_freq_stability_mean').is_not_nan()
            )
            if len(valid_h) > 0:
                print(f"\nHilbert freq stability (mean across signals):")
                print(f"  Mean:   {valid_h['h_freq_stability_mean'].mean():.3f}")
                print(f"  Median: {valid_h['h_freq_stability_mean'].median():.3f}")

            # Wavelet summary
            valid_w = df.filter(
                pl.col('w_entropy_mean').is_not_null() &
                pl.col('w_entropy_mean').is_not_nan()
            )
            if len(valid_w) > 0:
                print(f"\nWavelet entropy (mean across signals):")
                print(f"  Mean:   {valid_w['w_entropy_mean'].mean():.3f}")
                print(f"  Median: {valid_w['w_entropy_mean'].median():.3f}")

        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 33: Rolling Signal Stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Per-signal Hilbert and Wavelet stability metrics, aggregated across
signals per cohort per rolling window.

Output schema:
  cohort, I, n_signals, n_hilbert_signals, n_wavelet_signals,
  h_freq_stability_mean, h_freq_stability_min, h_freq_std_max,
  h_freq_drift_mean, h_freq_drift_spread, h_amp_cv_mean,
  h_amp_trend_mean, h_phase_coherence_mean, h_phase_coherence_min,
  h_am_fm_ratio_mean, w_energy_low_mean, w_energy_high_mean,
  w_energy_ratio_mean, w_entropy_mean, w_concentration_mean,
  w_energy_drift_mean, w_energy_drift_max, w_temporal_std_mean,
  w_intermittency_mean, w_intermittency_max

Example:
  python -m engines.entry_points.stage_33_signal_stability \\
      observations.parquet -o signal_stability.parquet \\
      --window-size 30 --stride 5
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='signal_stability.parquet',
                        help='Output path (default: signal_stability.parquet)')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Rolling window size (default: 30)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride between windows (default: 5)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        window_size=args.window_size,
        stride=args.stride,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

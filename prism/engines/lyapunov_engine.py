"""
PRISM Lyapunov Engine

Computes maximal Lyapunov exponent using Rosenstein's algorithm.
Measures rate of divergence of nearby trajectories in phase space.

Output: lyapunov_max, lyapunov_mean (numbers only, no interpretation)

INTERPRETATION (for reference - ORTHON's responsibility):
├── λ > 0: Chaos (trajectories diverge)
├── λ ≈ 0: Quasi-periodic (trajectories parallel)
├── λ < 0: Stable (trajectories converge)
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False


def compute_lyapunov(
    signal: np.ndarray,
    emb_dim: int = 10,
    lag: int = None,
    min_tsep: int = None,
    trajectory_len: int = 20,
) -> Dict[str, float]:
    """
    Compute maximal Lyapunov exponent using Rosenstein algorithm.

    Args:
        signal: 1D time series
        emb_dim: Embedding dimension for phase space reconstruction
        lag: Time delay for embedding (default: first zero of autocorr)
        min_tsep: Minimum temporal separation (default: lag)
        trajectory_len: Length of trajectories to track

    Returns:
        Dict with lyapunov_max, lyapunov_mean, lyapunov_std
    """
    result = {
        'lyapunov_max': np.nan,
        'lyapunov_mean': np.nan,
        'lyapunov_std': np.nan,
        'emb_dim_used': emb_dim,
        'lag_used': np.nan,
    }

    if not HAS_NOLDS:
        result['error'] = 'nolds not installed'
        return result

    # Clean signal
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[~np.isnan(signal)]

    # Estimate lag if not provided
    if lag is None:
        lag = estimate_lag(signal)

    # Need sufficient data
    min_length = (emb_dim + 1) * lag + trajectory_len
    if len(signal) < min_length:
        result['error'] = f'insufficient_data (need {min_length}, got {len(signal)})'
        return result

    if min_tsep is None:
        min_tsep = lag

    result['lag_used'] = lag

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lyap = nolds.lyap_r(
                signal,
                emb_dim=emb_dim,
                lag=lag,
                min_tsep=min_tsep,
                trajectory_len=trajectory_len,
            )

            result['lyapunov_max'] = float(lyap)

            # Compute over multiple trajectory lengths for robustness
            lyaps = []
            for tlen in [10, 15, 20, 25, 30]:
                if len(signal) >= (emb_dim + 1) * lag + tlen:
                    try:
                        l = nolds.lyap_r(
                            signal,
                            emb_dim=emb_dim,
                            lag=lag,
                            min_tsep=min_tsep,
                            trajectory_len=tlen,
                        )
                        if not np.isnan(l) and not np.isinf(l):
                            lyaps.append(l)
                    except Exception:
                        pass

            if lyaps:
                result['lyapunov_mean'] = float(np.mean(lyaps))
                result['lyapunov_std'] = float(np.std(lyaps))

    except Exception as e:
        result['error'] = str(e)

    return result


def estimate_lag(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Estimate optimal lag as first zero crossing of autocorrelation.
    """
    n = len(signal)
    max_lag = min(max_lag, n // 4)

    if max_lag < 1:
        return 1

    # Normalize
    signal = signal - np.mean(signal)

    # Handle zero variance
    if np.std(signal) < 1e-10:
        return 1

    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag+1]

    if autocorr[0] == 0:
        return 1

    autocorr = autocorr / autocorr[0]  # Normalize

    # Find first zero crossing
    for i in range(1, len(autocorr)):
        if autocorr[i] <= 0:
            return i

    # Fallback: first minimum
    return int(np.argmin(autocorr[1:]) + 1) if len(autocorr) > 1 else 1


def compute_lyapunov_for_signal_vector(
    signal_vector_path: str,
    observations_path: str,
    output_path: str,
    emb_dim: int = 10,
    min_window: int = 100,
    max_window: int = 1000,
    stride: int = None,
    verbose: bool = True,
) -> 'pl.DataFrame':
    """
    Compute Lyapunov exponents for all signals at each I.

    Reads signal_vector to get (signal_id, I) pairs.
    Reads observations to get raw signal values.
    Computes Lyapunov for each signal using a trailing window.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        observations_path: Path to observations.parquet
        output_path: Output path for lyapunov.parquet
        emb_dim: Embedding dimension
        min_window: Minimum trailing window size
        max_window: Maximum trailing window size (cap for efficiency)
        stride: Compute every Nth I point (None = compute all)
        verbose: Print progress

    Returns:
        DataFrame with lyapunov metrics per (signal_id, I)
    """
    import polars as pl
    import time

    signal_vector = pl.read_parquet(signal_vector_path)
    observations = pl.read_parquet(observations_path)

    # Get unique (signal_id, I) pairs from signal_vector
    pairs = signal_vector.select(['signal_id', 'I']).unique().sort(['signal_id', 'I'])

    # Apply stride if specified
    if stride is not None and stride > 1:
        pairs = pairs.with_row_count('_row').filter(pl.col('_row') % stride == 0).drop('_row')

    if verbose:
        print(f"  Computing Lyapunov for {len(pairs)} (signal_id, I) pairs...")
        start_time = time.time()

    results = []

    # Determine column names
    signal_col = 'signal_id' if 'signal_id' in observations.columns else 'signal_name'
    value_col = 'value' if 'value' in observations.columns else 'y'

    signals = pairs['signal_id'].unique().to_list()
    total_computed = 0

    for idx, signal_id in enumerate(signals):
        if verbose and idx > 0 and idx % max(1, len(signals) // 10) == 0:
            elapsed = time.time() - start_time
            rate = total_computed / elapsed if elapsed > 0 else 0
            print(f"    Progress: {idx}/{len(signals)} signals ({total_computed} computed, {rate:.1f}/sec)")

        # Get signal data sorted by I
        sig_data = (
            observations
            .filter(pl.col(signal_col) == signal_id)
            .sort('I')
        )

        if len(sig_data) == 0:
            continue

        I_values = sig_data['I'].to_numpy()
        values = sig_data[value_col].to_numpy()

        # Get I points for this signal from signal_vector
        signal_I_points = (
            pairs
            .filter(pl.col('signal_id') == signal_id)
            ['I']
            .to_list()
        )

        for target_I in signal_I_points:
            # Use trailing window up to target_I, capped at max_window
            mask = I_values <= target_I
            window_values = values[mask]

            # Cap window size for efficiency
            if len(window_values) > max_window:
                window_values = window_values[-max_window:]

            # Need minimum data for Lyapunov
            if len(window_values) < min_window:
                results.append({
                    'signal_id': signal_id,
                    'I': target_I,
                    'lyapunov_max': np.nan,
                    'lyapunov_mean': np.nan,
                    'lyapunov_std': np.nan,
                })
                continue

            # Compute Lyapunov
            lyap = compute_lyapunov(window_values, emb_dim=emb_dim)
            total_computed += 1

            results.append({
                'signal_id': signal_id,
                'I': target_I,
                'lyapunov_max': lyap['lyapunov_max'],
                'lyapunov_mean': lyap['lyapunov_mean'],
                'lyapunov_std': lyap['lyapunov_std'],
            })

    df = pl.DataFrame(results)
    df.write_parquet(output_path)

    if verbose:
        elapsed = time.time() - start_time
        valid = df.filter(pl.col('lyapunov_max').is_not_null() & pl.col('lyapunov_max').is_not_nan())
        print(f"  lyapunov.parquet: {len(df):,} rows ({elapsed:.1f}s)")
        print(f"    Valid: {len(valid)}/{len(df)}")
        if len(valid) > 0:
            print(f"    λ_max range: [{valid['lyapunov_max'].min():.4f}, {valid['lyapunov_max'].max():.4f}]")

    return df

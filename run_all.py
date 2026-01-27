#!/usr/bin/env python3
"""
PRISM Pipeline - One Command, Full Analysis

Usage:
    python run_all.py

Input:
    data/observations.parquet   (canonical schema: entity_id, signal_id, I, y, unit)

Output:
    data/primitives.parquet         (per-signal metrics)
    data/primitives_pairs.parquet   (pairwise relationships)
    data/manifold.parquet           (phase space coordinates)

No CLI flags. No prompts. No configuration.
Data determines what engines run.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Add prism to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# CONFIGURATION (internal, not user-facing)
# =============================================================================

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "observations.parquet"
OUTPUT_PRIMITIVES = DATA_DIR / "primitives.parquet"
OUTPUT_PAIRS = DATA_DIR / "primitives_pairs.parquet"
OUTPUT_MANIFOLD = DATA_DIR / "manifold.parquet"

# Limits for speed (full data used for stats, sampled for expensive engines)
MAX_POINTS_EXPENSIVE = 5000  # For Lyapunov, GARCH, etc.
MAX_POINTS_MODERATE = 10000  # For Hurst, entropy, etc.
MAX_PAIRS = 50  # Top N signal pairs for pairwise analysis


# =============================================================================
# SIGNAL TYPE DETECTION
# =============================================================================

def detect_signal_types(obs: pl.DataFrame) -> dict:
    """
    Automatically detect signal types based on data properties.

    Returns dict: {signal_id: {class, is_constant, is_periodic, ...}}
    """
    typology = {}

    for signal_id in obs['signal_id'].unique().sort().to_list():
        signal_data = obs.filter(pl.col('signal_id') == signal_id)
        y = signal_data['y'].to_numpy()

        n = len(y)
        std = np.std(y)
        unique_ratio = len(np.unique(y)) / max(n, 1)

        # Detect type
        is_constant = std < 1e-10
        is_digital = unique_ratio < 0.05 and not is_constant
        is_periodic = False  # Would need FFT to detect properly

        if is_constant:
            signal_class = 'constant'
        elif is_digital:
            signal_class = 'digital'
        else:
            signal_class = 'analog'

        typology[signal_id] = {
            'class': signal_class,
            'is_constant': is_constant,
            'is_periodic': is_periodic,
            'n_points': n,
            'std': std,
            'unique_ratio': unique_ratio,
        }

    return typology


# =============================================================================
# ENGINE MAPPING
# =============================================================================

def map_engines_to_signals(typology: dict) -> dict:
    """
    Map engines to signals based on detected types.
    No user input. Fully automatic.
    """
    engine_map = {}

    for signal_id, props in typology.items():
        engines = ['basic_stats']  # All signals get basic stats

        if props['is_constant']:
            # Constant signals: only basic stats
            pass
        elif props['class'] == 'digital':
            # Digital signals: state analysis
            engines.extend(['transition_count', 'state_duration'])
        else:
            # Analog signals: full suite
            engines.extend([
                'derivatives',
                'hurst',
                'entropy',
                'fft',
                'acf_decay',
                'spectral_slope',
            ])
            # Expensive engines only for non-constant
            if props['std'] > 1e-6:
                engines.extend(['lyapunov', 'garch'])

        engine_map[signal_id] = engines

    return engine_map


# =============================================================================
# ENGINE RUNNERS
# =============================================================================

def run_basic_stats(y: np.ndarray) -> dict:
    """Basic statistics for any signal."""
    return {
        'n_points': len(y),
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'y_median': float(np.median(y)),
    }


def run_derivatives(y: np.ndarray) -> dict:
    """Compute derivatives."""
    dy = np.diff(y)
    d2y = np.diff(dy)
    return {
        'dy_mean': float(np.mean(dy)) if len(dy) > 0 else 0.0,
        'dy_std': float(np.std(dy)) if len(dy) > 0 else 0.0,
        'd2y_mean': float(np.mean(d2y)) if len(d2y) > 0 else 0.0,
    }


def run_hurst(y: np.ndarray) -> dict:
    """Hurst exponent via DFA."""
    try:
        from prism.engines.core.hurst import compute
        result = compute(y[:MAX_POINTS_MODERATE])
        return {'hurst': result.get('hurst', 0.5), 'hurst_r2': result.get('r2', 0.0)}
    except Exception:
        return {'hurst': np.nan, 'hurst_r2': np.nan}


def run_entropy(y: np.ndarray) -> dict:
    """Sample entropy."""
    try:
        from prism.engines.core.entropy import compute
        result = compute(y[:MAX_POINTS_MODERATE])
        return {'entropy': result.get('sample_entropy', np.nan)}
    except Exception:
        return {'entropy': np.nan}


def run_fft(y: np.ndarray) -> dict:
    """FFT dominant frequency."""
    try:
        from prism.engines.core.fft import compute
        result = compute(y[:MAX_POINTS_MODERATE])
        return {
            'dominant_freq': result.get('dominant_frequency', np.nan),
            'spectral_centroid': result.get('spectral_centroid', np.nan),
        }
    except Exception:
        return {'dominant_freq': np.nan, 'spectral_centroid': np.nan}


def run_lyapunov(y: np.ndarray) -> dict:
    """Lyapunov exponent."""
    try:
        from prism.engines.core.lyapunov import compute
        result = compute(y[:MAX_POINTS_EXPENSIVE])
        return {'lyapunov': result.get('lyapunov', np.nan)}
    except Exception:
        return {'lyapunov': np.nan}


def run_garch(y: np.ndarray) -> dict:
    """GARCH volatility model."""
    try:
        from prism.engines.core.garch import compute
        result = compute(y[:MAX_POINTS_EXPENSIVE])
        return {
            'garch_omega': result.get('omega', np.nan),
            'garch_alpha': result.get('alpha', np.nan),
            'garch_beta': result.get('beta', np.nan),
        }
    except Exception:
        return {'garch_omega': np.nan, 'garch_alpha': np.nan, 'garch_beta': np.nan}


def run_acf_decay(y: np.ndarray) -> dict:
    """ACF decay rate."""
    try:
        from prism.engines.core.acf_decay import compute
        result = compute(y[:MAX_POINTS_MODERATE])
        return {'acf_decay_rate': result.get('decay_rate', np.nan)}
    except Exception:
        return {'acf_decay_rate': np.nan}


def run_spectral_slope(y: np.ndarray) -> dict:
    """Spectral slope (1/f noise)."""
    try:
        from prism.engines.core.spectral_slope import compute
        result = compute(y[:MAX_POINTS_MODERATE])
        return {'spectral_slope': result.get('slope', np.nan)}
    except Exception:
        return {'spectral_slope': np.nan}


ENGINE_FUNCTIONS = {
    'basic_stats': run_basic_stats,
    'derivatives': run_derivatives,
    'hurst': run_hurst,
    'entropy': run_entropy,
    'fft': run_fft,
    'lyapunov': run_lyapunov,
    'garch': run_garch,
    'acf_decay': run_acf_decay,
    'spectral_slope': run_spectral_slope,
    'transition_count': lambda y: {'n_transitions': int(np.sum(np.diff(y) != 0))},
    'state_duration': lambda y: {'mean_state_duration': float(len(y) / max(np.sum(np.diff(y) != 0), 1))},
}


# =============================================================================
# PAIRWISE ANALYSIS
# =============================================================================

def compute_pairwise(obs: pl.DataFrame, signals: list, typology: dict) -> list:
    """Compute pairwise relationships between signals."""
    pairs = []

    # Filter to analog signals only (constant/digital don't have meaningful correlation)
    analog_signals = [s for s in signals if typology.get(s, {}).get('class') == 'analog']

    if len(analog_signals) < 2:
        return pairs

    # Get all signal combinations (limit for speed)
    from itertools import combinations
    signal_pairs = list(combinations(analog_signals[:MAX_PAIRS], 2))

    # Use first entity for speed
    entity = obs['entity_id'].unique()[0]

    for s1, s2 in signal_pairs:
        d1 = obs.filter((pl.col('signal_id') == s1) & (pl.col('entity_id') == entity)).sort('I')
        d2 = obs.filter((pl.col('signal_id') == s2) & (pl.col('entity_id') == entity)).sort('I')

        if len(d1) < 10 or len(d2) < 10:
            continue

        y1 = d1['y'].to_numpy()[:2000]
        y2 = d2['y'].to_numpy()[:2000]

        # Truncate to same length
        min_len = min(len(y1), len(y2))
        y1, y2 = y1[:min_len], y2[:min_len]

        if min_len < 10:
            continue

        # Correlation (only if both have variance)
        std1, std2 = np.std(y1), np.std(y2)
        if std1 > 1e-10 and std2 > 1e-10:
            corr = float(np.corrcoef(y1, y2)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0  # Constant signals have no correlation, not NaN

        pairs.append({
            'signal_a': s1,
            'signal_b': s2,
            'correlation': corr,
        })

    return pairs


# =============================================================================
# MANIFOLD EMBEDDING
# =============================================================================

def compute_manifold(primitives: pl.DataFrame) -> pl.DataFrame:
    """Compute manifold coordinates via PCA."""
    try:
        from sklearn.decomposition import PCA

        # Select numeric columns for embedding
        numeric_cols = [c for c in primitives.columns
                       if c not in ['entity_id', 'signal_id', 'signal_class']
                       and primitives[c].dtype in [pl.Float64, pl.Float32, pl.Int64]]

        if not numeric_cols or len(primitives) < 3:
            return primitives.select(['entity_id', 'signal_id']).with_columns([
                pl.lit(0.0).alias('manifold_x'),
                pl.lit(0.0).alias('manifold_y'),
                pl.lit(0.0).alias('manifold_z'),
            ])

        # Build feature matrix
        matrix = primitives.select(numeric_cols).to_numpy()

        # Replace NaN with column means
        col_means = np.nanmean(matrix, axis=0)
        for i in range(matrix.shape[1]):
            mask = np.isnan(matrix[:, i])
            matrix[mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0

        # PCA with sklearn directly
        n_components = min(3, matrix.shape[0], matrix.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(matrix)

        # Pad to 3D if needed
        if coords.shape[1] < 3:
            padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
            coords = np.hstack([coords, padding])

        return primitives.select(['entity_id', 'signal_id']).with_columns([
            pl.Series('manifold_x', coords[:, 0]),
            pl.Series('manifold_y', coords[:, 1]),
            pl.Series('manifold_z', coords[:, 2]),
        ])
    except Exception as e:
        print(f"  Manifold computation failed: {e}")
        return primitives.select(['entity_id', 'signal_id']).with_columns([
            pl.lit(0.0).alias('manifold_x'),
            pl.lit(0.0).alias('manifold_y'),
            pl.lit(0.0).alias('manifold_z'),
        ])


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    THE PIPELINE.

    One command. Everything runs. Results appear.
    """
    start_time = time.time()

    print("="*70)
    print("PRISM PIPELINE")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading observations...")

    if not INPUT_FILE.exists():
        print(f"  ERROR: {INPUT_FILE} not found")
        print(f"  Required schema: entity_id, signal_id, I, y, unit")
        sys.exit(1)

    obs = pl.read_parquet(INPUT_FILE)
    n_obs = len(obs)
    n_entities = obs['entity_id'].n_unique()
    n_signals = obs['signal_id'].n_unique()

    print(f"  {n_obs:,} observations")
    print(f"  {n_entities} entities")
    print(f"  {n_signals} signals")

    # -------------------------------------------------------------------------
    # 2. DETECT TYPOLOGY
    # -------------------------------------------------------------------------
    print("\n[2/6] Detecting signal types...")

    typology = detect_signal_types(obs)

    type_counts = {}
    for props in typology.values():
        t = props['class']
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    # -------------------------------------------------------------------------
    # 3. MAP ENGINES
    # -------------------------------------------------------------------------
    print("\n[3/6] Mapping engines...")

    engine_map = map_engines_to_signals(typology)

    all_engines = set()
    for engines in engine_map.values():
        all_engines.update(engines)

    print(f"  {len(all_engines)} engines selected")

    # -------------------------------------------------------------------------
    # 4. COMPUTE PRIMITIVES
    # -------------------------------------------------------------------------
    print("\n[4/6] Computing primitives...")

    primitives_data = []
    signals = list(typology.keys())

    for i, signal_id in enumerate(signals):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Signal {i+1}/{len(signals)}: {signal_id}")

        # Get signal data (aggregate across entities for now)
        signal_data = obs.filter(pl.col('signal_id') == signal_id)
        y = signal_data['y'].to_numpy()

        # Initialize result
        result = {
            'signal_id': signal_id,
            'signal_class': typology[signal_id]['class'],
        }

        # Add entity_id (use first one for now)
        result['entity_id'] = signal_data['entity_id'][0]

        # Run mapped engines
        for engine_name in engine_map[signal_id]:
            if engine_name in ENGINE_FUNCTIONS:
                try:
                    engine_result = ENGINE_FUNCTIONS[engine_name](y)
                    result.update(engine_result)
                except Exception as e:
                    pass  # Skip failed engines silently

        primitives_data.append(result)

    primitives = pl.DataFrame(primitives_data)
    print(f"  {len(primitives)} signal primitives computed")

    # -------------------------------------------------------------------------
    # 5. COMPUTE PAIRWISE
    # -------------------------------------------------------------------------
    print("\n[5/6] Computing pairwise relationships...")

    pairs_data = compute_pairwise(obs, signals, typology)
    pairs = pl.DataFrame(pairs_data) if pairs_data else pl.DataFrame({'signal_a': [], 'signal_b': [], 'correlation': []})

    print(f"  {len(pairs)} pairs computed")

    # -------------------------------------------------------------------------
    # 6. COMPUTE MANIFOLD
    # -------------------------------------------------------------------------
    print("\n[6/6] Computing manifold embedding...")

    manifold = compute_manifold(primitives)

    print(f"  3D coordinates computed")

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Saving results...")

    primitives.write_parquet(OUTPUT_PRIMITIVES)
    print(f"  {OUTPUT_PRIMITIVES}")

    pairs.write_parquet(OUTPUT_PAIRS)
    print(f"  {OUTPUT_PAIRS}")

    manifold.write_parquet(OUTPUT_MANIFOLD)
    print(f"  {OUTPUT_MANIFOLD}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"\nOutputs:")
    print(f"  primitives.parquet      {len(primitives)} rows × {len(primitives.columns)} cols")
    print(f"  primitives_pairs.parquet {len(pairs)} pairs")
    print(f"  manifold.parquet        {len(manifold)} coordinates")

    # Show sample results
    print("\n[SAMPLE: Hurst Exponents]")
    if 'hurst' in primitives.columns:
        hurst_df = primitives.select(['signal_id', 'hurst']).sort('hurst', descending=True).head(10)
        for row in hurst_df.iter_rows(named=True):
            h = row['hurst']
            if h is not None and not np.isnan(h):
                behavior = "TRENDING" if h > 0.6 else "mean-reverting" if h < 0.4 else "random"
                print(f"  {row['signal_id']}: H={h:.3f} ({behavior})")

    print("\n[SAMPLE: Strongest Correlations]")
    if len(pairs) > 0 and 'correlation' in pairs.columns:
        top_pairs = pairs.filter(pl.col('correlation').is_not_null()).sort('correlation', descending=True).head(5)
        for row in top_pairs.iter_rows(named=True):
            print(f"  {row['signal_a']} <-> {row['signal_b']}: r={row['correlation']:.3f}")


if __name__ == '__main__':
    main()

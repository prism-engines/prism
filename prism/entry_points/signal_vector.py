"""
Signal Vector Entry Point.

Thin orchestrator that:
1. Reads manifest
2. Loads observations
3. Calls appropriate engines per-signal with per-signal config
4. Writes output to parquet

Entry point does NOT contain compute logic - only orchestration.

================================================================================
WARNING: WINDOW LOGIC BELONGS IN THE MANIFEST - NOT HERE
================================================================================
DO NOT add any window size calculation, adaptive windowing, or default window
values to this file. All windowing parameters MUST come from the manifest.

If the manifest is missing window_size or stride, this entry point MUST fail.
No defaults. No fallbacks. No "smart" calculations.

ORTHON determines window parameters. PRISM executes what the manifest says.
Hardcoding window logic here wrecks everything downstream.
================================================================================
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Set
from collections import defaultdict


# =============================================================================
# ENGINE REQUIREMENTS
# =============================================================================
# Minimum samples required for each engine to produce valid results.
# Engines not listed default to 4 (absolute minimum for any computation).
#
# These define the "ideal" window size for accurate results. Engines receiving
# smaller windows will return NaN for those observations.
# =============================================================================

ENGINE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # FFT-based engines (need sufficient frequency resolution)
    'spectral': {'min_samples': 64},
    'harmonics': {'min_samples': 64},
    'fundamental_freq': {'min_samples': 64},
    'thd': {'min_samples': 64},
    'frequency_bands': {'min_samples': 64},
    'band_power': {'min_samples': 64},

    # Entropy engines (need sufficient data for pattern detection)
    'sample_entropy': {'min_samples': 64},
    'complexity': {'min_samples': 50},
    'approximate_entropy': {'min_samples': 30},
    'permutation_entropy': {'min_samples': 20},
    'perm_entropy': {'min_samples': 20},

    # Fractal/memory engines (need long series for scaling analysis)
    'hurst': {'min_samples': 128},
    'dfa': {'min_samples': 20},
    'memory': {'min_samples': 20},
    'acf_decay': {'min_samples': 16},

    # Statistical engines (low requirements)
    'statistics': {'min_samples': 4},
    'kurtosis': {'min_samples': 4},
    'skewness': {'min_samples': 4},
    'crest_factor': {'min_samples': 4},

    # Spectral analysis (moderate requirements)
    'snr': {'min_samples': 32},
    'phase_coherence': {'min_samples': 32},

    # Trend engines
    'trend': {'min_samples': 8},
    'mann_kendall': {'min_samples': 8},
    'rate_of_change': {'min_samples': 4},

    # Advanced engines
    'attractor': {'min_samples': 64},
    'lyapunov': {'min_samples': 128},
    'garch': {'min_samples': 64},
    'dmd': {'min_samples': 32},
    'envelope': {'min_samples': 16},
    'variance_growth': {'min_samples': 16},

    # Domain-specific
    'basin': {'min_samples': 32},
    'cycle_counting': {'min_samples': 16},
    'lof': {'min_samples': 20},
    'pulsation_index': {'min_samples': 8},
    'time_constant': {'min_samples': 16},
}

# Default minimum for unlisted engines
DEFAULT_MIN_SAMPLES = 4


def get_engine_min_samples(engine_name: str) -> int:
    """Get minimum samples required for an engine."""
    return ENGINE_REQUIREMENTS.get(engine_name, {}).get('min_samples', DEFAULT_MIN_SAMPLES)


def validate_engine_can_run(engine_name: str, window_size: int) -> bool:
    """Check if engine can run with given window size."""
    min_required = get_engine_min_samples(engine_name)
    return window_size >= min_required


def group_engines_by_window(
    engines: List[str],
    overrides: Dict[str, int],
    default_window: int,
) -> Dict[int, List[str]]:
    """
    Group engines by their required window size.

    Args:
        engines: List of engine names
        overrides: Dict of {engine_name: window_size} from manifest
        default_window: System default window size

    Returns:
        dict: {window_size: [engine_list]}
    """
    groups: Dict[int, List[str]] = {}

    for engine in engines:
        window = overrides.get(engine, default_window)
        if window not in groups:
            groups[window] = []
        groups[window].append(engine)

    return groups


def _load_engine_registry() -> Dict[str, Callable]:
    """Load all signal engines. Each engine has a compute() method."""
    from prism.engines.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
        basin, cycle_counting, lof, pulsation_index, time_constant,
        rate_of_change, variance_growth,
        fundamental_freq, phase_coherence, snr, thd,
    )

    return {
        # Core engines
        'statistics': statistics.compute,
        'memory': memory.compute,
        'complexity': complexity.compute,
        'spectral': spectral.compute,
        'trend': trend.compute,

        # Individual statistic engines
        'kurtosis': statistics.compute_kurtosis,
        'skewness': statistics.compute_skewness,
        'crest_factor': statistics.compute_crest_factor,

        # Individual memory engines
        'hurst': hurst.compute,
        'dfa': memory.compute_dfa,
        'acf_decay': memory.compute_acf_decay,

        # Individual complexity engines
        'sample_entropy': complexity.compute_sample_entropy,
        'permutation_entropy': complexity.compute_permutation_entropy,
        'perm_entropy': complexity.compute_permutation_entropy,  # alias
        'approximate_entropy': complexity.compute_approximate_entropy,

        # Individual trend engines
        'mann_kendall': trend.compute_mann_kendall,
        'rate_of_change': trend.compute_rate_of_change,

        # Trend aliases (trend.compute returns these)
        'trend_r2': trend.compute,
        'detrend_std': trend.compute,
        'cusum': trend.compute,

        # Spectral aliases (spectral.compute returns these)
        'spectral_entropy': spectral.compute,

        # Frequency band aliases
        'band_power': frequency_bands.compute,
        'frequency_bands': frequency_bands.compute,

        # Advanced engines
        'attractor': attractor.compute,
        'lyapunov': lyapunov.compute,
        'garch': garch.compute,
        'dmd': dmd.compute,
        'envelope': envelope.compute,
        'harmonics': harmonics.compute,

        # Domain-specific engines
        'basin': basin.compute,
        'cycle_counting': cycle_counting.compute,
        'lof': lof.compute,
        'pulsation_index': pulsation_index.compute,
        'time_constant': time_constant.compute,

        # Rate of change (detailed version with mean_rate, max_rate, etc.)
        'rate_of_change_detailed': rate_of_change.compute,

        # Variance growth (non-stationarity detection)
        'variance_growth': variance_growth.compute,

        # Spectral analysis engines
        'fundamental_freq': fundamental_freq.compute,
        'phase_coherence': phase_coherence.compute,
        'snr': snr.compute,
        'thd': thd.compute,
    }


# Global engine registry (loaded once)
_ENGINE_REGISTRY: Dict[str, Callable] = None


def _get_engine_registry() -> Dict[str, Callable]:
    """Get or load the engine registry."""
    global _ENGINE_REGISTRY
    if _ENGINE_REGISTRY is None:
        _ENGINE_REGISTRY = _load_engine_registry()
    return _ENGINE_REGISTRY


def get_signal_data(
    observations: pl.DataFrame,
    cohort_name: str,
    signal_id: str,
) -> np.ndarray:
    """
    Extract signal data from observations.

    Args:
        observations: Observations DataFrame
        cohort_name: Cohort name (unused, signals identified by signal_id)
        signal_id: Signal identifier

    Returns:
        numpy array of signal values sorted by I
    """
    signal_data = (
        observations
        .filter(pl.col('signal_id') == signal_id)
        .sort('I')
    )
    return signal_data['value'].to_numpy()


def run_engine(engine_name: str, window_data: np.ndarray) -> Dict[str, Any]:
    """
    Run a single engine on window data.

    Args:
        engine_name: Name of the engine
        window_data: numpy array of window values

    Returns:
        Dict of {output_key: value}
    """
    registry = _get_engine_registry()
    if engine_name not in registry:
        return {}
    return registry[engine_name](window_data)


def null_output_for_engine(engine_name: str) -> Dict[str, float]:
    """
    Get NaN output for an engine that can't run (insufficient data).

    Args:
        engine_name: Name of the engine

    Returns:
        Dict of {output_key: np.nan}
    """
    registry = _get_engine_registry()
    if engine_name not in registry:
        return {}

    try:
        # Call with minimal data to get output structure
        sample_output = registry[engine_name](np.array([0.0, 0.0, 0.0, 0.0]))
        return {k: np.nan for k in sample_output.keys()}
    except Exception:
        return {}


def _validate_engines(
    engine_names: List[str],
    registry: Dict[str, Callable]
) -> Tuple[List[str], List[str]]:
    """Validate engine names against registry. Returns (valid, unknown)."""
    valid = [name for name in engine_names if name in registry]
    unknown = [name for name in engine_names if name not in registry]
    return valid, unknown


def _diagnose_manifest_engines(
    manifest: Dict[str, Any],
    registry: Dict[str, Callable]
) -> Dict[str, Any]:
    """Check all manifest engines against registry."""
    all_engines: Set[str] = set()
    for cohort_signals in manifest.get('cohorts', {}).values():
        for signal_config in cohort_signals.values():
            all_engines.update(signal_config.get('engines', []))

    available = [e for e in all_engines if e in registry]
    missing = [e for e in all_engines if e not in registry]
    coverage = len(available) / len(all_engines) if all_engines else 1.0

    return {
        'available': sorted(available),
        'missing': sorted(missing),
        'coverage': coverage,
        'total_requested': len(all_engines),
    }


def compute_signal_vector(
    observations: pl.DataFrame,
    manifest: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute signal vector with per-engine window support.

    This is the core computation function that processes observations
    according to the manifest specification.

    Args:
        observations: Observations DataFrame with signal_id, I, value columns
        manifest: Manifest dict with system, cohorts, engine_windows sections

    Returns:
        DataFrame with computed features per signal per window
    """
    system_window = manifest['system']['window']
    system_stride = manifest['system']['stride']
    engine_windows = manifest.get('engine_windows', {})

    results = []

    for cohort_name, cohort_config in manifest['cohorts'].items():
        for signal_id, signal_config in cohort_config.items():
            if not isinstance(signal_config, dict):
                continue

            signal_data = get_signal_data(observations, cohort_name, signal_id)
            engines = signal_config.get('engines', [])
            overrides = signal_config.get('engine_window_overrides', {})

            if not engines or len(signal_data) == 0:
                continue

            # Group engines by window requirement
            engine_groups = group_engines_by_window(engines, overrides, system_window)

            # Compute windows at system stride
            for window_end in range(system_window - 1, len(signal_data), system_stride):
                row = {
                    'signal_id': signal_id,
                    'I': window_end,
                }

                # Run each engine group with appropriate window
                for window_size, engine_list in engine_groups.items():
                    window_start = max(0, window_end - window_size + 1)

                    # Skip if not enough data for this window
                    if window_end - window_start + 1 < window_size:
                        # Fill with NaN for these engines
                        for engine in engine_list:
                            row.update(null_output_for_engine(engine))
                        continue

                    window_data = signal_data[window_start:window_end + 1]

                    for engine in engine_list:
                        try:
                            engine_output = run_engine(engine, window_data)
                            row.update(engine_output)
                        except Exception:
                            row.update(null_output_for_engine(engine))

                results.append(row)

    return pl.DataFrame(results) if results else pl.DataFrame()


def run(
    observations_path: str,
    output_path: str,
    manifest: Dict[str, Any],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector computation.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path to write signal_vector.parquet
        manifest: Manifest dict from ORTHON (REQUIRED)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    _validate_manifest(manifest)

    # Load engine registry for validation
    engine_registry = _get_engine_registry()

    # Validate engines upfront (PR13)
    diagnosis = _diagnose_manifest_engines(manifest, engine_registry)
    if verbose:
        if diagnosis['missing']:
            print(f"WARNING: Missing engines: {diagnosis['missing']}")
            print(f"  Coverage: {diagnosis['coverage']:.1%} ({len(diagnosis['available'])}/{diagnosis['total_requested']})")
        else:
            print(f"All {diagnosis['total_requested']} engines available")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_signals = obs['signal_id'].n_unique()
        n_obs = len(obs)
        print(f"Loaded {n_obs:,} observations across {n_signals} signals")
        system = manifest.get('system', {})
        print(f"System window={system.get('window')}, stride={system.get('stride')}")

    # Compute signal vector using core function
    df = compute_signal_vector(obs, manifest)

    # Write output
    df.write_parquet(output_path)

    if verbose:
        print(f"Wrote {len(df):,} rows to {output_path}")

    return df


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required structure."""
    if 'system' not in manifest:
        raise ValueError("Manifest missing 'system' section.")

    system = manifest['system']
    if 'window' not in system:
        raise ValueError("Manifest 'system' section missing 'window'.")
    if 'stride' not in system:
        raise ValueError("Manifest 'system' section missing 'stride'.")

    if 'cohorts' not in manifest:
        raise ValueError("Manifest missing 'cohorts' section.")

    if not manifest['cohorts']:
        raise ValueError("Manifest 'cohorts' is empty.")


def run_from_manifest(
    manifest_path: str,
    data_dir: str = None,
    output_dir: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector from manifest file.

    Args:
        manifest_path: Path to manifest.yaml
        data_dir: Directory with observations.parquet (optional, derived from manifest)
        output_dir: Directory for output (optional, derived from manifest)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    import yaml

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Derive paths from manifest if not provided
    if data_dir is None:
        # Use manifest's paths.observations or default to manifest directory
        obs_rel = manifest.get('paths', {}).get('observations', 'observations.parquet')
        obs_path = manifest_dir / obs_rel
    else:
        obs_path = Path(data_dir) / 'observations.parquet'

    if output_dir is None:
        # Use manifest directory for output
        out_path = manifest_dir / 'signal_vector.parquet'
    else:
        out_path = Path(output_dir) / 'signal_vector.parquet'

    return run(
        observations_path=str(obs_path),
        output_path=str(out_path),
        manifest=manifest,
        verbose=verbose,
    )


def main():
    """CLI entry point: python -m prism.entry_points.signal_vector <manifest.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute signal vectors from manifest',
        usage='python -m prism.entry_points.signal_vector <manifest.yaml>'
    )
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run_from_manifest(
        manifest_path=args.manifest,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

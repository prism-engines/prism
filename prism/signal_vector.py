"""
PRISM Signal Vector Runner

Creates signal_vector.parquet from typology-guided engine selection.

Flow:
    1. Load typology.parquet (engine recommendations per signal)
    2. Run Python engines (scale-invariant only)
    3. Run SQL engines (derived metrics)
    4. Output signal_vector.parquet

All engines are scale-invariant per engine_manifest.yaml.
"""

import numpy as np
import pandas as pd
import polars as pl
import duckdb
from pathlib import Path
from typing import Dict, List, Any, Set
import importlib
import warnings
import yaml

warnings.filterwarnings('ignore')


def load_manifest() -> Dict[str, Any]:
    """Load the engine manifest."""
    manifest_path = Path(__file__).parent / 'engines' / 'engine_manifest.yaml'
    if manifest_path.exists():
        with open(manifest_path) as f:
            return yaml.safe_load(f)
    return {}


ENGINE_MANIFEST = load_manifest()


# Map manifest engine names to actual Python modules
ENGINE_MODULE_MAP = {
    # Core
    'kurtosis': 'kurtosis',
    'skewness': 'skewness',
    'crest_factor': 'crest_factor',

    # Tail behavior
    'peak_ratio': 'peak',  # peak module computes peak_ratio

    # Periodicity
    'harmonics_ratio': 'harmonics',  # compute ratio from harmonics
    'band_ratios': 'frequency_bands',  # compute ratios from bands
    'spectral_centroid': 'spectral',
    'spectral_entropy': 'spectral',

    # Complexity
    'entropy': 'entropy',
    'hurst': 'hurst',

    # Rate
    'rate_of_change_ratio': 'rate_of_change',

    # Rolling (windowed)
    'rolling_kurtosis': 'rolling_kurtosis',
    'rolling_skewness': 'rolling_skewness',
    'rolling_entropy': 'rolling_entropy',
    'rolling_crest_factor': 'rolling_crest_factor',
}


def load_engine(engine_name: str):
    """Load an engine's compute function."""
    module_name = ENGINE_MODULE_MAP.get(engine_name, engine_name)

    # Try rolling engines first
    if engine_name.startswith('rolling_'):
        try:
            module = importlib.import_module(f'prism.engines.rolling.{module_name}')
            return module.compute
        except (ImportError, AttributeError):
            pass

    # Try signal engines
    try:
        module = importlib.import_module(f'prism.engines.signal.{module_name}')
        return module.compute
    except (ImportError, AttributeError):
        return None


class SignalVectorRunner:
    """
    Creates signal_vector.parquet using typology-guided engine selection.

    Only scale-invariant engines are run.
    """

    def __init__(
        self,
        data_dir: Path,
        window_size: int = None,
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        # Load typology
        typology_path = self.data_dir / 'typology.parquet'
        if not typology_path.exists():
            raise FileNotFoundError(f"typology.parquet not found in {data_dir}. Run typology engine first.")
        self.typology = pl.read_parquet(typology_path)

        # Load observations
        obs_path = self.data_dir / 'observations.parquet'
        if not obs_path.exists():
            raise FileNotFoundError(f"observations.parquet not found in {data_dir}")
        self.observations = pl.read_parquet(obs_path)

        # Determine window size from typology or parameter
        if window_size is not None:
            self.window_size = window_size
        else:
            active = self.typology.filter(~pl.col('is_constant'))
            if len(active) > 0:
                self.window_size = int(active['recommended_window'].min())
            else:
                self.window_size = 50

        # Get engine recommendations
        self.recommendations = self._get_recommendations()

    def _get_recommendations(self) -> Dict[str, Any]:
        """Get engine recommendations from typology."""
        from prism.engines.typology_engine import get_engine_recommendations
        return get_engine_recommendations(self.typology)

    def run(self) -> pl.DataFrame:
        """Run signal vector computation."""
        if self.verbose:
            print("=" * 70)
            print("SIGNAL VECTOR RUNNER")
            print("=" * 70)
            print(f"Data: {self.data_dir}")
            print(f"Window: {self.window_size}")
            print(f"Engines needed: {len(self.recommendations['all_engines'])}")
            print(f"Signals to process: {len(self.recommendations['active_signals'])}")
            print(f"Signals to skip: {len(self.recommendations['skip_signals'])}")
            print()

        # Step 1: Run Python engines
        if self.verbose:
            print("[PYTHON ENGINES]")
        python_results = self._run_python_engines()

        # Step 2: Run SQL engines for derived metrics
        if self.verbose:
            print("\n[SQL ENGINES]")
        sql_results = self._run_sql_engines(python_results)

        # Step 3: Combine and output
        if self.verbose:
            print("\n[OUTPUT]")
        signal_vector = self._combine_results(python_results, sql_results)

        # Save
        output_path = self.data_dir / 'signal_vector.parquet'
        signal_vector.write_parquet(output_path)

        if self.verbose:
            print(f"  signal_vector.parquet: {len(signal_vector)} rows × {len(signal_vector.columns)} cols")
            print()
            print("Columns:")
            for col in signal_vector.columns[:20]:
                print(f"  - {col}")
            if len(signal_vector.columns) > 20:
                print(f"  ... and {len(signal_vector.columns) - 20} more")

        return signal_vector

    def _run_python_engines(self) -> Dict[str, Dict]:
        """Run Python engines per signal based on typology."""
        results = {}

        # Get unique (unit_id, signal_id) combinations
        active_signals = self.recommendations['active_signals']
        signal_engines = self.recommendations['signal_engines']

        # Index observations by (unit_id, signal_id)
        obs_pd = self.observations.to_pandas()

        for (unit_id, signal_id), group in obs_pd.groupby(['unit_id', 'signal_id']):
            # Skip null signal_id (unit_id can be null, signal_id cannot)
            if signal_id is None or pd.isna(signal_id):
                continue

            if signal_id not in active_signals:
                continue  # Skip constant signals

            # Get engines for this signal
            engines_to_run = signal_engines.get(signal_id, [])
            if not engines_to_run:
                continue

            # Get signal data
            sorted_group = group.sort_values('I')
            y = sorted_group['value'].values.astype(np.float64)
            I = sorted_group['I'].values

            if len(y) < 10:
                continue

            # Run engines
            row = {
                'unit_id': unit_id,
                'signal_id': signal_id,
                'n_samples': len(y),
            }

            for engine_name in engines_to_run:
                func = load_engine(engine_name)
                if func is None:
                    continue

                try:
                    # Handle different engine signatures
                    if engine_name.startswith('rolling_'):
                        # Rolling engines return time series, take mean/std
                        result = func(y, {'window': self.window_size})
                        for key, vals in result.items():
                            if isinstance(vals, np.ndarray) and len(vals) > 0:
                                row[f'{key}_mean'] = float(np.nanmean(vals))
                                row[f'{key}_std'] = float(np.nanstd(vals))
                                row[f'{key}_last'] = float(vals[-1]) if not np.isnan(vals[-1]) else np.nan
                    elif engine_name in ['rate_of_change_ratio', 'rate_of_change']:
                        result = func(y, I)
                        row.update(result)
                    else:
                        result = func(y)
                        row.update(result)
                except Exception as e:
                    pass  # Engine failed silently

            results[(unit_id, signal_id)] = row

        if self.verbose:
            print(f"  Processed {len(results)} signals with Python engines")

        return results

    def _run_sql_engines(self, python_results: Dict) -> pl.DataFrame:
        """Run SQL engines for derived scale-invariant metrics."""
        con = duckdb.connect()

        # Load observations
        con.execute(f"""
            CREATE TABLE observations AS
            SELECT * FROM read_parquet('{self.data_dir / 'observations.parquet'}')
        """)

        # Load typology
        con.execute(f"""
            CREATE TABLE typology AS
            SELECT * FROM read_parquet('{self.data_dir / 'typology.parquet'}')
        """)

        # SQL for scale-invariant derived metrics
        sql = """
        WITH signal_stats AS (
            SELECT
                o.unit_id,
                o.signal_id,

                -- Scale-invariant metrics
                KURTOSIS(o.value) AS kurtosis_sql,
                SKEWNESS(o.value) AS skewness_sql,

                -- Crest factor (max / rms)
                MAX(ABS(o.value)) / NULLIF(SQRT(AVG(o.value * o.value)), 0) AS crest_factor_sql,

                -- Peak ratio (max / mean_abs)
                MAX(ABS(o.value)) / NULLIF(AVG(ABS(o.value)), 0) AS peak_ratio_sql,

                -- Coefficient of variation (std / mean)
                STDDEV(o.value) / NULLIF(ABS(AVG(o.value)), 0) AS cv_sql,

                -- Range ratio (range / mean)
                (MAX(o.value) - MIN(o.value)) / NULLIF(ABS(AVG(o.value)), 0) AS range_ratio_sql,

                -- Join typology
                t.signal_type,
                t.periodicity_type,
                t.tail_type,
                t.stationarity_type,
                t.smoothness,
                t.memory_proxy

            FROM observations o
            LEFT JOIN typology t
                ON o.unit_id = t.unit_id AND o.signal_id = t.signal_id
            WHERE t.is_constant = FALSE OR t.is_constant IS NULL
            GROUP BY o.unit_id, o.signal_id,
                     t.signal_type, t.periodicity_type, t.tail_type,
                     t.stationarity_type, t.smoothness, t.memory_proxy
        )
        SELECT * FROM signal_stats
        """

        result = con.execute(sql).pl()
        con.close()

        if self.verbose:
            print(f"  Computed {len(result)} signal metrics with SQL")

        return result

    def _combine_results(
        self,
        python_results: Dict,
        sql_results: pl.DataFrame
    ) -> pl.DataFrame:
        """Combine Python and SQL results into signal_vector."""

        # Convert Python results to DataFrame
        if python_results:
            python_df = pl.DataFrame(list(python_results.values()))
        else:
            python_df = pl.DataFrame({'unit_id': [], 'signal_id': []})

        # Remove deprecated (scale-dependent) columns from Python results
        deprecated = set(ENGINE_MANIFEST.get('deprecated', []))
        deprecated_patterns = [
            'rms', 'peak', 'total_power', 'harmonic_2x', 'harmonic_3x',
            'band_low', 'band_mid', 'band_high', 'fundamental_amplitude',
            'envelope', 'mean', 'std', 'range'
        ]

        # Filter columns - keep only scale-invariant
        if len(python_df) > 0:
            keep_cols = ['unit_id', 'signal_id', 'n_samples']
            for col in python_df.columns:
                if col in keep_cols:
                    continue
                # Keep if scale-invariant (ratios, entropy, hurst, kurtosis, etc.)
                is_scale_invariant = any([
                    'ratio' in col.lower(),
                    'rel' in col.lower(),
                    'entropy' in col.lower(),
                    'kurtosis' in col.lower(),
                    'skewness' in col.lower(),
                    'hurst' in col.lower(),
                    'crest' in col.lower(),
                    'thd' in col.lower(),
                    'r2' in col.lower(),
                    'slope' in col.lower(),
                    'centroid' in col.lower(),
                    'bandwidth' in col.lower(),
                ])
                is_deprecated = any(p in col.lower() for p in deprecated_patterns)

                if is_scale_invariant and not is_deprecated:
                    keep_cols.append(col)

            python_df = python_df.select([c for c in keep_cols if c in python_df.columns])

        # Join with SQL results
        if len(python_df) > 0 and len(sql_results) > 0:
            # Merge on unit_id, signal_id
            signal_vector = python_df.join(
                sql_results,
                on=['unit_id', 'signal_id'],
                how='outer'
            )
        elif len(python_df) > 0:
            signal_vector = python_df
        else:
            signal_vector = sql_results

        # Remove any _right columns from join
        signal_vector = signal_vector.select([
            c for c in signal_vector.columns if not c.endswith('_right')
        ])

        # Sort by unit_id, signal_id
        signal_vector = signal_vector.sort(['unit_id', 'signal_id'])

        return signal_vector


def run_signal_vector(
    data_dir: str,
    window_size: int = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Run signal vector computation.

    Args:
        data_dir: Directory containing observations.parquet and typology.parquet
        window_size: Override window size (default: from typology)
        verbose: Print progress

    Returns:
        DataFrame with signal vectors
    """
    runner = SignalVectorRunner(
        data_dir=Path(data_dir),
        window_size=window_size,
        verbose=verbose
    )
    return runner.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python signal_vector_runner.py <data_dir>")
        print()
        print("Creates signal_vector.parquet from typology-guided engine selection.")
        print("Requires: observations.parquet, typology.parquet in data_dir")
        sys.exit(1)

    data_dir = sys.argv[1]
    run_signal_vector(data_dir)

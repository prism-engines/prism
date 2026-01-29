"""
PRISM Dynamics Engine

Main orchestration for computing dynamical systems metrics.
Complements the geometric eigenvalue analysis with:
- Lyapunov exponents (stability/chaos)
- Attractor dimension (complexity)
- Recurrence quantification (predictability)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path

from .reconstruction import optimal_delay, optimal_embedding_dim, embed_time_series
from .lyapunov import largest_lyapunov_exponent
from .dimension import correlation_dimension
from .recurrence import recurrence_matrix, rqa_metrics


class DynamicsEngine:
    """
    Compute dynamical systems metrics for time series data.

    Parameters
    ----------
    window_size : int
        Size of sliding window for analysis
    step_size : int
        Step between windows
    embedding_dim : int, optional
        Fixed embedding dimension (auto-detected if None)
    time_delay : int, optional
        Fixed time delay (auto-detected if None)
    min_data_points : int
        Minimum points required for analysis

    Examples
    --------
    >>> engine = DynamicsEngine(window_size=200, step_size=20)
    >>> signals = {'sensor1': np.random.randn(1000)}
    >>> result = engine.compute_for_entity(signals, 'entity_1')
    """

    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 10,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
        min_data_points: int = 50
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.min_data_points = min_data_points

    def compute_for_signal(
        self,
        x: np.ndarray,
        entity_id: str,
        signal_id: str = None
    ) -> pd.DataFrame:
        """
        Compute dynamics metrics for a single signal using sliding windows.

        Parameters
        ----------
        x : array
            Time series
        entity_id : str
            Entity identifier
        signal_id : str, optional
            Signal identifier

        Returns
        -------
        DataFrame with dynamics metrics per window
        """
        x = np.asarray(x).flatten()
        n = len(x)

        if n < self.window_size:
            return pd.DataFrame()

        results = []

        for i in range(0, n - self.window_size + 1, self.step_size):
            window = x[i : i + self.window_size]

            if len(window) < self.min_data_points:
                continue

            # Get embedding parameters
            try:
                tau = self.time_delay or optimal_delay(window, max_tau=min(30, len(window) // 5))
                dim = self.embedding_dim or optimal_embedding_dim(window, tau, max_dim=min(6, len(window) // (3 * tau)))
            except:
                tau = 5
                dim = 3

            # Lyapunov exponent
            try:
                lyap = largest_lyapunov_exponent(window, tau, dim)
            except Exception:
                lyap = np.nan

            # Correlation dimension
            try:
                corr_dim = correlation_dimension(window, tau, dim)
            except Exception:
                corr_dim = np.nan

            # RQA metrics
            try:
                R = recurrence_matrix(window, tau, dim)
                rqa = rqa_metrics(R)
            except Exception:
                rqa = {
                    'recurrence_rate': np.nan,
                    'determinism': np.nan,
                    'avg_diagonal_length': np.nan,
                    'max_diagonal_length': 0,
                    'entropy': np.nan,
                    'laminarity': np.nan,
                    'trapping_time': np.nan,
                    'max_vertical_length': 0,
                }

            result = {
                'entity_id': entity_id,
                'I': i + self.window_size // 2,  # Center of window
                'window_start': i,
                'window_end': i + self.window_size,
                'lyapunov_max': lyap,
                'correlation_dim': corr_dim,
                'embedding_dim': dim,
                'time_delay': tau,
                **rqa
            }

            if signal_id is not None:
                result['signal_id'] = signal_id

            results.append(result)

        return pd.DataFrame(results)

    def compute_for_entity(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str
    ) -> pd.DataFrame:
        """
        Compute dynamics metrics for all signals of an entity.

        Parameters
        ----------
        signals : dict
            {signal_name: time_series_array}
        entity_id : str
            Entity identifier

        Returns
        -------
        DataFrame with dynamics metrics per window, aggregated across signals
        """
        all_results = []

        for signal_name, x in signals.items():
            if len(x) < self.min_data_points:
                continue

            df = self.compute_for_signal(x, entity_id, signal_name)
            if not df.empty:
                all_results.append(df)

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)

        # Aggregate across signals per window (by I)
        numeric_cols = [
            'lyapunov_max', 'correlation_dim', 'recurrence_rate',
            'determinism', 'avg_diagonal_length', 'entropy',
            'laminarity', 'trapping_time'
        ]

        agg_dict = {
            'lyapunov_max': ['mean', 'max', 'std'],
            'correlation_dim': 'mean',
            'recurrence_rate': 'mean',
            'determinism': 'mean',
            'avg_diagonal_length': 'mean',
            'entropy': 'mean',
            'laminarity': 'mean',
            'trapping_time': 'mean',
            'embedding_dim': 'mean',
            'time_delay': 'mean',
        }

        aggregated = combined.groupby(['entity_id', 'I']).agg(agg_dict)
        aggregated.columns = [
            'lyapunov_max', 'lyapunov_max_signal', 'lyapunov_spread',
            'correlation_dim', 'recurrence_rate', 'determinism',
            'avg_diagonal_length', 'rqa_entropy', 'laminarity',
            'trapping_time', 'embedding_dim', 'time_delay'
        ]
        aggregated = aggregated.reset_index()

        return aggregated.sort_values(['entity_id', 'I'])

    def to_parquet(self, df: pd.DataFrame, path: Path):
        """Save dynamics results to parquet."""
        df.to_parquet(path, index=False)


def compute_dynamics_for_entity(
    obs_enriched: pd.DataFrame,
    entity_id: str,
    window_size: int = 100,
    step_size: int = 10
) -> pd.DataFrame:
    """
    Compute dynamics metrics for a single entity from enriched observations.

    Parameters
    ----------
    obs_enriched : DataFrame
        Enriched observations with columns: entity_id, signal_id, I, y
    entity_id : str
        Entity to process
    window_size : int
        Analysis window size
    step_size : int
        Step between windows

    Returns
    -------
    DataFrame with dynamics metrics
    """
    engine = DynamicsEngine(window_size=window_size, step_size=step_size)

    entity_data = obs_enriched[obs_enriched['entity_id'] == entity_id]

    if entity_data.empty:
        return pd.DataFrame()

    # Extract signals
    signals = {}
    for signal_id in entity_data['signal_id'].unique():
        signal_data = entity_data[entity_data['signal_id'] == signal_id]
        signal_data = signal_data.sort_values('I')
        signals[signal_id] = signal_data['y'].values

    return engine.compute_for_entity(signals, entity_id)


def compute_dynamics(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 10,
    progress: bool = True
) -> pd.DataFrame:
    """
    Compute dynamics metrics for all entities.

    Parameters
    ----------
    obs_enriched : DataFrame
        Enriched observations (from observations_enriched.parquet)
    window_size : int
        Analysis window size
    step_size : int
        Step between windows
    progress : bool
        Print progress messages

    Returns
    -------
    DataFrame with dynamics metrics for all entities
    """
    entities = obs_enriched['entity_id'].unique()

    if progress:
        print(f"Computing dynamics for {len(entities)} entities...")

    all_results = []

    for i, entity_id in enumerate(entities):
        if progress and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(entities)} entities")

        try:
            result = compute_dynamics_for_entity(
                obs_enriched, entity_id, window_size, step_size
            )
            if not result.empty:
                all_results.append(result)
        except Exception as e:
            if progress:
                print(f"  Warning: entity {entity_id} failed: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    if progress:
        print(f"  dynamics: {len(combined):,} rows × {len(combined.columns)} cols")

    return combined


def compute_dynamics_for_all_entities(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 10
) -> pd.DataFrame:
    """
    Compute dynamics for all entities (alias for compute_dynamics).

    This function name matches the pattern used in physics.py.
    """
    return compute_dynamics(obs_enriched, window_size, step_size, progress=True)

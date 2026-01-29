"""
PRISM Topology Engine

Main orchestration for computing topological data analysis metrics.
Captures the SHAPE of system dynamics through persistent homology.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass

from .point_cloud import multivariate_point_cloud, subsample_point_cloud
from .persistence import compute_rips_persistence, PersistenceDiagram
from .features import (
    betti_numbers,
    persistence_statistics,
    persistence_landscape,
    topological_complexity
)


@dataclass
class TopologyResult:
    """Container for topology analysis results."""
    entity_id: str
    observation_idx: int

    # Betti numbers at characteristic scale
    betti_0: int  # Connected components
    betti_1: int  # Loops
    betti_2: int  # Voids

    # H0 statistics (components)
    h0_n_features: int
    h0_total_persistence: float
    h0_max_persistence: float
    h0_mean_persistence: float

    # H1 statistics (loops)
    h1_n_features: int
    h1_total_persistence: float
    h1_max_persistence: float
    h1_mean_persistence: float
    h1_persistence_entropy: float

    # H2 statistics (voids)
    h2_n_features: int
    h2_total_persistence: float

    # Overall complexity
    topological_complexity: float

    # Landscape features (for ML)
    landscape_h1_integral: float
    landscape_h1_max: float


class TopologyEngine:
    """
    Compute topological data analysis metrics for time series.

    Complements geometric (eigenvalue) and dynamic (Lyapunov) analysis
    with shape-based features from persistent homology.
    """

    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 20,
        max_homology_dim: int = 2,
        n_landmarks: int = 200,
        min_data_points: int = 20
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.max_homology_dim = max_homology_dim
        self.n_landmarks = n_landmarks
        self.min_data_points = min_data_points

    def compute_for_window(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str,
        window_start: int
    ) -> TopologyResult:
        """
        Compute topology metrics for a single window.
        """
        # Extract window
        window_signals = {
            name: sig[window_start:window_start + self.window_size]
            for name, sig in signals.items()
        }

        # Build point cloud
        point_cloud = multivariate_point_cloud(window_signals, method='direct')

        # Subsample if needed
        if len(point_cloud) > self.n_landmarks:
            point_cloud = subsample_point_cloud(point_cloud, self.n_landmarks, method='random')

        # Normalize
        mean = point_cloud.mean(axis=0)
        std = point_cloud.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        point_cloud = (point_cloud - mean) / std

        # Compute persistent homology
        diagrams = compute_rips_persistence(
            point_cloud,
            max_dimension=self.max_homology_dim,
            n_landmarks=None  # Already subsampled
        )

        # Betti numbers at characteristic scale
        all_deaths = []
        for d in diagrams:
            deaths = d.death_times[np.isfinite(d.death_times)]
            all_deaths.extend(deaths)

        char_scale = np.median(all_deaths) if all_deaths else 1.0
        betti = betti_numbers(diagrams, char_scale)

        # Statistics per dimension
        h0_stats = persistence_statistics(diagrams[0]) if len(diagrams) > 0 else {}
        h1_stats = persistence_statistics(diagrams[1]) if len(diagrams) > 1 else {}
        h2_stats = persistence_statistics(diagrams[2]) if len(diagrams) > 2 else {}

        # Landscape features for H1
        if len(diagrams) > 1 and diagrams[1].n_features > 0:
            landscapes = persistence_landscape(diagrams[1], n_landscapes=3, n_points=50)
            landscape_integral = float(np.sum(landscapes[0]))
            landscape_max = float(np.max(landscapes[0]))
        else:
            landscape_integral = 0.0
            landscape_max = 0.0

        # Overall complexity
        complexity = topological_complexity(diagrams)

        return TopologyResult(
            entity_id=entity_id,
            observation_idx=window_start + self.window_size // 2,

            betti_0=betti.get(0, 0),
            betti_1=betti.get(1, 0),
            betti_2=betti.get(2, 0),

            h0_n_features=h0_stats.get('n_features', 0),
            h0_total_persistence=h0_stats.get('total_persistence', 0.0),
            h0_max_persistence=h0_stats.get('max_persistence', 0.0),
            h0_mean_persistence=h0_stats.get('mean_persistence', 0.0),

            h1_n_features=h1_stats.get('n_features', 0),
            h1_total_persistence=h1_stats.get('total_persistence', 0.0),
            h1_max_persistence=h1_stats.get('max_persistence', 0.0),
            h1_mean_persistence=h1_stats.get('mean_persistence', 0.0),
            h1_persistence_entropy=h1_stats.get('persistence_entropy', 0.0),

            h2_n_features=h2_stats.get('n_features', 0),
            h2_total_persistence=h2_stats.get('total_persistence', 0.0),

            topological_complexity=complexity,
            landscape_h1_integral=landscape_integral,
            landscape_h1_max=landscape_max,
        )

    def compute_for_entity(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str
    ) -> pd.DataFrame:
        """
        Compute topology metrics across all windows for an entity.
        """
        n_samples = min(len(s) for s in signals.values())
        results = []

        for start in range(0, n_samples - self.window_size, self.step_size):
            try:
                result = self.compute_for_window(signals, entity_id, start)
                results.append(result.__dict__)
            except Exception as e:
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


def compute_topology_for_entity(
    obs_enriched: pd.DataFrame,
    entity_id: str,
    window_size: int = 100,
    step_size: int = 20
) -> pd.DataFrame:
    """
    Compute topology metrics for a single entity from enriched observations.
    """
    engine = TopologyEngine(window_size=window_size, step_size=step_size)

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


def compute_topology(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 20,
    progress: bool = True
) -> pd.DataFrame:
    """
    Compute topology metrics for all entities.
    """
    entities = obs_enriched['entity_id'].unique()

    if progress:
        print(f"Computing topology for {len(entities)} entities...")

    all_results = []

    for i, entity_id in enumerate(entities):
        if progress and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(entities)} entities")

        try:
            result = compute_topology_for_entity(
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
        print(f"  topology: {len(combined):,} rows x {len(combined.columns)} cols")

    return combined


def compute_topology_for_all_entities(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 20
) -> pd.DataFrame:
    """
    Compute topology for all entities (alias for compute_topology).
    """
    return compute_topology(obs_enriched, window_size, step_size, progress=True)

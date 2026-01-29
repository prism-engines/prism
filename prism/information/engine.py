"""
PRISM Information Flow Engine

Main orchestration for computing causal and information-theoretic metrics.
Captures WHO DRIVES WHOM through directional information flow.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from .transfer_entropy import transfer_entropy_matrix
from .granger import granger_causality_matrix
from .network import CausalNetwork, network_metrics, build_network_from_te
from .entropy import mutual_information


@dataclass
class InformationResult:
    """Container for information flow analysis results."""
    entity_id: str
    observation_idx: int

    # Network summary
    n_causal_edges: int
    network_density: float
    network_reciprocity: float
    hierarchy_score: float
    n_feedback_loops: int

    # Dominant flows
    max_transfer_entropy: float
    mean_transfer_entropy: float
    top_driver: str
    top_sink: str

    # Information content
    total_mutual_information: float
    mean_pairwise_mi: float

    # Change detection
    network_changed: bool


class InformationEngine:
    """
    Compute information-theoretic causality metrics for multivariate time series.

    Captures directional information flow between signals.
    """

    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 20,
        te_lag: int = 1,
        te_history: int = 1,
        te_bins: int = 8,
        granger_max_lag: int = 5,
        significance_threshold: float = 0.05,
        te_threshold: float = 0.1
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.te_lag = te_lag
        self.te_history = te_history
        self.te_bins = te_bins
        self.granger_max_lag = granger_max_lag
        self.significance_threshold = significance_threshold
        self.te_threshold = te_threshold

        self.previous_network: Optional[CausalNetwork] = None

    def compute_for_window(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str,
        window_start: int
    ) -> Tuple[InformationResult, CausalNetwork]:
        """
        Compute information flow metrics for a single window.
        """
        # Extract window
        window_signals = {
            name: sig[window_start:window_start + self.window_size]
            for name, sig in signals.items()
        }

        signal_names = list(window_signals.keys())
        n_signals = len(signal_names)

        if n_signals < 2:
            raise ValueError("Need at least 2 signals")

        # Transfer entropy matrix
        te_matrix, _ = transfer_entropy_matrix(
            window_signals,
            lag=self.te_lag,
            history_length=self.te_history,
            bins=self.te_bins
        )

        # Build causal network
        network = build_network_from_te(te_matrix, signal_names)

        # Network metrics
        metrics = network_metrics(network, threshold=self.te_threshold)

        # Find top driver and sink
        out_degree = np.sum(te_matrix > self.te_threshold, axis=1)
        in_degree = np.sum(te_matrix > self.te_threshold, axis=0)

        top_driver_idx = int(np.argmax(out_degree))
        top_sink_idx = int(np.argmax(in_degree))

        # Mutual information matrix (symmetric)
        mi_total = 0.0
        mi_count = 0
        for i, name_i in enumerate(signal_names):
            for j, name_j in enumerate(signal_names):
                if i < j:
                    mi = mutual_information(
                        window_signals[name_i],
                        window_signals[name_j]
                    )
                    mi_total += mi
                    mi_count += 1

        mean_mi = mi_total / mi_count if mi_count > 0 else 0.0

        # Detect network change
        network_changed = False
        if self.previous_network is not None:
            prev_edges = self.previous_network.adjacency_matrix > self.te_threshold
            curr_edges = te_matrix > self.te_threshold
            if prev_edges.shape == curr_edges.shape:
                change_ratio = np.sum(prev_edges != curr_edges) / (n_signals * n_signals)
                network_changed = change_ratio > 0.2

        self.previous_network = network

        # Transfer entropy statistics
        te_nonzero = te_matrix[te_matrix > 0]
        max_te = float(np.max(te_matrix)) if te_matrix.size > 0 else 0.0
        mean_te = float(np.mean(te_nonzero)) if len(te_nonzero) > 0 else 0.0

        result = InformationResult(
            entity_id=entity_id,
            observation_idx=window_start + self.window_size // 2,

            n_causal_edges=metrics['n_edges'],
            network_density=metrics['density'],
            network_reciprocity=metrics['reciprocity'],
            hierarchy_score=metrics['hierarchy_score'],
            n_feedback_loops=metrics['n_feedback_loops'],

            max_transfer_entropy=max_te,
            mean_transfer_entropy=mean_te,
            top_driver=signal_names[top_driver_idx],
            top_sink=signal_names[top_sink_idx],

            total_mutual_information=mi_total,
            mean_pairwise_mi=mean_mi,

            network_changed=network_changed,
        )

        return result, network

    def compute_for_entity(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str
    ) -> pd.DataFrame:
        """
        Compute information flow metrics across all windows for an entity.
        """
        if len(signals) < 2:
            return pd.DataFrame()

        n_samples = min(len(s) for s in signals.values())
        results = []

        self.previous_network = None  # Reset for new entity

        for start in range(0, n_samples - self.window_size, self.step_size):
            try:
                result, _ = self.compute_for_window(signals, entity_id, start)
                results.append({
                    'entity_id': result.entity_id,
                    'I': result.observation_idx,
                    'n_causal_edges': result.n_causal_edges,
                    'network_density': result.network_density,
                    'network_reciprocity': result.network_reciprocity,
                    'hierarchy_score': result.hierarchy_score,
                    'n_feedback_loops': result.n_feedback_loops,
                    'max_transfer_entropy': result.max_transfer_entropy,
                    'mean_transfer_entropy': result.mean_transfer_entropy,
                    'top_driver': result.top_driver,
                    'top_sink': result.top_sink,
                    'total_mutual_information': result.total_mutual_information,
                    'mean_pairwise_mi': result.mean_pairwise_mi,
                    'network_changed': result.network_changed,
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


def compute_information_flow_for_entity(
    obs_enriched: pd.DataFrame,
    entity_id: str,
    window_size: int = 100,
    step_size: int = 20
) -> pd.DataFrame:
    """
    Compute information flow metrics for a single entity.
    """
    engine = InformationEngine(window_size=window_size, step_size=step_size)

    entity_data = obs_enriched[obs_enriched['entity_id'] == entity_id]

    if entity_data.empty:
        return pd.DataFrame()

    # Extract signals
    signals = {}
    for signal_id in entity_data['signal_id'].unique():
        signal_data = entity_data[entity_data['signal_id'] == signal_id]
        signal_data = signal_data.sort_values('I')
        signals[str(signal_id)] = signal_data['y'].values

    return engine.compute_for_entity(signals, entity_id)


def compute_information_flow(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 20,
    progress: bool = True
) -> pd.DataFrame:
    """
    Compute information flow metrics for all entities.
    """
    entities = obs_enriched['entity_id'].unique()

    if progress:
        print(f"Computing information flow for {len(entities)} entities...")

    all_results = []

    for i, entity_id in enumerate(entities):
        if progress and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(entities)} entities")

        try:
            result = compute_information_flow_for_entity(
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
        print(f"  information_flow: {len(combined):,} rows x {len(combined.columns)} cols")

    return combined


def compute_information_flow_for_all_entities(
    obs_enriched: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 20
) -> pd.DataFrame:
    """
    Compute information flow for all entities (alias).
    """
    return compute_information_flow(obs_enriched, window_size, step_size, progress=True)

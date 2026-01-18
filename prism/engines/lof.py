"""
PRISM Local Outlier Factor Engine

Detects anomalous signals in behavioral space using density-based analysis.

Measures:
- LOF score per signal (>1 = outlier, <1 = inlier)
- Number of outliers at various thresholds
- Most anomalous signals
- Average local reachability density

Phase: Structure
Normalization: Z-score required

Interpretation:
- LOF > 1.5: Moderately unusual behavioral signature
- LOF > 2.0: Strongly unusual behavioral signature  
- LOF > 3.0: Extreme outlier in behavioral space

Use Cases:
- Identify signals with unusual behavioral patterns
- Early warning: outliers may be leading signals
- Data quality: extreme outliers may have data issues
"""

import logging
from typing import Dict, Any, List
from datetime import date

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="lof",
    engine_type="geometry",
    description="Local outlier factor for anomaly detection in behavioral space",
    domains={"structure", "anomaly"},
    requires_window=True,
    deterministic=True,
)


class LOFEngine(BaseEngine):
    """
    Local Outlier Factor engine for behavioral space.
    
    Identifies signals whose behavioral signatures are unusual
    compared to their local neighborhood in behavioral space.
    
    LOF compares the local density around each point to the local
    density of its neighbors. Points with substantially lower density
    than their neighbors are considered outliers.
    
    Outputs:
        - results.lof_scores: Per-signal LOF scores
    """
    
    name = "lof"
    phase = "structure"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_neighbors: int = 5,
        contamination: str = "auto",
        **params
    ) -> Dict[str, Any]:
        """
        Run LOF analysis on behavioral space.
        
        Args:
            df: Behavioral vectors (rows=dimensions, cols=signals)
            run_id: Unique run identifier
            n_neighbors: Number of neighbors for LOF (default 5)
            contamination: Expected proportion of outliers ("auto" or float)
        
        Returns:
            Dict with summary metrics
        """
        signals = list(df.columns)
        n_signals = len(signals)
        
        if n_signals < n_neighbors + 1:
            n_neighbors = max(2, n_signals - 1)
            logger.warning(f"Reduced n_neighbors to {n_neighbors} due to small sample")
        
        window_start, window_end = get_window_dates(df)
        
        # Prepare data: LOF expects (n_samples, n_features)
        # df.T gives us (n_signals, n_dimensions)
        X = df.T.values
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination if contamination != "auto" else "auto",
            novelty=False,  # We're doing outlier detection, not novelty detection
        )
        
        # fit_predict returns -1 for outliers, 1 for inliers
        labels = lof.fit_predict(X)
        
        # Negative LOF scores (sklearn convention: more negative = more outlier)
        # We negate to get positive scores where higher = more outlier
        lof_scores = -lof.negative_outlier_factor_
        
        # Create score DataFrame
        score_df = pd.DataFrame({
            "signal_id": signals,
            "lof_score": lof_scores,
            "is_outlier": labels == -1,
        })
        
        # Sort by LOF score (most anomalous first)
        score_df = score_df.sort_values("lof_score", ascending=False)
        
        # Store scores
        self._store_scores(
            score_df, window_start, window_end, run_id
        )
        
        # Compute metrics
        n_outliers = (labels == -1).sum()
        
        # Outliers at various thresholds
        outliers_1_5 = (lof_scores > 1.5).sum()
        outliers_2_0 = (lof_scores > 2.0).sum()
        outliers_3_0 = (lof_scores > 3.0).sum()
        
        # Top outliers
        top_outliers = score_df.head(5)["signal_id"].tolist()
        
        # Most normal (lowest LOF)
        most_normal = score_df.tail(5)["signal_id"].tolist()
        
        metrics = {
            "n_signals": n_signals,
            "n_neighbors": n_neighbors,
            "n_outliers_auto": int(n_outliers),
            "outlier_rate_auto": float(n_outliers / n_signals),
            "n_outliers_1_5": int(outliers_1_5),
            "n_outliers_2_0": int(outliers_2_0),
            "n_outliers_3_0": int(outliers_3_0),
            "avg_lof_score": float(np.mean(lof_scores)),
            "max_lof_score": float(np.max(lof_scores)),
            "min_lof_score": float(np.min(lof_scores)),
            "std_lof_score": float(np.std(lof_scores)),
            "median_lof_score": float(np.median(lof_scores)),
        }
        
        logger.info(
            f"LOF complete: {n_signals} signals, "
            f"{n_outliers} outliers (auto), "
            f"max LOF={metrics['max_lof_score']:.2f}, "
            f"top outliers: {top_outliers[:3]}"
        )
        
        return metrics
    
    def _store_scores(
        self,
        score_df: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store LOF scores per signal."""
        records = []
        for _, row in score_df.iterrows():
            records.append({
                "signal_id": row["signal_id"],
                "window_start": window_start,
                "window_end": window_end,
                "lof_score": float(row["lof_score"]),
                "is_outlier": bool(row["is_outlier"]),
                "outlier_severity": self._classify_severity(row["lof_score"]),
                "run_id": run_id,
            })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("lof_scores", df, run_id)
    
    def _classify_severity(self, lof_score: float) -> str:
        """Classify outlier severity based on LOF score."""
        if lof_score > 3.0:
            return "extreme"
        elif lof_score > 2.0:
            return "strong"
        elif lof_score > 1.5:
            return "moderate"
        elif lof_score > 1.0:
            return "mild"
        else:
            return "normal"

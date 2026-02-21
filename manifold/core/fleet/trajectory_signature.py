"""
Trajectory Signature Library — Core Engine
==========================================

Captures the shape of degradation as a multivariate geometric fingerprint,
clusters cohorts by trajectory similarity via DTW, and matches each cohort
against the resulting library.

Four functions:
    extract_signatures  — join upstream outputs into signature vectors
    compute_signature_distances — pairwise component-wise DTW distance matrix
    cluster_signatures  — hierarchical clustering with silhouette selection
    match_cohorts_to_library — per-cohort nearest-cluster match

Pure computation — DataFrames in, DataFrames out. No file I/O.
"""

import numpy as np
import polars as pl
from typing import Tuple, List, Optional

from manifold.core._pmtvs import dynamic_time_warping
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ════════════════════════════════════════════════════════════════
# SIGNATURE COLUMNS — the multivariate fingerprint
# ════════════════════════════════════════════════════════════════

SIGNATURE_COLS = [
    'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
    'effective_dim', 'total_variance', 'condition_number',
    'effective_dim_velocity', 'effective_dim_acceleration', 'effective_dim_curvature',
    'speed', 'curvature', 'acceleration_magnitude',
    'torsion',
    'arc_length',
]

# Shape-only columns for DTW distance computation.
# Excludes position features (eigenvalue_1-3, effective_dim, total_variance)
# which cause DTW to cluster by magnitude rather than trajectory shape.
# Excludes arc_length (cumulative, always larger for longer trajectories).
# Excludes torsion (NaN placeholder).
DTW_COLS_BASE = [
    'effective_dim_velocity', 'effective_dim_acceleration', 'effective_dim_curvature',
    'condition_number',
    'speed', 'curvature', 'acceleration_magnitude',
]

# When use_derivatives=True, also include first-differences of base features.
# This adds rate-of-change information to DTW: not just "how fast is it bending"
# but "is the bending accelerating or decelerating."
# Derivative column names: append '_d1' to each base column.
DTW_COLS_DERIVATIVES = DTW_COLS_BASE  # base cols are always included; derivatives appended inline


# ════════════════════════════════════════════════════════════════
# 1. EXTRACT SIGNATURES
# ════════════════════════════════════════════════════════════════

def extract_signatures(
    cohort_geometry: pl.DataFrame,
    geometry_dynamics: pl.DataFrame,
    velocity_field: pl.DataFrame,
    engine: str = 'shape',
) -> pl.DataFrame:
    """
    Join three upstream outputs into a signature DataFrame.

    Filters cohort_geometry and geometry_dynamics to a single engine
    (default 'shape'), then joins all three on (cohort, signal_0_end).
    Adds arc_length (cumulative Euclidean distance) and torsion (NaN placeholder).

    Args:
        cohort_geometry: From stage 03 (grain: cohort, engine, signal_0_end)
        geometry_dynamics: From stage 07 (grain: cohort, engine, signal_0_end)
        velocity_field: From stage 21 (grain: cohort, signal_0_end)
        engine: Feature group to use (default 'shape')

    Returns:
        Signature DataFrame with grain (cohort, signal_0_end)
    """
    if len(cohort_geometry) == 0:
        return pl.DataFrame()

    # Ensure cohort column exists
    for df_name in ['cohort_geometry', 'geometry_dynamics', 'velocity_field']:
        df = locals()[df_name]
        if 'cohort' not in df.columns and len(df) > 0:
            locals()[df_name] = df.with_columns(pl.lit('').alias('cohort'))

    # Re-bind after potential mutation
    cg = cohort_geometry if 'cohort' in cohort_geometry.columns else cohort_geometry.with_columns(pl.lit('').alias('cohort'))
    gd = geometry_dynamics if 'cohort' in geometry_dynamics.columns else geometry_dynamics.with_columns(pl.lit('').alias('cohort'))
    vf = velocity_field if 'cohort' in velocity_field.columns else velocity_field.with_columns(pl.lit('').alias('cohort'))

    # Filter to chosen engine; fall back to engine with most rows
    if 'engine' in cg.columns:
        engines = cg['engine'].unique().to_list()
        if engine in engines:
            cg = cg.filter(pl.col('engine') == engine)
        elif engines:
            fallback = cg.group_by('engine').len().sort('len', descending=True)['engine'][0]
            cg = cg.filter(pl.col('engine') == fallback)

    if 'engine' in gd.columns:
        engines = gd['engine'].unique().to_list()
        if engine in engines:
            gd = gd.filter(pl.col('engine') == engine)
        elif engines:
            fallback = gd.group_by('engine').len().sort('len', descending=True)['engine'][0]
            gd = gd.filter(pl.col('engine') == fallback)

    # Select columns for join
    cg_cols = ['cohort', 'signal_0_end']
    for c in ['eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 'effective_dim',
              'total_variance', 'condition_number']:
        if c in cg.columns:
            cg_cols.append(c)
    cg = cg.select([c for c in cg_cols if c in cg.columns])

    gd_cols = ['cohort', 'signal_0_end']
    for c in ['effective_dim_velocity', 'effective_dim_acceleration', 'effective_dim_curvature']:
        if c in gd.columns:
            gd_cols.append(c)
    gd = gd.select([c for c in gd_cols if c in gd.columns])

    vf_cols = ['cohort', 'signal_0_end']
    for c in ['speed', 'curvature', 'acceleration_magnitude']:
        if c in vf.columns:
            vf_cols.append(c)
    vf = vf.select([c for c in vf_cols if c in vf.columns])

    # Join on (cohort, signal_0_end)
    joined = cg.join(gd, on=['cohort', 'signal_0_end'], how='inner')
    joined = joined.join(vf, on=['cohort', 'signal_0_end'], how='inner')

    if len(joined) == 0:
        return pl.DataFrame()

    # Add torsion placeholder (NaN)
    joined = joined.with_columns(pl.lit(float('nan')).alias('torsion'))

    # Compute arc_length per cohort (cumulative Euclidean distance in signature space)
    num_cols = [c for c in joined.columns if c not in ('cohort', 'signal_0_end', 'torsion', 'arc_length')]
    joined = joined.sort('cohort', 'signal_0_end')

    arc_lengths = []
    for cohort_val in sorted(joined['cohort'].unique().to_list()):
        cohort_df = joined.filter(pl.col('cohort') == cohort_val).sort('signal_0_end')
        mat = cohort_df.select(num_cols).to_numpy().astype(float)
        # Replace NaN with 0 for distance computation
        mat_clean = np.nan_to_num(mat, nan=0.0)
        diffs = np.diff(mat_clean, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0.0], np.cumsum(step_dists)))
        arc_lengths.extend(cum_dist.tolist())

    joined = joined.with_columns(pl.Series('arc_length', arc_lengths))

    return joined


# ════════════════════════════════════════════════════════════════
# 2. COMPUTE SIGNATURE DISTANCES
# ════════════════════════════════════════════════════════════════

def _resample_trajectory(mat: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a T×D trajectory matrix to target_len rows via linear interpolation."""
    t_orig = mat.shape[0]
    if t_orig == target_len:
        return mat
    d = mat.shape[1]
    x_orig = np.linspace(0, 1, t_orig)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, d))
    for dim in range(d):
        resampled[:, dim] = np.interp(x_new, x_orig, mat[:, dim])
    return resampled


def compute_signature_distances(
    signatures: pl.DataFrame,
    signature_cols: Optional[List[str]] = None,
    min_windows: int = 5,
    use_derivatives: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise DTW distance matrix between cohort trajectories.

    Groups by cohort to get T×D trajectory matrices. Resamples all
    trajectories to the median length so DTW compares shape, not duration.
    Computes component-wise DTW (mean of per-dimension DTW distances),
    normalized by path length.

    Uses raw upstream values — no Z-scoring. If engines have genuinely
    different curvature magnitudes, that's information (Road A vs Road B).

    Args:
        signatures: Output of extract_signatures
        signature_cols: Columns to use for DTW. Defaults to DTW_COLS_BASE.
        min_windows: Minimum windows required per cohort (default 5)
        use_derivatives: If True, append first-differences of each base
            feature as additional DTW dimensions.  (default True)

    Returns:
        (N×N distance matrix, list of cohort labels)
    """
    if len(signatures) == 0:
        return np.array([]).reshape(0, 0), []

    if signature_cols is None:
        signature_cols = [c for c in DTW_COLS_BASE if c in signatures.columns]

    # Group by cohort, build raw trajectory matrices
    raw_trajectories = {}
    for cohort_val in sorted(signatures['cohort'].unique().to_list()):
        cohort_df = signatures.filter(pl.col('cohort') == cohort_val).sort('signal_0_end')
        if len(cohort_df) < min_windows:
            continue
        mat = cohort_df.select(signature_cols).to_numpy().astype(float)
        # NaN → 0, no Z-scoring — raw upstream values preserved
        mat = np.nan_to_num(mat, nan=0.0)
        raw_trajectories[cohort_val] = mat

    cohort_labels = sorted(raw_trajectories.keys())
    n = len(cohort_labels)

    if n < 2:
        if n == 1:
            return np.zeros((1, 1)), cohort_labels
        return np.array([]).reshape(0, 0), []

    # Resample all trajectories to median length to remove duration bias
    lengths = [raw_trajectories[c].shape[0] for c in cohort_labels]
    target_len = int(np.median(lengths))
    target_len = max(target_len, min_windows)

    cohort_trajectories = {}
    for c in cohort_labels:
        resampled = _resample_trajectory(raw_trajectories[c], target_len)
        if use_derivatives and resampled.shape[0] >= 2:
            # Append first-differences as additional dimensions
            diffs = np.diff(resampled, axis=0)
            # Pad with leading zero row to keep length equal
            diffs = np.vstack([np.zeros((1, diffs.shape[1])), diffs])
            resampled = np.hstack([resampled, diffs])
        cohort_trajectories[c] = resampled

    # Global feature scaling: divide each dimension by its global std across
    # all cohorts. This makes features comparable for DTW averaging without
    # destroying inter-cohort magnitude differences (not per-trajectory Z-scoring).
    # condition_number (std~2500) and curvature (std~0.1) differ 24000x —
    # without this, condition_number alone determines every cluster.
    all_stacked = np.vstack([cohort_trajectories[c] for c in cohort_labels])
    global_std = np.std(all_stacked, axis=0)
    global_std[global_std < 1e-12] = 1.0  # avoid division by zero for constant dims
    for c in cohort_labels:
        cohort_trajectories[c] = cohort_trajectories[c] / global_std

    # Pairwise component-wise DTW, normalized by path length
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat_i = cohort_trajectories[cohort_labels[i]]
            mat_j = cohort_trajectories[cohort_labels[j]]
            d_dims = mat_i.shape[1]

            dtw_sum = 0.0
            for d in range(d_dims):
                dtw_sum += dynamic_time_warping(mat_i[:, d], mat_j[:, d])
            # Normalize: average across dimensions, then divide by sequence length
            avg_dtw = (dtw_sum / d_dims) / target_len

            dist_matrix[i, j] = avg_dtw
            dist_matrix[j, i] = avg_dtw

    return dist_matrix, cohort_labels


# ════════════════════════════════════════════════════════════════
# 3. CLUSTER SIGNATURES
# ════════════════════════════════════════════════════════════════

def _silhouette_score(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score directly from distance matrix."""
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    silhouettes = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]
        same = labels == cluster_i
        same[i] = False  # exclude self

        if same.sum() == 0:
            silhouettes[i] = 0.0
            continue

        # a(i) = mean distance to same-cluster points
        a_i = dist_matrix[i, same].mean()

        # b(i) = min over other clusters of mean distance
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == cluster_i:
                continue
            other = labels == lbl
            if other.sum() > 0:
                b_i = min(b_i, dist_matrix[i, other].mean())

        if b_i == np.inf:
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)

    return float(np.mean(silhouettes))


def cluster_signatures(
    distance_matrix: np.ndarray,
    cohort_labels: List[str],
    max_clusters: int = 20,
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Hierarchical clustering on the DTW distance matrix.

    Uses Ward linkage on the condensed distance matrix.
    Selects optimal k via silhouette score sweep (2..min(n//3, max_clusters)).

    Args:
        distance_matrix: N×N DTW distance matrix
        cohort_labels: Cohort names corresponding to matrix rows
        max_clusters: Maximum clusters to try (default 20)

    Returns:
        (library DataFrame, cluster assignment array of length N)
    """
    n = len(cohort_labels)

    if n < 2:
        return pl.DataFrame(), np.array([0] * n)

    # Ensure symmetry and zero diagonal
    dm = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(dm, 0)

    # Condensed distance vector for scipy
    condensed = squareform(dm, checks=False)

    # Ward linkage
    Z = linkage(condensed, method='ward')

    # Sweep k for best silhouette
    max_k = min(n // 3, max_clusters) if n > 6 else min(n - 1, max_clusters)
    max_k = max(max_k, 2)

    best_k = 2
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels_k = fcluster(Z, t=k, criterion='maxclust')
        if len(np.unique(labels_k)) < 2:
            continue
        score = _silhouette_score(dm, labels_k)
        if score > best_score:
            best_score = score
            best_k = k

    assignments = fcluster(Z, t=best_k, criterion='maxclust')
    # Shift to 0-based
    assignments = assignments - 1

    # Build library DataFrame
    library_rows = []
    unique_clusters = sorted(np.unique(assignments))

    for cluster_id in unique_clusters:
        members_mask = assignments == cluster_id
        member_indices = np.where(members_mask)[0]
        member_names = [cohort_labels[i] for i in member_indices]
        n_members = len(member_indices)

        # Intra-cluster distances
        if n_members > 1:
            intra_dists = []
            for i_idx in range(len(member_indices)):
                for j_idx in range(i_idx + 1, len(member_indices)):
                    intra_dists.append(dm[member_indices[i_idx], member_indices[j_idx]])
            mean_intra = float(np.mean(intra_dists))
        else:
            mean_intra = 0.0

        # Inter-cluster distances (to all non-members)
        non_member_mask = ~members_mask
        if non_member_mask.sum() > 0:
            inter_dists = dm[np.ix_(members_mask, non_member_mask)]
            mean_inter = float(np.mean(inter_dists))
        else:
            mean_inter = 0.0

        # Medoid: member with smallest sum of intra-cluster distances
        if n_members > 1:
            sub_dm = dm[np.ix_(member_indices, member_indices)]
            medoid_local = int(np.argmin(sub_dm.sum(axis=1)))
            medoid_cohort = cohort_labels[member_indices[medoid_local]]
        else:
            medoid_cohort = member_names[0]

        # Cluster silhouette
        cluster_sils = []
        for idx in member_indices:
            a_i = dm[idx, members_mask].sum() / max(n_members - 1, 1)
            b_i = np.inf
            for other_cid in unique_clusters:
                if other_cid == cluster_id:
                    continue
                other_mask = assignments == other_cid
                if other_mask.sum() > 0:
                    b_i = min(b_i, dm[idx, other_mask].mean())
            if b_i == np.inf:
                cluster_sils.append(0.0)
            else:
                cluster_sils.append((b_i - a_i) / max(a_i, b_i, 1e-12))
        sil = float(np.mean(cluster_sils)) if cluster_sils else 0.0

        # Compactness: max intra-cluster distance
        compactness = float(np.max(intra_dists)) if n_members > 1 else 0.0

        library_rows.append({
            'trajectory_id': int(cluster_id),
            'n_members': int(n_members),
            'medoid_cohort': medoid_cohort,
            'member_cohorts': ','.join(sorted(member_names)),
            'mean_intra_distance': mean_intra,
            'mean_inter_distance': mean_inter,
            'silhouette': sil,
            'compactness': compactness,
            'mean_n_windows': 0.0,  # filled by caller if needed
        })

    library = pl.DataFrame(library_rows) if library_rows else pl.DataFrame()

    return library, assignments


# ════════════════════════════════════════════════════════════════
# 4. MATCH COHORTS TO LIBRARY
# ════════════════════════════════════════════════════════════════

def match_cohorts_to_library(
    distance_matrix: np.ndarray,
    cohort_labels: List[str],
    library: pl.DataFrame,
    assignments: np.ndarray,
    signatures: pl.DataFrame,
) -> pl.DataFrame:
    """
    Match each cohort to its nearest cluster and compute match quality.

    Args:
        distance_matrix: N×N DTW distance matrix
        cohort_labels: Cohort names
        library: Library DataFrame from cluster_signatures
        assignments: Cluster assignment per cohort
        signatures: Full signature DataFrame (for n_windows and trajectory_position)

    Returns:
        Match DataFrame with grain (cohort)
    """
    n = len(cohort_labels)
    if n == 0 or len(library) == 0:
        return pl.DataFrame()

    # Pre-compute medoid indices per cluster
    medoid_map = {}
    for row in library.iter_rows(named=True):
        tid = row['trajectory_id']
        mc = row['medoid_cohort']
        if mc in cohort_labels:
            medoid_map[tid] = cohort_labels.index(mc)

    unique_clusters = sorted(np.unique(assignments))

    match_rows = []
    for i in range(n):
        cohort = cohort_labels[i]
        cluster_id = int(assignments[i])

        # Distance to own cluster medoid
        if cluster_id in medoid_map:
            match_dist = float(distance_matrix[i, medoid_map[cluster_id]])
        else:
            match_dist = float('nan')

        # Distance to second-nearest cluster medoid
        second_dist = float('inf')
        for cid in unique_clusters:
            if cid == cluster_id:
                continue
            if cid in medoid_map:
                d = float(distance_matrix[i, medoid_map[cid]])
                if d < second_dist:
                    second_dist = d

        if second_dist == float('inf'):
            second_dist = float('nan')
            confidence = float('nan')
        else:
            confidence = 1.0 - (match_dist / (second_dist + 1e-12))

        # Trajectory position: fraction of arc_length completed
        cohort_sigs = signatures.filter(pl.col('cohort') == cohort).sort('signal_0_end')
        n_windows = len(cohort_sigs)
        if n_windows > 0 and 'arc_length' in cohort_sigs.columns:
            max_arc = cohort_sigs['arc_length'].max()
            if max_arc is not None and max_arc > 0:
                last_arc = cohort_sigs['arc_length'][-1]
                trajectory_position = float(last_arc / max_arc) if last_arc is not None else float('nan')
            else:
                trajectory_position = float('nan')
        else:
            trajectory_position = float('nan')

        match_rows.append({
            'cohort': cohort,
            'trajectory_id': cluster_id,
            'match_distance': match_dist,
            'second_distance': second_dist,
            'match_confidence': confidence,
            'trajectory_position': trajectory_position,
            'n_windows': n_windows,
        })

    return pl.DataFrame(match_rows) if match_rows else pl.DataFrame()

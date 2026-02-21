"""
Stage 32: Trajectory Signature Library
======================================

Captures the shape of degradation as a multivariate geometric fingerprint,
clusters cohorts by trajectory similarity via DTW, and matches each cohort
against the resulting library.

Inputs:
    - cohort_geometry.parquet (stage 03)
    - geometry_dynamics.parquet (stage 07)
    - velocity_field.parquet (stage 21)

Outputs:
    - trajectory_signatures.parquet (system/)
    - trajectory_library.parquet (system/)
    - trajectory_match.parquet (system/)
"""

import polars as pl
from pathlib import Path

from manifold.core.fleet.trajectory_signature import (
    extract_signatures,
    compute_signature_distances,
    cluster_signatures,
    match_cohorts_to_library,
)
from manifold.io.writer import write_output
from manifold.utils import safe_fmt


def run(
    cohort_geometry_path: str,
    geometry_dynamics_path: str,
    velocity_field_path: str,
    data_path: str = ".",
    min_windows: int = 5,
    max_clusters: int = 20,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Build trajectory signature library and match cohorts.

    Args:
        cohort_geometry_path: Path to cohort_geometry.parquet
        geometry_dynamics_path: Path to geometry_dynamics.parquet
        velocity_field_path: Path to velocity_field.parquet
        data_path: Root data directory for output
        min_windows: Minimum windows per cohort for DTW (default 5)
        max_clusters: Maximum clusters to try (default 20)
        verbose: Print progress

    Returns:
        trajectory_match DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 32: TRAJECTORY SIGNATURE LIBRARY")
        print("DTW clustering of cohort degradation trajectories")
        print("=" * 70)

    # Read inputs (guard against missing files)
    cg_path = Path(cohort_geometry_path)
    gd_path = Path(geometry_dynamics_path)
    vf_path = Path(velocity_field_path)

    if not cg_path.exists():
        if verbose:
            print("  Skipped (cohort_geometry.parquet not found)")
        return pl.DataFrame()

    cohort_geometry = pl.read_parquet(str(cg_path))
    geometry_dynamics = pl.read_parquet(str(gd_path)) if gd_path.exists() else pl.DataFrame()
    velocity_field = pl.read_parquet(str(vf_path)) if vf_path.exists() else pl.DataFrame()

    if len(cohort_geometry) == 0:
        if verbose:
            print("  Skipped (empty cohort_geometry)")
        return pl.DataFrame()

    # 1. Extract signatures
    if verbose:
        print("  Extracting signatures...")

    sigs = extract_signatures(cohort_geometry, geometry_dynamics, velocity_field)

    if len(sigs) == 0:
        if verbose:
            print("  No signatures produced (join was empty)")
        return pl.DataFrame()

    write_output(sigs, data_path, 'trajectory_signatures', verbose=verbose)

    n_cohorts = sigs['cohort'].n_unique()
    if verbose:
        print(f"  Signatures: {sigs.shape} ({n_cohorts} cohorts)")

    # 2. Compute pairwise DTW distances
    if verbose:
        print("  Computing pairwise DTW distances...")

    dist_matrix, cohort_labels = compute_signature_distances(
        sigs, min_windows=min_windows,
    )

    if len(cohort_labels) < 2:
        if verbose:
            print(f"  Skipped clustering (n_cohorts with >= {min_windows} windows = {len(cohort_labels)} < 2)")
        write_output(pl.DataFrame(), data_path, 'trajectory_library', verbose=verbose)
        write_output(pl.DataFrame(), data_path, 'trajectory_match', verbose=verbose)
        return pl.DataFrame()

    if verbose:
        print(f"  Distance matrix: {dist_matrix.shape[0]}x{dist_matrix.shape[1]} cohorts")

    # 3. Cluster
    if verbose:
        print("  Clustering trajectories...")

    library, assignments = cluster_signatures(
        dist_matrix, cohort_labels, max_clusters=max_clusters,
    )

    # Fill mean_n_windows from signatures
    if len(library) > 0:
        windows_per_cohort = (
            sigs.group_by('cohort').len().rename({'len': 'n_windows'})
        )
        updated_rows = []
        for row in library.iter_rows(named=True):
            members = row['member_cohorts'].split(',')
            member_windows = windows_per_cohort.filter(
                pl.col('cohort').is_in(members)
            )
            mean_nw = float(member_windows['n_windows'].mean()) if len(member_windows) > 0 else 0.0
            row_copy = dict(row)
            row_copy['mean_n_windows'] = mean_nw
            updated_rows.append(row_copy)
        library = pl.DataFrame(updated_rows)

    write_output(library, data_path, 'trajectory_library', verbose=verbose)

    if verbose and len(library) > 0:
        print(f"  Clusters: {len(library)}")
        for row in library.iter_rows(named=True):
            print(f"    trajectory_{row['trajectory_id']}: "
                  f"{row['n_members']} members, "
                  f"medoid={row['medoid_cohort']}, "
                  f"silhouette={safe_fmt(row['silhouette'], '.3f')}")

    # 4. Match cohorts
    if verbose:
        print("  Matching cohorts to library...")

    match = match_cohorts_to_library(
        dist_matrix, cohort_labels, library, assignments, sigs,
    )

    write_output(match, data_path, 'trajectory_match', verbose=verbose)

    if verbose and len(match) > 0:
        print(f"  Matched {len(match)} cohorts")
        print(f"  Mean confidence: {safe_fmt(match['match_confidence'].mean(), '.3f')}")
        print(f"  Mean match distance: {safe_fmt(match['match_distance'].mean(), '.2f')}")

    return match

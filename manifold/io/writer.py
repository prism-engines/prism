"""
Writer — all parquet writes go through here.

No other module should call df.write_parquet directly.
"""

import polars as pl
from pathlib import Path

from manifold.io.reader import STAGE_DIRS


def _safe_write(df: pl.DataFrame, path: Path, verbose: bool = True) -> bool:
    """
    Guard against writing invalid parquet files.

    Returns True if a file was written, False if skipped.
    """
    if df is None:
        return False

    if len(df.columns) == 0:
        if verbose:
            print(f"  !! Skipped {path} (empty schema — 0 columns)")
        return False

    if df.height == 0:
        # Schema-only parquet: columns defined, 0 rows.
        # DuckDB/Polars can read this; downstream SQL won't crash.
        df.head(0).write_parquet(str(path))
        return True

    df.write_parquet(str(path))
    return True


def write_output(
    df: pl.DataFrame,
    data_path: str,
    name: str,
    verbose: bool = True,
) -> Path:
    """
    Write a stage output to the correct subdirectory.

    Args:
        df: DataFrame to write (None or empty-schema → skip)
        data_path: Root data directory (e.g., domains/rossler)
        name: Output name (e.g., 'signal_vector', 'state_geometry')
        verbose: Print path on write

    Returns:
        Path to written file, or None if skipped
    """
    from manifold.io.reader import output_path

    path = output_path(data_path, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not _safe_write(df, path, verbose=verbose):
        return None

    if verbose:
        print(f"  -> {path} ({len(df)} rows)")

    return path


def write_sidecar(
    df: pl.DataFrame,
    data_path: str,
    parent_name: str,
    sidecar_suffix: str,
    verbose: bool = True,
) -> Path:
    """
    Write a sidecar file alongside its parent.

    Example: write_sidecar(df, path, 'state_geometry', 'loadings')
    -> output/2_system_state/state_geometry_loadings.parquet
    """
    from manifold.io.reader import output_path

    sidecar_name = f"{parent_name}_{sidecar_suffix}"

    parent_path = output_path(data_path, parent_name)
    sidecar_path = parent_path.parent / f"{sidecar_name}.parquet"

    if not _safe_write(df, sidecar_path, verbose=verbose):
        return None

    if verbose:
        print(f"  -> {sidecar_path} ({len(df)} rows, sidecar)")

    return sidecar_path

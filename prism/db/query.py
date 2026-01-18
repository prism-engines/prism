"""
PRISM Query Layer

Polars-based query utilities for Parquet files.

Key Functions:
    describe_table(schema, table) - Get column information for a table
    table_stats(schema, table) - Get basic statistics for a table

Note:
    SQL query functions have been removed. Use Polars DataFrame operations
    directly instead:

    >>> import polars as pl
    >>> from prism.db import read_table
    >>>
    >>> # Read and filter
    >>> observations = read_table('raw', 'observations')
    >>> spy_data = observations.filter(pl.col('signal_id') == 'SENSOR_01')
    >>>
    >>> # Aggregations
    >>> avg_by_signal = observations.group_by('signal_id').agg(
    ...     pl.col('value').mean().alias('avg_value')
    ... )
    >>>
    >>> # Joins
    >>> from prism.db import get_parquet_path
    >>> obs = pl.read_parquet(get_parquet_path('raw', 'observations'))
    >>> members = pl.read_parquet(get_parquet_path('config', 'cohort_members'))
    >>> joined = obs.join(members, on='signal_id')
"""

import polars as pl

from prism.db.parquet_store import get_parquet_path


def describe_table(schema: str, table: str) -> pl.DataFrame:
    """
    Get column information for a parquet table.

    Args:
        schema: Schema name
        table: Table name

    Returns:
        DataFrame with column_name, column_type columns

    Example:
        >>> describe_table('raw', 'observations')
        shape: (3, 2)
        +--------------+-------------+
        | column_name  | column_type |
        | ---          | ---         |
        | str          | str         |
        +==============+=============+
        | signal_id | Utf8        |
        | obs_date     | Date        |
        | value        | Float64     |
        +--------------+-------------+
    """
    path = get_parquet_path(schema, table)
    if not path.exists():
        return pl.DataFrame({"column_name": [], "column_type": []})

    lf = pl.scan_parquet(path)
    schema_dict = lf.schema

    return pl.DataFrame(
        {
            "column_name": list(schema_dict.keys()),
            "column_type": [str(dt) for dt in schema_dict.values()],
        }
    )


def table_stats(schema: str, table: str) -> dict:
    """
    Get basic statistics for a parquet table.

    Args:
        schema: Schema name
        table: Table name

    Returns:
        Dict with row_count, column_count, file_size_bytes

    Example:
        >>> stats = table_stats('raw', 'observations')
        >>> print(stats)
        {'row_count': 50000, 'column_count': 3, 'file_size_bytes': 1234567}
    """
    path = get_parquet_path(schema, table)

    if not path.exists():
        return {"row_count": 0, "column_count": 0, "file_size_bytes": 0}

    lf = pl.scan_parquet(path)
    row_count = lf.select(pl.len()).collect().item()
    column_count = len(lf.schema)
    file_size = path.stat().st_size

    return {
        "row_count": row_count,
        "column_count": column_count,
        "file_size_bytes": file_size,
    }


# CLI support
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="PRISM Query Interface")
    parser.add_argument("--describe", "-d", help="Describe table (format: schema.table)")
    parser.add_argument("--stats", help="Show stats for table (format: schema.table)")

    args = parser.parse_args()

    if args.describe:
        parts = args.describe.split(".")
        if len(parts) != 2:
            print("Error: Use format schema.table (e.g., raw.observations)")
            sys.exit(1)
        schema, table = parts
        result = describe_table(schema, table)
        print(result)

    elif args.stats:
        parts = args.stats.split(".")
        if len(parts) != 2:
            print("Error: Use format schema.table (e.g., raw.observations)")
            sys.exit(1)
        schema, table = parts
        stats = table_stats(schema, table)
        for k, v in stats.items():
            print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")

    else:
        parser.print_help()

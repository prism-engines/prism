"""
PRISM Database Layer

Parquet-based storage with Polars I/O.

Modules:
    parquet_store: Path management and directory structure
    polars_io: Atomic writes, upsert operations
    query: Table introspection utilities
    scratch: Temporary storage for parallel workers

Usage:
    from prism.db import get_parquet_path, read_table, write_table

    # Read/write tables
    observations = read_table('raw', 'observations')
    write_table(results, 'vector', 'signals', mode='upsert', key_cols=[...])

    # Query with Polars
    import polars as pl
    obs = pl.read_parquet(get_parquet_path('raw', 'observations'))
    spy_data = obs.filter(pl.col('signal_id') == 'SENSOR_01')
"""

# Path management
from prism.db.parquet_store import (
    get_data_root,
    get_parquet_path,
    get_schema_path,
    ensure_directories,
    list_schemas,
    list_tables,
    table_exists,
    SCHEMAS,
    SCHEMA_TABLES,
)

# Polars I/O
from prism.db.polars_io import (
    read_parquet,
    write_parquet_atomic,
    upsert_parquet,
    append_parquet,
    read_table,
    write_table,
    get_row_count,
    get_parquet_schema,
)

# Query utilities
from prism.db.query import (
    describe_table,
    table_stats,
)

# Temporary storage
from prism.db.scratch import (
    TempParquet,
    ParquetBatchWriter,
    merge_temp_results,
    merge_to_table,
)

__all__ = [
    # parquet_store
    "get_data_root",
    "get_parquet_path",
    "get_schema_path",
    "ensure_directories",
    "list_schemas",
    "list_tables",
    "table_exists",
    "SCHEMAS",
    "SCHEMA_TABLES",
    # polars_io
    "read_parquet",
    "write_parquet_atomic",
    "upsert_parquet",
    "append_parquet",
    "read_table",
    "write_table",
    "get_row_count",
    "get_parquet_schema",
    # query
    "describe_table",
    "table_stats",
    # scratch
    "TempParquet",
    "ParquetBatchWriter",
    "merge_temp_results",
    "merge_to_table",
]

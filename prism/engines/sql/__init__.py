"""
PRISM SQL Engines - Pure SQL computations via DuckDB.

Fast primitives for statistics, z-scores, correlations, and regime assignment.
"""

from pathlib import Path

SQL_DIR = Path(__file__).parent


def get_sql(name: str) -> str:
    """Load SQL file by name."""
    path = SQL_DIR / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"SQL engine not found: {name}")
    return path.read_text()


__all__ = ['get_sql', 'SQL_DIR']

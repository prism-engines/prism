"""
PRISM Engines
=============

Atomic engines organized by execution type:

    python/          - Signal-level (one value per signal)
    python_windowed/ - Observation-level (rolling window)
    sql/             - Pure SQL (DuckDB)

Each engine computes ONE thing. No domain prefixes.
"""

# Lazy imports to allow direct engine access
__all__ = ['python', 'python_windowed', 'sql']


def __getattr__(name):
    """Lazy import of subpackages."""
    if name == 'python':
        from . import python
        return python
    elif name == 'python_windowed':
        from . import python_windowed
        return python_windowed
    elif name == 'sql':
        from . import sql
        return sql
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

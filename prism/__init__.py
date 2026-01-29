"""
PRISM - Signal Analysis Engine
==============================

Math is universal. Domains are Orthon's problem.

Architecture:
    - engines/python/:          Signal-level engines (one value per signal)
    - engines/python_windowed/: Observation-level engines (rolling window)
    - engines/sql/:             SQL engines (DuckDB)
    - runner.py:                ManifestRunner (executes manifests)
    - cli.py:                   Command line interface

Usage:
    # CLI
    python -m prism run --manifest manifest.json
    python -m prism list

    # Python
    from prism.runner import ManifestRunner
    runner = ManifestRunner(manifest)
    runner.run()
"""

__version__ = "3.0.0"
__architecture__ = "atomic"

# Lazy imports to avoid circular dependencies
__all__ = ['runner', 'engines', 'stream', 'server', '__version__']


def __getattr__(name):
    """Lazy import of submodules."""
    if name == 'runner':
        from . import runner
        return runner
    elif name == 'engines':
        from . import engines
        return engines
    elif name == 'stream':
        from . import stream
        return stream
    elif name == 'server':
        from . import server
        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

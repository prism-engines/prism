"""
PRISM Parquet Storage Layer

Path management and directory structure for Parquet-based storage.
Domain-first organization with consistent analysis subfolders.

Directory Structure (domain-first):
    data/
      {domain}/                   # e.g., cmapss/, climate/, cheme/
        raw/
          observations.parquet
          signals.parquet
        config/
          cohort_members.parquet
          cohorts.parquet
          windows.parquet
        filter/
          pairs.parquet
          redundant.parquet
        vector/
          signal.parquet       # individual signal metrics
          cohort.parquet          # aggregated cohort metrics
        geometry/
          signal_pair.parquet  # pairwise between signals
          cohort.parquet          # cohort-level structure
          cohort_pair.parquet     # pairwise between cohorts (future)
        state/
          signal_pair.parquet  # temporal relationships
          cohort.parquet          # cohort temporal dynamics
          cohort_pair.parquet     # cohort pairwise dynamics (future)
        characterization/
          signal.parquet       # 6-axis classification per signal
          cohort.parquet          # 6-axis classification per cohort

      cross_domain/               # cross-domain comparisons (future)
        geometry/
          domain_pair.parquet

Naming Convention:
    - Top folder = domain (cmapss, climate, cheme, etc.)
    - Second folder = analysis type (vector, geometry, state, characterization)
    - Filename = scope (signal, cohort) + _pair suffix for pairwise

Usage:
    # Domain defaults to active domain
    get_parquet_path('vector', 'signal')  # -> data/{domain}/vector/signal.parquet

    # Explicit domain
    get_parquet_path('vector', 'signal', domain='cmapss')  # -> data/cmapss/vector/signal.parquet
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml

# Schema definitions (folders under data/)
SCHEMAS = [
    "raw",
    "config",
    "filter",
    "vector",
    "geometry",
    "state",
    "characterization",
    "physics",
    "delta",      # Delta pipeline: break detection
    "event",      # Event geometry: density/regime analysis
]

# Table definitions per schema
# Naming: {scope}.parquet or {scope}_pair.parquet for pairwise
SCHEMA_TABLES = {
    "raw": ["observations", "signals", "domain_config"],
    "config": ["cohort_members", "cohorts", "windows", "engine_min_obs"],
    "filter": ["pairs", "redundant", "curated"],
    "vector": ["signal", "cohort", "domain"],
    "geometry": ["signal_pair", "cohort", "cohort_pair", "domain"],
    "state": ["signal_pair", "cohort", "cohort_pair", "domain"],
    "characterization": ["signal", "cohort", "domain"],
    "physics": ["signal", "cohort", "conservation"],
    "delta": ["breaks", "timing", "geometry", "pairwise"],  # Delta pipeline
    "event": ["density", "regimes", "sync"],            # Event geometry
}

# =============================================================================
# FILE PATH CONSTANTS - Import these in runners for consistency
# =============================================================================
# Usage: from prism.db.parquet_store import VECTOR_INDICATOR, GEOMETRY_COHORT
#        path = get_parquet_path(*VECTOR_INDICATOR)

# Vector paths (intrinsic properties)
VECTOR_INDICATOR = ("vector", "signal")
VECTOR_COHORT = ("vector", "cohort")
VECTOR_DOMAIN = ("vector", "domain")

# Geometry paths (structural relationships)
GEOMETRY_INDICATOR_PAIR = ("geometry", "signal_pair")
GEOMETRY_COHORT = ("geometry", "cohort")
GEOMETRY_COHORT_PAIR = ("geometry", "cohort_pair")
GEOMETRY_DOMAIN = ("geometry", "domain")

# State paths (temporal dynamics)
STATE_INDICATOR_PAIR = ("state", "signal_pair")
STATE_COHORT = ("state", "cohort")
STATE_COHORT_PAIR = ("state", "cohort_pair")
STATE_DOMAIN = ("state", "domain")

# Characterization paths (6-axis classification)
CHAR_INDICATOR = ("characterization", "signal")
CHAR_COHORT = ("characterization", "cohort")
CHAR_DOMAIN = ("characterization", "domain")

# Config paths
CONFIG_COHORT_MEMBERS = ("config", "cohort_members")
CONFIG_COHORTS = ("config", "cohorts")
CONFIG_WINDOWS = ("config", "windows")

# Delta pipeline paths (break detection)
DELTA_BREAKS = ("delta", "breaks")
DELTA_TIMING = ("delta", "timing")

# Event geometry paths (density/regime analysis)
EVENT_DENSITY = ("event", "density")
EVENT_REGIMES = ("event", "regimes")
EVENT_SYNC = ("event", "sync")


def get_active_domain() -> str:
    """
    Get the active domain from PRISM_DOMAIN environment variable.

    Returns:
        Active domain name (e.g., 'cheme', 'cmapss', 'climate')

    Raises:
        RuntimeError: If no domain is set (PRISM_DOMAIN env var required)
    """
    # Check environment variable - REQUIRED
    env_domain = os.environ.get("PRISM_DOMAIN")
    if env_domain:
        return env_domain

    # No fallback - domain must be explicitly set
    raise RuntimeError(
        "No domain specified. Set PRISM_DOMAIN environment variable or use --domain flag. "
        "Available domains can be listed with: python -m prism.db.parquet_store --list-domains"
    )


def get_data_root(domain: str = None) -> Path:
    """
    Return the root data directory for a domain.

    Args:
        domain: Domain name (cmapss, climate, etc.). Defaults to active domain from config.

    Returns:
        Path to domain data directory (e.g., data/cmapss/)

    Examples:
        get_data_root()           -> ~/prism-mac/data/cmapss/  (if cmapss is active)
        get_data_root('climate')  -> ~/prism-mac/data/climate/
    """
    # Base path from env var or default
    env_path = os.environ.get("PRISM_DATA_PATH")
    if env_path:
        base = Path(env_path)
    else:
        base = Path(os.path.expanduser("~/prism-mac/data"))

    # Always use domain-first structure
    domain = domain or get_active_domain()
    return base / domain


def get_schema_path(schema: str, domain: str = None) -> Path:
    """
    Return the directory path for a schema within a domain.

    Args:
        schema: Schema name (raw, vector, geometry, state, etc.)
        domain: Domain name. Defaults to active domain.

    Returns:
        Path to schema directory (e.g., data/cmapss/vector/)
    """
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema}. Valid schemas: {SCHEMAS}")
    return get_data_root(domain) / schema


def get_parquet_path(schema: str, table: str, domain: str = None) -> Path:
    """
    Return the parquet file path for a table.

    Args:
        schema: Schema name (raw, vector, geometry, state, etc.)
        table: Table name (signal, cohort, signal_pair, etc.)
        domain: Domain name. Defaults to active domain.

    Returns:
        Path to parquet file (e.g., data/cmapss/vector/signal.parquet)

    Examples:
        >>> get_parquet_path('vector', 'signal')
        PosixPath('.../data/cmapss/vector/signal.parquet')

        >>> get_parquet_path('vector', 'signal', domain='g7')
        PosixPath('.../data/g7/vector/signal.parquet')
    """
    return get_schema_path(schema, domain) / f"{table}.parquet"


def ensure_directories(domain: str = None) -> None:
    """
    Create all schema directories if they don't exist.

    Args:
        domain: Optional domain name (oceania, g7, etc.)

    Creates:
        data/{domain}/raw/
        data/{domain}/config/
        data/{domain}/filter/
        data/{domain}/vector/
        data/{domain}/geometry/
        data/{domain}/state/
        data/{domain}/cohort/
    """
    root = get_data_root(domain)
    root.mkdir(parents=True, exist_ok=True)

    for schema in SCHEMAS:
        schema_path = root / schema
        schema_path.mkdir(parents=True, exist_ok=True)


def list_schemas(domain: str = None) -> List[str]:
    """
    List all schema directories that exist.

    Args:
        domain: Optional domain name

    Returns:
        List of schema names that have directories
    """
    root = get_data_root(domain)
    if not root.exists():
        return []

    return [d.name for d in root.iterdir() if d.is_dir() and d.name in SCHEMAS]


def list_domains() -> List[str]:
    """
    List all domain directories under PRISM_DATA_PATH.

    Returns:
        List of domain names that have directories
    """
    data_root = get_data_root()
    if not data_root.exists():
        return []

    # Filter to only directories that look like domains (have schema subdirs)
    domains = []
    for d in data_root.iterdir():
        if d.is_dir() and (d / "raw").exists():
            domains.append(d.name)
    return sorted(domains)


def list_tables(schema: str, domain: str = None) -> List[str]:
    """
    List all parquet tables in a schema.

    Args:
        schema: Schema name
        domain: Optional domain name

    Returns:
        List of table names (without .parquet extension)
    """
    schema_path = get_schema_path(schema, domain)
    if not schema_path.exists():
        return []

    return [
        f.stem for f in schema_path.iterdir() if f.is_file() and f.suffix == ".parquet"
    ]


def table_exists(schema: str, table: str, domain: str = None) -> bool:
    """
    Check if a parquet table exists.

    Args:
        schema: Schema name
        table: Table name
        domain: Optional domain name

    Returns:
        True if the parquet file exists
    """
    return get_parquet_path(schema, table, domain).exists()


def get_table_size(schema: str, table: str, domain: str = None) -> Optional[int]:
    """
    Get the size of a parquet file in bytes.

    Args:
        schema: Schema name
        table: Table name
        domain: Optional domain name

    Returns:
        File size in bytes, or None if file doesn't exist
    """
    path = get_parquet_path(schema, table, domain)
    if not path.exists():
        return None
    return path.stat().st_size


def delete_table(schema: str, table: str, domain: str = None) -> bool:
    """
    Delete a parquet table.

    Args:
        schema: Schema name
        table: Table name
        domain: Optional domain name

    Returns:
        True if file was deleted, False if it didn't exist
    """
    path = get_parquet_path(schema, table, domain)
    if path.exists():
        path.unlink()
        return True
    return False


def get_all_parquet_paths(domain: str = None) -> List[Path]:
    """
    Get paths to all existing parquet files.

    Args:
        domain: Optional domain name

    Returns:
        List of paths to all parquet files across all schemas
    """
    paths = []
    for schema in list_schemas(domain):
        schema_path = get_schema_path(schema, domain)
        paths.extend(schema_path.glob("*.parquet"))
    return sorted(paths)


def validate_schema_structure(domain: str = None) -> dict:
    """
    Validate that the expected schema structure exists.

    Args:
        domain: Optional domain name

    Returns:
        Dict with 'valid' bool and 'missing' list of missing tables
    """
    missing = []
    for schema, tables in SCHEMA_TABLES.items():
        for table in tables:
            if not table_exists(schema, table, domain):
                missing.append(f"{schema}.{table}")

    return {"valid": len(missing) == 0, "missing": missing}


# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Parquet Storage Management")
    parser.add_argument("--domain", help="Domain name (cmapss, climate, cheme, etc.)")
    parser.add_argument("--init", action="store_true", help="Initialize directory structure")
    parser.add_argument("--list", action="store_true", help="List all tables")
    parser.add_argument("--list-domains", action="store_true", help="List all domains")
    parser.add_argument("--validate", action="store_true", help="Validate schema structure")
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")

    args = parser.parse_args()

    if args.init:
        ensure_directories(args.domain)
        root = get_data_root(args.domain)
        print(f"Initialized directories at {root}")

    elif args.list_domains:
        domains = list_domains()
        if domains:
            print("Domains:")
            for d in sorted(domains):
                print(f"  {d}/")
        else:
            print("No domains found in ./data/")

    elif args.list:
        domain = args.domain
        if domain:
            print(f"Domain: {domain}")
        for schema in list_schemas(domain):
            tables = list_tables(schema, domain)
            if tables:
                print(f"{schema}/")
                for table in tables:
                    size = get_table_size(schema, table, domain)
                    size_str = f"{size:,} bytes" if size else "?"
                    print(f"  {table}.parquet ({size_str})")

    elif args.validate:
        result = validate_schema_structure(args.domain)
        if result["valid"]:
            print("Schema structure is valid")
        else:
            print("Missing tables:")
            for missing in result["missing"]:
                print(f"  - {missing}")

    elif args.stats:
        total_size = 0
        total_files = 0
        for path in get_all_parquet_paths(args.domain):
            total_size += path.stat().st_size
            total_files += 1
        print(f"Data root: {get_data_root(args.domain)}")
        print(f"Total files: {total_files}")
        print(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

    else:
        parser.print_help()

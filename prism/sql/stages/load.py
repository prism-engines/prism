"""
Load Stage Orchestrator

PURE: Loads 00_load.sql, creates base views.
NO computation. NO inline SQL.
"""

from .base import StageOrchestrator


class LoadStage(StageOrchestrator):
    """Load observations and create base view."""

    SQL_FILE = '00_load.sql'

    VIEWS = [
        'v_base',
        'v_schema_validation',
        'v_signal_inventory',
        'v_data_quality',
    ]

    DEPENDS_ON = []  # First stage, no dependencies

    def load_observations(self, path: str) -> None:
        """
        Load observations parquet into database.

        PURE: Just creates table from file. No transformation.
        """
        self.conn.execute(f"CREATE OR REPLACE TABLE observations AS SELECT * FROM '{path}'")

    def get_row_count(self) -> int:
        """Return number of rows loaded."""
        return self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def get_signal_count(self) -> int:
        """Return number of distinct signals."""
        return self.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]

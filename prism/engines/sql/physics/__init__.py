"""
SQL Physics Reports

SQL queries for analyzing physics engine outputs.

Reports:
- 10_energy_reports.sql: Energy balance analysis
- 11_mass_reports.sql: Mass balance and leak/blockage detection
- 12_momentum_reports.sql: Torque/force balance and vibration analysis
- 13_constitutive_reports.sql: Constitutive relationship drift tracking
"""

from pathlib import Path

SQL_DIR = Path(__file__).parent

REPORTS = {
    'energy': SQL_DIR / '10_energy_reports.sql',
    'mass': SQL_DIR / '11_mass_reports.sql',
    'momentum': SQL_DIR / '12_momentum_reports.sql',
    'constitutive': SQL_DIR / '13_constitutive_reports.sql',
}


def get_report_sql(report_name: str) -> str:
    """Load SQL for a report."""
    if report_name not in REPORTS:
        raise ValueError(f"Unknown report: {report_name}. Available: {list(REPORTS.keys())}")
    return REPORTS[report_name].read_text()


def list_reports() -> list:
    """List available reports."""
    return list(REPORTS.keys())

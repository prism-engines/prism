"""
SQL Structure Reports

SQL queries for analyzing structure engine outputs.

Reports:
- covariance_report.sql: Correlation and covariance analysis
- eigenvalue_report.sql: PCA and dimensionality analysis
- koopman_report.sql: DMD stability and dynamics analysis
- spectral_report.sql: Frequency domain coherence analysis
- wavelet_report.sql: Time-frequency coherence analysis
- summary_report.sql: Combined multi-engine analysis
"""

from pathlib import Path

SQL_DIR = Path(__file__).parent

REPORTS = {
    'covariance': SQL_DIR / 'covariance_report.sql',
    'eigenvalue': SQL_DIR / 'eigenvalue_report.sql',
    'koopman': SQL_DIR / 'koopman_report.sql',
    'spectral': SQL_DIR / 'spectral_report.sql',
    'wavelet': SQL_DIR / 'wavelet_report.sql',
    'summary': SQL_DIR / 'summary_report.sql',
}


def get_report_sql(report_name: str) -> str:
    """Load SQL for a report."""
    if report_name not in REPORTS:
        raise ValueError(f"Unknown report: {report_name}. Available: {list(REPORTS.keys())}")
    return REPORTS[report_name].read_text()


def list_reports() -> list:
    """List available reports."""
    return list(REPORTS.keys())

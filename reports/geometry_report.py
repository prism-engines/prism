#!/usr/bin/env python3
"""
PRISM Geometry Report — Coupling, structure, and PCA analysis.

Shows:
- System coherence and coupling metrics
- PCA variance explained (collapse to single mode = failure)
- Cohort-level structural patterns
- Divergence analysis (stress sources)

Usage:
    python -m reports.geometry_report
    python -m reports.geometry_report --domain cmapss
    python -m reports.geometry_report --output report.md
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import polars as pl

from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, STATE
from reports.report_utils import (
    ReportBuilder,
    load_domain_config,
    translate_signal_id,
    format_number,
    format_percentage,
    get_time_unit,
)


# =============================================================================
# Geometry Interpretation
# =============================================================================

def interpret_pca_variance(pc1_var: float) -> str:
    """Interpret PC1 variance ratio."""
    if pc1_var is None:
        return "Unknown"
    if pc1_var > 0.9:
        return "CRITICAL: System collapsed to single mode (failure imminent)"
    if pc1_var > 0.7:
        return "WARNING: Reduced complexity, dominant failure mode emerging"
    if pc1_var > 0.5:
        return "CAUTION: Moderate complexity reduction"
    return "NORMAL: Healthy multi-dimensional behavior"


def interpret_coherence(coherence: float) -> str:
    """Interpret system coherence."""
    if coherence is None:
        return "Unknown"
    if coherence < 0.3:
        return "LOW: Signals decoupled, system fragmented"
    if coherence < 0.6:
        return "MODERATE: Partial coupling"
    return "HIGH: Strong inter-signal coupling"


def interpret_divergence(divergence: float) -> str:
    """Interpret divergence (stress source indicator)."""
    if divergence is None:
        return "Unknown"
    if divergence < -0.5:
        return "STRESS SOURCE: This component is driving instability"
    if divergence < 0:
        return "MILD STRESS: Minor negative contribution"
    if divergence > 0.5:
        return "STABILIZER: This component absorbs disturbances"
    return "NEUTRAL: Neither source nor sink"


# =============================================================================
# Report Generation
# =============================================================================

def generate_geometry_report(domain: str = None) -> ReportBuilder:
    """Generate geometry report."""
    
    config = load_domain_config(domain) if domain else {}
    time_unit = get_time_unit(config)
    report = ReportBuilder("Geometry Report", domain=domain)
    
    # Load geometry (single unified file now)
    geometry_path = get_path(GEOMETRY)

    geometry = None
    cohort_geometry = None

    if Path(geometry_path).exists():
        geometry = pl.read_parquet(geometry_path)
        cohort_geometry = geometry  # Same file now
    
    if geometry is None and cohort_geometry is None:
        report.add_section("Error", "No geometry data found. Run geometry entry point first.")
        return report
    
    # ==========================================================================
    # Key Metrics
    # ==========================================================================
    if geometry is not None:
        n_rows = len(geometry)
        # Use timestamp as window identifier (not window_id)
        n_timestamps = geometry['timestamp'].n_unique() if 'timestamp' in geometry.columns else 0
        n_entities = geometry['entity_id'].n_unique() if 'entity_id' in geometry.columns else 0
        report.add_metric("Geometry Snapshots", n_rows)
        report.add_metric("Timestamps", n_timestamps)
        report.add_metric("Entities", n_entities)
    
    # ==========================================================================
    # PCA Analysis
    # ==========================================================================
    pca_cols = [c for c in (geometry.columns if geometry is not None else []) 
                if 'pca' in c.lower() or 'variance' in c.lower()]
    
    if not pca_cols and cohort_geometry is not None:
        pca_cols = [c for c in cohort_geometry.columns 
                    if 'pca' in c.lower() or 'variance' in c.lower()]
        geometry = cohort_geometry  # Use cohort geometry for PCA
    
    if pca_cols and geometry is not None:
        # Get latest timestamp's PCA stats
        if 'timestamp' in geometry.columns:
            latest = geometry.filter(pl.col('timestamp') == geometry['timestamp'].max())
        else:
            latest = geometry.tail(1)

        # Support both column naming conventions
        pc1_col = 'pca_var_1' if 'pca_var_1' in geometry.columns else 'pca_variance_1'
        pc2_col = 'pca_var_2' if 'pca_var_2' in geometry.columns else 'pca_variance_2'
        eff_dim_col = 'pca_effective_dim' if 'pca_effective_dim' in geometry.columns else None

        if pc1_col in geometry.columns:
            pc1 = latest[pc1_col].mean() if len(latest) > 0 else None
            pc2 = latest[pc2_col].mean() if pc2_col in latest.columns and len(latest) > 0 else None
            eff_dim = latest[eff_dim_col].mean() if eff_dim_col and eff_dim_col in latest.columns and len(latest) > 0 else None
            pc3 = None  # Don't use effective_dim as pc3
            
            report.add_metric("PC1 Variance", format_percentage(pc1) if pc1 else "N/A")
            report.add_metric("PC2 Variance", format_percentage(pc2) if pc2 else "N/A")
            if eff_dim:
                report.add_metric("Effective Dimension", format_number(eff_dim, 1))

            # Cumulative (PC1 + PC2)
            if pc1 and pc2:
                cum2 = pc1 + pc2
                report.add_metric("Top 2 PCs (cumulative)", format_percentage(cum2))
            
            # Interpretation
            if pc1:
                report.add_section(
                    "PCA Interpretation",
                    f"PC1 explains {format_percentage(pc1)} of variance.\n\n"
                    f"**Assessment:** {interpret_pca_variance(pc1)}\n\n"
                    "Note: Failing systems collapse toward PC1 variance → 1.0\n"
                    "(All variation explained by single mode = loss of complexity)"
                )
        
        # PCA over time
        if 'timestamp' in geometry.columns and pc1_col in geometry.columns:
            pca_trend = (
                geometry
                .group_by('timestamp')
                .agg([
                    pl.col(pc1_col).mean().alias('pc1_mean'),
                ])
                .sort('timestamp')
            )
            
            if len(pca_trend) >= 2:
                first_pc1 = pca_trend['pc1_mean'][0]
                last_pc1 = pca_trend['pc1_mean'][-1]
                
                if first_pc1 and last_pc1:
                    delta = last_pc1 - first_pc1
                    direction = "INCREASING ↑" if delta > 0.05 else ("DECREASING ↓" if delta < -0.05 else "STABLE →")
                    
                    report.add_section(
                        "PCA Trend",
                        f"PC1 variance: {format_percentage(first_pc1)} → {format_percentage(last_pc1)}\n"
                        f"Change: {format_percentage(delta)} ({direction})\n\n"
                        "Rising PC1 = complexity loss = potential degradation"
                    )
    
    # ==========================================================================
    # Coherence Analysis
    # ==========================================================================
    coherence_col = None
    for col in ['system_coherence', 'coherence', 'coupling_strength']:
        if geometry is not None and col in geometry.columns:
            coherence_col = col
            break
    
    if coherence_col:
        coherence_stats = geometry.select([
            pl.col(coherence_col).mean().alias('mean'),
            pl.col(coherence_col).std().alias('std'),
            pl.col(coherence_col).min().alias('min'),
            pl.col(coherence_col).max().alias('max'),
        ]).row(0, named=True)
        
        report.add_section(
            "System Coherence",
            f"Mean coherence: {format_number(coherence_stats['mean'], 3)}\n"
            f"Range: {format_number(coherence_stats['min'], 3)} → {format_number(coherence_stats['max'], 3)}\n\n"
            f"**Assessment:** {interpret_coherence(coherence_stats['mean'])}\n\n"
            "Note: Failing systems LOSE coherence — signals decouple before failure."
        )
    
    # ==========================================================================
    # Divergence Analysis (Stress Sources)
    # ==========================================================================
    divergence_col = None
    for col in ['divergence', 'stress_contribution', 'energy_flow']:
        if geometry is not None and col in geometry.columns:
            divergence_col = col
            break
    
    if divergence_col and 'signal_id' in geometry.columns:
        # Find stress sources (negative divergence)
        stress_sources = (
            geometry
            .group_by('signal_id')
            .agg(pl.col(divergence_col).mean().alias('avg_divergence'))
            .sort('avg_divergence')
        )
        
        rows = []
        for row in stress_sources.head(10).iter_rows(named=True):
            signal_name = translate_signal_id(row['signal_id'], config)
            interpretation = interpret_divergence(row['avg_divergence'])
            rows.append([
                row['signal_id'],
                signal_name,
                format_number(row['avg_divergence'], 3),
                interpretation.split(':')[0],  # Just the label
            ])
        
        report.add_table(
            "Divergence Analysis (Stress Sources)",
            ["Signal ID", "Name", "Divergence", "Role"],
            rows,
            alignments=['l', 'l', 'r', 'l'],
        )
        
        report.add_section(
            "Divergence Interpretation",
            "Negative divergence = STRESS SOURCE (driving instability)\n"
            "Positive divergence = STABILIZER (absorbing disturbances)\n\n"
            "Monitor stress sources — they often lead degradation."
        )
    
    # ==========================================================================
    # Cohort-Level Structure
    # ==========================================================================
    if cohort_geometry is not None and 'cohort_id' in cohort_geometry.columns:
        # Get structural metrics per cohort
        struct_cols = [c for c in cohort_geometry.columns 
                       if c not in ['cohort_id', 'window_id', 'timestamp', 'window_start', 'window_end']]
        
        if struct_cols:
            cohort_summary = (
                cohort_geometry
                .group_by('cohort_id')
                .agg([
                    pl.col(struct_cols[0]).mean().alias('metric_mean') if struct_cols else None,
                    pl.count().alias('n_windows'),
                ])
                .sort('cohort_id')
            )
            
            rows = []
            for row in cohort_summary.iter_rows(named=True):
                rows.append([
                    str(row['cohort_id']),
                    str(row['n_windows']),
                    format_number(row['metric_mean'], 3) if row.get('metric_mean') else "N/A",
                ])
            
            report.add_table(
                "Cohort Structure Summary",
                ["Cohort", "Windows", f"{struct_cols[0]} (mean)" if struct_cols else "Metric"],
                rows,
                alignments=['l', 'r', 'r'],
            )
    
    # ==========================================================================
    # Key Findings
    # ==========================================================================
    findings = []
    
    # Check for high PC1 (system collapse)
    if 'pca_variance_1' in (geometry.columns if geometry is not None else []):
        max_pc1 = geometry['pca_variance_1'].max()
        if max_pc1 and max_pc1 > 0.8:
            findings.append(f"⚠️ PC1 reached {format_percentage(max_pc1)} — system approaching single-mode collapse")
    
    # Check for coherence loss
    if coherence_col and geometry is not None:
        min_coherence = geometry[coherence_col].min()
        if min_coherence and min_coherence < 0.3:
            findings.append(f"⚠️ Coherence dropped to {format_number(min_coherence, 2)} — significant decoupling detected")
    
    if findings:
        report.add_section("⚠️ Alerts", "\n".join(findings))
    else:
        report.add_section("Status", "✓ No critical geometry alerts")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='PRISM Geometry Report')
    parser.add_argument('--domain', type=str, default=None, help='Domain name')
    parser.add_argument('--output', type=str, default=None, help='Output file (md or json)')
    args = parser.parse_args()
    
    if not args.domain:
        import os
        args.domain = os.environ.get('PRISM_DOMAIN')
    
    report = generate_geometry_report(args.domain)
    
    if args.output:
        if args.output.endswith('.json'):
            report.save_json(args.output)
        else:
            report.save_markdown(args.output)
        print(f"Report saved to: {args.output}")
    else:
        print(report.to_text())


if __name__ == "__main__":
    main()

"""
Export page - Download analysis results.

Provides various export formats for data and reports.
"""

import streamlit as st
import pandas as pd
import json
from typing import Optional
from datetime import datetime
from pathlib import Path


def render(
    signals_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    geometry_df: pd.DataFrame = None,
    dynamics_df: pd.DataFrame = None,
    mechanics_df: pd.DataFrame = None,
    data_dir=None,
):
    """Render the export page."""
    st.header("Export Data")

    name = st.session_state.get('current_example', 'dataset')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    st.markdown("Download your analysis results in various formats.")

    st.divider()

    # Data exports
    st.subheader("📊 Data Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Signal Data**")
        if signals_df is not None:
            st.caption(f"{len(signals_df)} rows")

            # CSV
            csv_signals = signals_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_signals,
                file_name=f"{name}_signals_{timestamp}.csv",
                mime="text/csv",
                key="signals_csv",
            )

            # Parquet
            parquet_signals = signals_df.to_parquet()
            st.download_button(
                "Download Parquet",
                parquet_signals,
                file_name=f"{name}_signals_{timestamp}.parquet",
                mime="application/octet-stream",
                key="signals_parquet",
            )
        else:
            st.info("No signal data loaded")

    with col2:
        st.markdown("**Typology Profile**")
        if profile_df is not None:
            st.caption(f"{len(profile_df)} signals")

            csv_profile = profile_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_profile,
                file_name=f"{name}_typology_{timestamp}.csv",
                mime="text/csv",
                key="typology_csv",
            )

            parquet_profile = profile_df.to_parquet()
            st.download_button(
                "Download Parquet",
                parquet_profile,
                file_name=f"{name}_typology_{timestamp}.parquet",
                mime="application/octet-stream",
                key="typology_parquet",
            )
        else:
            st.info("No typology data available")

    st.divider()

    # Additional data
    st.subheader("📈 Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Geometry**")
        if geometry_df is not None and len(geometry_df) > 0:
            csv_geo = geometry_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_geo,
                file_name=f"{name}_geometry_{timestamp}.csv",
                mime="text/csv",
                key="geometry_csv",
            )
        else:
            st.caption("Not computed")

    with col2:
        st.markdown("**Dynamics**")
        if dynamics_df is not None and len(dynamics_df) > 0:
            csv_dyn = dynamics_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_dyn,
                file_name=f"{name}_dynamics_{timestamp}.csv",
                mime="text/csv",
                key="dynamics_csv",
            )
        else:
            st.caption("Not computed")

    with col3:
        st.markdown("**Mechanics**")
        if mechanics_df is not None and len(mechanics_df) > 0:
            csv_mech = mechanics_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_mech,
                file_name=f"{name}_mechanics_{timestamp}.csv",
                mime="text/csv",
                key="mechanics_csv",
            )
        else:
            st.caption("Not computed")

    st.divider()

    # Combined export
    st.subheader("📦 Complete Export")

    st.markdown("Download all available data as a combined JSON file.")

    if st.button("Generate Complete Export", type="primary"):
        export_data = create_complete_export(
            signals_df, profile_df, geometry_df, dynamics_df, mechanics_df
        )

        json_str = json.dumps(export_data, indent=2, default=str)

        st.download_button(
            "📥 Download JSON",
            json_str,
            file_name=f"{name}_complete_{timestamp}.json",
            mime="application/json",
            key="complete_json",
        )

    st.divider()

    # API/Integration info
    st.subheader("🔗 Integration")

    with st.expander("Python Integration"):
        st.code("""
import pandas as pd

# Load exported data
typology = pd.read_csv('typology.csv')
signals = pd.read_parquet('signals.parquet')

# Access typology scores
memory_scores = typology['memory']
volatile_signals = typology[typology['volatility'] > 0.7]
        """, language="python")

    with st.expander("Command Line"):
        st.code(f"""
# Run full ORTHON pipeline
python -m prism.entry_points.run --input {name}.csv

# Run individual layers
python -m prism.entry_points.signal_typology
python -m prism.entry_points.structural_geometry
python -m prism.entry_points.dynamical_systems
python -m prism.entry_points.causal_mechanics
        """, language="bash")


def create_complete_export(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    mechanics_df: pd.DataFrame,
) -> dict:
    """Create complete export dictionary."""

    export = {
        'metadata': {
            'name': st.session_state.get('current_example', 'dataset'),
            'exported_at': datetime.now().isoformat(),
            'orthon_version': '0.1.0',
        },
        'data': {},
    }

    if signals_df is not None:
        export['data']['signals'] = {
            'n_rows': len(signals_df),
            'columns': signals_df.columns.tolist(),
            'sample': signals_df.head(10).to_dict('records'),
        }

    if profile_df is not None:
        export['data']['typology'] = profile_df.to_dict('records')

    if geometry_df is not None and len(geometry_df) > 0:
        export['data']['geometry'] = geometry_df.to_dict('records')

    if dynamics_df is not None and len(dynamics_df) > 0:
        export['data']['dynamics'] = dynamics_df.to_dict('records')

    if mechanics_df is not None and len(mechanics_df) > 0:
        export['data']['mechanics'] = mechanics_df.to_dict('records')

    # Include analysis results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        export['analysis'] = {
            'groups': results.get('groups', {}),
            'dynamics_summary': results.get('dynamics', {}),
            'mechanics_summary': results.get('mechanics', {}),
        }

    return export

"""
Report page - Generated analysis report.

Creates formatted reports from analysis results.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime


def render(
    signals_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    geometry_df: pd.DataFrame = None,
    dynamics_df: pd.DataFrame = None,
    mechanics_df: pd.DataFrame = None,
    data_dir=None,
):
    """Render the report page."""

    # Horizontal tabs
    tab1, tab2, tab3 = st.tabs(["Executive", "Technical", "Thesis"])

    with tab1:
        render_executive_report(
            signals_df, profile_df, dynamics_df, mechanics_df
        )

    with tab2:
        render_technical_report(
            signals_df, profile_df, geometry_df, dynamics_df, mechanics_df
        )

    with tab3:
        render_thesis_format(
            signals_df, profile_df, dynamics_df, mechanics_df
        )


def render_executive_report(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    mechanics_df: pd.DataFrame,
):
    """Render executive summary report."""
    st.header("Executive Summary")

    # Dataset info
    name = st.session_state.get('current_example', 'Dataset')
    n_signals = len(profile_df) if profile_df is not None else 0
    n_samples = len(signals_df) if signals_df is not None else 0

    st.markdown(f"**Dataset:** {name}")
    st.markdown(f"**Signals:** {n_signals} | **Samples:** {n_samples}")
    st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    # Key Findings section
    st.subheader("Key Findings")

    with st.container(border=True):
        # Typology findings
        if profile_df is not None and len(profile_df) > 0:
            st.markdown("**1. Signal Typology**")

            # Count dominant traits
            if 'dominant_trait' in profile_df.columns:
                trait_counts = profile_df['dominant_trait'].value_counts()
                traits_desc = ", ".join([f"{count} {trait}" for trait, count in trait_counts.head(3).items()])
                st.markdown(f"- Signals classified as: {traits_desc}")

            # High memory signals
            if 'memory' in profile_df.columns:
                high_mem = profile_df[profile_df['memory'] > 0.7]
                if len(high_mem) > 0:
                    st.markdown(f"- {len(high_mem)} signals show persistent behavior (high memory)")

            # High volatility signals
            if 'volatility' in profile_df.columns:
                high_vol = profile_df[profile_df['volatility'] > 0.7]
                if len(high_vol) > 0:
                    st.markdown(f"- {len(high_vol)} signals show volatility clustering")

        st.markdown("")

        # Groups findings
        if 'groups' in st.session_state.get('analysis_results', {}):
            groups = st.session_state.analysis_results['groups']
            n_clusters = groups.get('n_clusters', 0)
            if n_clusters > 1:
                st.markdown("**2. Signal Groups**")
                st.markdown(f"- {n_clusters} distinct behavioral groups identified")
                silhouette = groups.get('silhouette', 0)
                st.markdown(f"- Clustering quality: {silhouette:.2f} (silhouette score)")

        st.markdown("")

        # Dynamics findings
        st.markdown("**3. System Dynamics**")
        if 'analysis_results' in st.session_state:
            dynamics = st.session_state.analysis_results.get('dynamics', {})
            mean_coh = dynamics.get('mean_coherence', 0.5)
            st.markdown(f"- Mean coherence: {mean_coh:.2f}")

            transitions = dynamics.get('transitions', [])
            if transitions:
                st.markdown(f"- {len(transitions)} regime transition(s) detected")
            else:
                st.markdown("- No significant transitions detected")

        st.markdown("")

        # Causality findings
        st.markdown("**4. Causal Structure**")
        if 'analysis_results' in st.session_state:
            mechanics = st.session_state.analysis_results.get('mechanics', {})
            drivers = mechanics.get('drivers', [])
            if drivers:
                st.markdown(f"- Primary driver: **{drivers[0]}**")
                followers = mechanics.get('followers', [])
                if followers:
                    st.markdown(f"- Primary followers: {', '.join(followers[:3])}")
            else:
                st.markdown("- No dominant causal drivers identified")

    st.divider()

    # Download button
    report_text = generate_report_text(signals_df, profile_df, dynamics_df, mechanics_df)
    st.download_button(
        "📄 Download Report (TXT)",
        report_text,
        file_name=f"orthon_report_{name}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )


def render_technical_report(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    mechanics_df: pd.DataFrame,
):
    """Render technical details report."""
    st.header("Technical Report")

    # Signal Typology
    st.subheader("Signal Typology Metrics")
    if profile_df is not None:
        st.dataframe(profile_df, hide_index=True)

        # Download
        csv = profile_df.to_csv(index=False)
        st.download_button(
            "Download Typology CSV",
            csv,
            file_name="typology.csv",
            mime="text/csv",
        )
    else:
        st.info("No typology data available")

    st.divider()

    # Geometry
    st.subheader("Structural Geometry")
    if geometry_df is not None and len(geometry_df) > 0:
        st.dataframe(geometry_df.head(20), hide_index=True)
    else:
        st.info("Run structural geometry analysis to populate this section")

    st.divider()

    # Dynamics
    st.subheader("Dynamical Systems")
    if dynamics_df is not None and len(dynamics_df) > 0:
        st.dataframe(dynamics_df.head(20), hide_index=True)
    else:
        st.info("Run dynamical systems analysis to populate this section")

    st.divider()

    # Mechanics
    st.subheader("Causal Mechanics")
    if mechanics_df is not None and len(mechanics_df) > 0:
        st.dataframe(mechanics_df.head(20), hide_index=True)
    else:
        st.info("Run causal mechanics analysis to populate this section")


def render_thesis_format(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    mechanics_df: pd.DataFrame,
):
    """Render thesis-style scientific report."""
    st.header("Scientific Report")

    name = st.session_state.get('current_example', 'Dataset')
    n_signals = len(profile_df) if profile_df is not None else 0
    n_samples = len(signals_df) if signals_df is not None else 0

    with st.container(border=True):
        st.markdown("### Abstract")
        st.markdown(f"""
This analysis of {n_signals} signals from {name} reveals the behavioral
characteristics and causal structure of the system. Signal typology identifies
distinct behavioral profiles across nine measurement axes including memory
(Hurst exponent), information content (permutation entropy), and volatility
clustering (GARCH). Structural geometry and dynamical systems analysis
characterize inter-signal relationships and temporal evolution.
        """)

    st.divider()

    st.markdown("### 1. Introduction")
    st.markdown(f"""
The dataset contains {n_signals} signals with {n_samples} observations each.
ORTHON's analysis framework proceeds through four layers:

1. **Signal Typology** — Characterizing individual signal behavior
2. **Structural Geometry** — Mapping inter-signal relationships
3. **Dynamical Systems** — Tracking temporal evolution
4. **Causal Mechanics** — Identifying directional influences
    """)

    st.divider()

    st.markdown("### 2. Methodology")

    st.markdown("**2.1 Signal Typology**")
    st.markdown("Each signal is characterized along nine behavioral axes:")

    with st.expander("Memory (Hurst Exponent)"):
        st.latex(r"H = \lim_{n \to \infty} \frac{\log(R/S)}{\log(n)}")
        st.markdown("""
The Hurst exponent measures long-range dependence:
- H > 0.5: Persistent (trends continue)
- H = 0.5: Random walk
- H < 0.5: Anti-persistent (mean-reverting)
        """)

    with st.expander("Information (Permutation Entropy)"):
        st.latex(r"H_p = -\sum_{i=1}^{n!} p(\pi_i) \log p(\pi_i)")
        st.markdown("""
Permutation entropy measures complexity:
- Low: Predictable, regular patterns
- High: Complex, unpredictable dynamics
        """)

    with st.expander("Volatility (GARCH)"):
        st.latex(r"\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2")
        st.markdown("""
GARCH modeling captures volatility clustering:
- High α+β: Volatility clusters persist
- Low α+β: Stable variance
        """)

    st.markdown("**2.2 Causal Analysis**")

    with st.expander("Granger Causality"):
        st.latex(r"Y_t = \sum_{i=1}^{p} \alpha_i Y_{t-i} + \sum_{j=1}^{q} \beta_j X_{t-j} + \epsilon_t")
        st.markdown("""
Tests whether past values of X help predict Y:
- Significant β coefficients indicate X Granger-causes Y
        """)

    with st.expander("Transfer Entropy"):
        st.latex(r"TE_{X \to Y} = H(Y_t | Y_{t-1}^{(k)}) - H(Y_t | Y_{t-1}^{(k)}, X_{t-1}^{(l)})")
        st.markdown("""
Measures directed information transfer in bits:
- Non-zero TE indicates information flows from X to Y
        """)

    st.divider()

    st.markdown("### 3. Results")
    st.markdown("Results are presented in the data tables above.")

    st.divider()

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Export Options")
        st.button("📄 Export LaTeX", disabled=True, help="Coming soon")
    with col2:
        st.markdown("### Citation")
        st.code("""
@software{orthon2026,
  author = {ORTHON Team},
  title = {ORTHON: Domain-Agnostic Signal Analysis},
  year = {2026},
  url = {https://github.com/orthon/orthon}
}
        """, language="bibtex")


def generate_report_text(
    signals_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    mechanics_df: pd.DataFrame,
) -> str:
    """Generate plain text report."""
    lines = []

    name = st.session_state.get('current_example', 'Dataset')
    n_signals = len(profile_df) if profile_df is not None else 0
    n_samples = len(signals_df) if signals_df is not None else 0

    lines.append("=" * 60)
    lines.append("ORTHON ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset: {name}")
    lines.append(f"Signals: {n_signals}")
    lines.append(f"Samples: {n_samples}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("SIGNAL TYPOLOGY")
    lines.append("-" * 60)

    if profile_df is not None:
        lines.append(profile_df.to_string())
    else:
        lines.append("No typology data available")

    lines.append("")
    lines.append("-" * 60)
    lines.append("KEY FINDINGS")
    lines.append("-" * 60)

    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        groups = results.get('groups', {})
        n_clusters = groups.get('n_clusters', 0)
        lines.append(f"- Groups detected: {n_clusters}")

        dynamics = results.get('dynamics', {})
        lines.append(f"- Mean coherence: {dynamics.get('mean_coherence', 0):.2f}")

        mechanics = results.get('mechanics', {})
        drivers = mechanics.get('drivers', [])
        if drivers:
            lines.append(f"- Primary driver: {drivers[0]}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)

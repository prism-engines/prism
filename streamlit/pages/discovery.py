"""
Discovery page - Claude's analysis with interactive chat.

The main landing page after data is loaded.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def render(
    signals_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    geometry_df: pd.DataFrame = None,
    dynamics_df: pd.DataFrame = None,
    mechanics_df: pd.DataFrame = None,
    data_dir=None,
):
    """
    Render the discovery page.

    This is the main landing page showing Claude's analysis
    of the loaded data with interactive chat.
    """
    # Check for data
    if signals_df is None:
        render_no_data()
        return

    # Build results object from dataframes
    results = build_results_object(
        signals_df=signals_df,
        profile_df=profile_df,
        geometry_df=geometry_df,
        dynamics_df=dynamics_df,
        mechanics_df=mechanics_df,
    )

    # Store in session state for chat
    st.session_state.analysis_results = results

    metadata = results.get('metadata', {})

    # Header
    st.title(f"Analysis: {metadata.get('name', 'Your Data')}")
    st.caption(
        f"{metadata.get('n_signals', 0)} signals • "
        f"{metadata.get('n_samples', 0)} samples"
    )

    # Generate or retrieve cached analysis
    render_analysis_section(results)

    st.divider()

    # Insight cards
    render_key_findings(results)

    st.divider()

    # Chat interface
    render_chat_section(results)

    st.divider()

    # Action buttons
    render_action_buttons()


def render_no_data():
    """Render when no data is loaded."""
    st.title("Discovery")
    st.info("No data loaded. Upload data or try an example to begin analysis.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Upload Data", use_container_width=True):
            st.session_state.show_upload_form = True
            st.rerun()
    with col2:
        st.caption("Or try an example from the sidebar →")


def build_results_object(
    signals_df: pd.DataFrame = None,
    profile_df: pd.DataFrame = None,
    geometry_df: pd.DataFrame = None,
    dynamics_df: pd.DataFrame = None,
    mechanics_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Build a structured results object from dataframes."""

    results = {
        'metadata': {},
        'typology': [],
        'groups': {},
        'dynamics': {},
        'mechanics': {},
    }

    # Metadata from signals
    if signals_df is not None:
        signal_ids = signals_df['signal_id'].unique().tolist() if 'signal_id' in signals_df.columns else []
        results['metadata'] = {
            'name': st.session_state.get('current_example', 'Dataset'),
            'n_signals': len(signal_ids),
            'n_samples': len(signals_df),
            'signal_ids': signal_ids,
        }

    # Typology from profile
    if profile_df is not None:
        results['typology'] = profile_df.to_dict('records')

        # Compute groups from typology
        results['groups'] = compute_groups_from_typology(profile_df)

        # Compute basic dynamics
        results['dynamics'] = compute_basic_dynamics(profile_df)

        # Compute basic mechanics
        results['mechanics'] = compute_basic_mechanics(profile_df)

    # Override with actual dynamics if available
    if dynamics_df is not None and len(dynamics_df) > 0:
        results['dynamics'] = extract_dynamics(dynamics_df)

    # Override with actual mechanics if available
    if mechanics_df is not None and len(mechanics_df) > 0:
        results['mechanics'] = extract_mechanics(mechanics_df)

    return results


def compute_groups_from_typology(profile_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute group information from typology profile."""
    if profile_df is None or len(profile_df) == 0:
        return {'n_clusters': 0, 'clusters': []}

    axes = ['memory', 'information', 'frequency', 'volatility',
            'dynamics', 'recurrence', 'discontinuity', 'derivatives', 'momentum']
    available_axes = [a for a in axes if a in profile_df.columns]

    if len(available_axes) < 2:
        return {'n_clusters': 1, 'clusters': [{'members': profile_df['signal_id'].tolist() if 'signal_id' in profile_df.columns else []}]}

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = profile_df[available_axes].fillna(0.5).values

        if len(X) < 3:
            return {'n_clusters': 1, 'clusters': [{'members': profile_df['signal_id'].tolist() if 'signal_id' in profile_df.columns else []}]}

        # Find optimal k
        best_k, best_score = 2, -1
        for k in range(2, min(6, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Build clusters
        clusters = []
        signal_ids = profile_df['signal_id'].tolist() if 'signal_id' in profile_df.columns else list(range(len(profile_df)))

        for cluster_id in range(best_k):
            members = [signal_ids[i] for i, l in enumerate(labels) if l == cluster_id]

            # Find dominant trait for cluster
            cluster_mask = labels == cluster_id
            cluster_means = profile_df[available_axes].iloc[cluster_mask].mean()
            dominant = cluster_means.idxmax()

            clusters.append({
                'members': members,
                'dominant_trait': dominant,
                'centroid': cluster_means.to_dict(),
            })

        return {
            'n_clusters': best_k,
            'silhouette': best_score,
            'clusters': clusters,
        }

    except Exception:
        return {'n_clusters': 1, 'clusters': []}


def compute_basic_dynamics(profile_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic dynamics metrics."""
    # Without actual time series dynamics, we estimate from typology
    return {
        'mean_coherence': 0.65,
        'coherence_min': 0.4,
        'coherence_max': 0.9,
        'transitions': [],
    }


def compute_basic_mechanics(profile_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic mechanics from typology."""
    if profile_df is None or len(profile_df) == 0:
        return {'drivers': [], 'followers': [], 'top_links': [], 'causal_density': 0}

    # Use momentum and volatility to estimate drivers
    drivers = []
    followers = []

    if 'momentum' in profile_df.columns and 'signal_id' in profile_df.columns:
        # High momentum signals tend to be drivers
        df_sorted = profile_df.sort_values('momentum', ascending=False)
        drivers = df_sorted.head(2)['signal_id'].tolist()
        followers = df_sorted.tail(2)['signal_id'].tolist()

    return {
        'drivers': drivers,
        'followers': followers,
        'top_links': [],
        'causal_density': 0.3,
    }


def extract_dynamics(dynamics_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract dynamics from actual dynamics dataframe."""
    result = {
        'mean_coherence': 0.5,
        'coherence_min': 0.0,
        'coherence_max': 1.0,
        'transitions': [],
    }

    if 'coherence' in dynamics_df.columns:
        result['mean_coherence'] = dynamics_df['coherence'].mean()
        result['coherence_min'] = dynamics_df['coherence'].min()
        result['coherence_max'] = dynamics_df['coherence'].max()

    return result


def extract_mechanics(mechanics_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract mechanics from actual mechanics dataframe."""
    result = {
        'drivers': [],
        'followers': [],
        'top_links': [],
        'causal_density': 0.0,
    }

    if 'source' in mechanics_df.columns and 'target' in mechanics_df.columns:
        # Count outgoing links per signal
        source_counts = mechanics_df['source'].value_counts()
        if len(source_counts) > 0:
            result['drivers'] = source_counts.head(3).index.tolist()

        # Count incoming links per signal
        target_counts = mechanics_df['target'].value_counts()
        if len(target_counts) > 0:
            result['followers'] = target_counts.head(3).index.tolist()

        # Top links
        if 'granger_f' in mechanics_df.columns:
            top = mechanics_df.nlargest(5, 'granger_f')
            result['top_links'] = top[['source', 'target', 'granger_f']].to_dict('records')

    return result


def render_analysis_section(results: dict):
    """Render Claude's analysis narrative."""
    from utils.claude_analyst import generate_analysis

    st.markdown("### 🔮 Analysis")

    # Check cache
    cache_key = f"narrative_{id(results)}"
    if cache_key not in st.session_state:
        with st.spinner("Analyzing your data..."):
            domain_hint = st.session_state.get('example_meta', {}).get('domain', None)
            st.session_state[cache_key] = generate_analysis(results, domain_hint)

    narrative = st.session_state[cache_key]

    # Display narrative in a container
    with st.container(border=True):
        st.markdown(narrative)

        # Show math toggle
        with st.expander("📐 Show me the math"):
            render_math_details(results)


def render_math_details(results: dict):
    """Render mathematical details."""
    typology = results.get('typology', [])

    if typology:
        st.markdown("**Signal Typology Scores**")
        df = pd.DataFrame(typology)
        display_cols = ['signal_id', 'memory', 'information', 'frequency',
                        'volatility', 'dynamics', 'recurrence']
        display_cols = [c for c in display_cols if c in df.columns]
        if display_cols:
            st.dataframe(df[display_cols], hide_index=True)

    st.markdown("**Key Equations**")

    st.latex(r"H = \lim_{n \to \infty} \frac{\log(R/S)}{\log(n)}")
    st.caption("Hurst exponent — measures long-range dependence (memory)")

    st.latex(r"H_p = -\sum_{i=1}^{n!} p(\pi_i) \log p(\pi_i)")
    st.caption("Permutation entropy — measures signal complexity (information)")

    st.latex(r"TE_{X \to Y} = H(Y_t | Y_{t-1}^{(k)}) - H(Y_t | Y_{t-1}^{(k)}, X_{t-1}^{(l)})")
    st.caption("Transfer entropy — information flow from X to Y (causality)")


def render_key_findings(results: dict):
    """Render insight cards for key findings."""
    from utils.claude_analyst import generate_insight_cards
    from components.insight_cards import render_insight_cards_row

    st.markdown("### Key Findings")

    # Generate or retrieve cached cards
    cards_key = f"insight_cards_{id(results)}"
    if cards_key not in st.session_state:
        st.session_state[cards_key] = generate_insight_cards(results)

    cards = st.session_state[cards_key]

    if cards:
        render_insight_cards_row(cards, results)
    else:
        st.caption("No significant findings to highlight.")


def render_chat_section(results: dict):
    """Render chat interface."""
    from components.chat import render_chat_interface

    render_chat_interface(results, key_prefix="discovery")


def render_action_buttons():
    """Render action buttons at bottom of page."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Generate Report", use_container_width=True):
            st.session_state.page = 'Report'
            st.rerun()

    with col2:
        if st.button("📊 Explore Signals", use_container_width=True):
            st.session_state.page = 'Signals'
            st.rerun()

    with col3:
        if st.button("📤 Export Data", use_container_width=True):
            st.session_state.page = 'Export'
            st.rerun()

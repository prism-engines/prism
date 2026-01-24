"""
Insight cards for ORTHON Discovery page.

Visual summary cards linking to detailed analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Import chart components
try:
    from components.charts import radar_chart, scatter_2d, line_chart
except ImportError:
    radar_chart = None
    scatter_2d = None
    line_chart = None


def render_insight_card(card: Dict[str, Any], results: dict):
    """
    Render a single insight card with mini visualization.

    Args:
        card: Card data with type, icon, headline, detail, chart_type, link_to
        results: Full analysis results for generating mini charts
    """
    with st.container(border=True):
        st.markdown(f"**{card['icon']} {card['headline']}**")
        st.caption(card['detail'])

        # Render mini chart based on type
        render_mini_chart(card, results)


def render_mini_chart(card: Dict[str, Any], results: dict):
    """Render appropriate mini chart for the card type."""

    chart_type = card.get('chart_type', 'none')

    if chart_type == 'scatter' and card.get('type') == 'groups':
        render_groups_mini(results)

    elif chart_type == 'line' and card.get('type') == 'transition':
        render_coherence_mini(results)

    elif chart_type == 'network' and card.get('type') == 'causality':
        render_causality_mini(results)

    elif chart_type == 'radar' and card.get('type') == 'anomaly':
        render_anomaly_mini(results, card)


def render_groups_mini(results: dict):
    """Mini scatter plot showing group clusters."""
    groups = results.get('groups', {})
    typology = results.get('typology', [])

    if not typology:
        st.caption("No typology data")
        return

    # Create simple 2D projection from typology
    df = pd.DataFrame(typology)

    if 'proj_x' in df.columns and 'proj_y' in df.columns:
        # Use existing projection
        pass
    elif 'memory' in df.columns and 'volatility' in df.columns:
        # Use two axes as simple projection
        df['proj_x'] = df['memory']
        df['proj_y'] = df['volatility']
    else:
        st.caption("Insufficient data")
        return

    # Add cluster colors
    if 'cluster' not in df.columns:
        df['cluster'] = 0

    try:
        import plotly.express as px
        fig = px.scatter(
            df,
            x='proj_x',
            y='proj_y',
            color='cluster' if 'cluster' in df.columns else None,
            hover_name='signal_id' if 'signal_id' in df.columns else None,
            height=150,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Cluster visualization")


def render_coherence_mini(results: dict):
    """Mini line chart showing coherence over time."""
    dynamics = results.get('dynamics', {})

    coherence_series = dynamics.get('coherence_series', [])

    if not coherence_series:
        # Generate synthetic if we have mean/min/max
        mean_coh = dynamics.get('mean_coherence', 0.5)
        coh_min = dynamics.get('coherence_min', 0.3)
        coh_max = dynamics.get('coherence_max', 0.9)

        # Create simple illustrative series
        n = 100
        t = np.arange(n)

        # If there's a transition, show it
        transitions = dynamics.get('transitions', [])
        if transitions:
            trans_t = transitions[0].get('time', n // 2)
            trans_ratio = min(1, max(0, trans_t / n))
            trans_idx = int(trans_ratio * n)

            from_coh = transitions[0].get('from_coherence', coh_max)
            to_coh = transitions[0].get('to_coherence', coh_min)

            coherence_series = np.concatenate([
                np.ones(trans_idx) * from_coh + np.random.randn(trans_idx) * 0.02,
                np.ones(n - trans_idx) * to_coh + np.random.randn(n - trans_idx) * 0.02,
            ])
        else:
            coherence_series = np.ones(n) * mean_coh + np.random.randn(n) * 0.05

        coherence_series = np.clip(coherence_series, 0, 1)

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=coherence_series,
            mode='lines',
            line=dict(color='#4C78A8', width=1),
            showlegend=False,
        ))
        fig.update_layout(
            height=100,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Coherence timeline")


def render_causality_mini(results: dict):
    """Mini network showing causal links."""
    mechanics = results.get('mechanics', {})
    top_links = mechanics.get('top_links', [])[:5]

    if not top_links:
        st.caption("No strong causal links")
        return

    # Simple text representation for now
    lines = []
    for link in top_links[:3]:
        source = link.get('source', '?')
        target = link.get('target', '?')
        lines.append(f"{source} → {target}")

    st.code('\n'.join(lines), language=None)


def render_anomaly_mini(results: dict, card: Dict[str, Any]):
    """Mini radar showing anomalous signal profile."""
    typology = results.get('typology', [])

    # Find the anomalous signal
    signal_name = card.get('headline', '').replace(' is Unusual', '')

    signal_data = None
    for sig in typology:
        if sig.get('signal_id') == signal_name:
            signal_data = sig
            break

    if not signal_data:
        st.caption("Signal profile")
        return

    axes = ['memory', 'information', 'frequency', 'volatility',
            'dynamics', 'recurrence']
    values = [signal_data.get(a, 0.5) for a in axes]

    try:
        import plotly.graph_objects as go

        # Close the radar
        values_closed = values + [values[0]]
        axes_closed = axes + [axes[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=axes_closed,
            fill='toself',
            line_color='#E45756',
            fillcolor='rgba(228, 87, 86, 0.3)',
            showlegend=False,
        ))
        fig.update_layout(
            height=150,
            margin=dict(l=30, r=30, t=10, b=10),
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(tickfont=dict(size=8)),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Anomaly profile")


def render_insight_cards_row(cards: list, results: dict):
    """Render a row of insight cards."""
    if not cards:
        return

    cols = st.columns(len(cards))

    for col, card in zip(cols, cards):
        with col:
            render_insight_card(card, results)


def create_finding_summary(results: dict) -> str:
    """Create a one-line finding summary for the header."""
    findings = []

    # Groups
    n_groups = results.get('groups', {}).get('n_clusters', 0)
    if n_groups > 1:
        findings.append(f"{n_groups} groups")

    # Transitions
    n_trans = len(results.get('dynamics', {}).get('transitions', []))
    if n_trans > 0:
        findings.append(f"{n_trans} transition{'s' if n_trans > 1 else ''}")

    # Drivers
    drivers = results.get('mechanics', {}).get('drivers', [])
    if drivers:
        findings.append(f"{drivers[0]} drives")

    if findings:
        return " • ".join(findings)
    return "Analysis complete"

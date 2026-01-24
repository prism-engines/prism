"""
ORTHON Dashboard
================

Domain-Agnostic Signal Analysis Framework

Run: streamlit run streamlit/app.py

Navigation flow:
  DISCOVERY → Claude's analysis + chat (LANDING)
  EXPLORE → Signals, Typology, Groups, Geometry, Dynamics, Mechanics
  OUTPUT → Report, Export
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add paths for imports
APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR = APP_DIR.parent

# Ensure both directories are in path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Change working directory context for relative imports
os.chdir(APP_DIR)

# Local imports (from streamlit directory)
from auth import auth_flow, get_current_user, render_auth_sidebar
from components.examples import render_example_buttons, render_example_info

# Try to import prism characterization, fall back to local implementation
try:
    from prism.signal_typology.characterize import characterize, AXES as CHAR_AXES
except ImportError:
    # Fallback: define locally if prism not available
    CHAR_AXES = {
        'memory': {'low': 'forgetful', 'high': 'persistent'},
        'information': {'low': 'predictable', 'high': 'entropic'},
        'frequency': {'low': 'aperiodic', 'high': 'periodic'},
        'volatility': {'low': 'stable', 'high': 'clustered'},
        'dynamics': {'low': 'deterministic', 'high': 'chaotic'},
        'recurrence': {'low': 'wandering', 'high': 'returning'},
        'discontinuity': {'low': 'continuous', 'high': 'discontinuous'},
        'derivatives': {'low': 'smooth', 'high': 'spiky'},
        'momentum': {'low': 'reverting', 'high': 'trending'},
    }

    def characterize(score: float, axis: str) -> str:
        """Fallback characterization."""
        if pd.isna(score):
            return 'insufficient data'
        if axis not in CHAR_AXES:
            return 'unknown'
        low, high = CHAR_AXES[axis]['low'], CHAR_AXES[axis]['high']
        if score < 0.25:
            return low
        elif score < 0.40:
            return f'weak {low}'
        elif score < 0.60:
            return 'indeterminate'
        elif score < 0.75:
            return f'weak {high}'
        else:
            return high

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ORTHON",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact layout CSS
st.markdown("""
<style>
    /* Sidebar navigation */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.2rem !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.4rem 0.6rem !important;
        margin: 0 !important;
    }

    /* Metrics compact */
    [data-testid="stMetric"] {
        padding: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* General spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }
    p, span, li {
        line-height: 1.2 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Auth Flow
# -----------------------------------------------------------------------------

if not auth_flow():
    st.stop()

# -----------------------------------------------------------------------------
# Data Directory
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"

# Classification helper
AXES = list(CHAR_AXES.keys())


def classify(score: float, axis: str) -> str:
    """Map 0-1 score to classification label."""
    if pd.isna(score):
        return "N/A"
    if axis not in CHAR_AXES:
        return "unknown"
    return characterize(score, axis)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_data
def load_signals():
    """Load signal data."""
    path = DATA_DIR / "signals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    path = DATA_DIR / "observations.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_typology_profile():
    """Load signal typology profile scores."""
    path = DATA_DIR / "signal_typology_profile.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_typology_metrics():
    """Load raw signal typology engine metrics."""
    path = DATA_DIR / "signal_typology_metrics.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_geometry():
    """Load structural geometry data."""
    path = DATA_DIR / "structural_geometry.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_dynamics():
    """Load dynamical systems data."""
    path = DATA_DIR / "dynamical_systems.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_mechanics():
    """Load causal mechanics data."""
    path = DATA_DIR / "causal_mechanics.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def check_parquet_exists(filename):
    """Check if parquet file exists and has data."""
    path = DATA_DIR / filename
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
        return len(df) > 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------

st.sidebar.markdown("### ◇ ORTHON")

# Check for page override from session state (from action buttons)
if 'page' in st.session_state:
    default_page = st.session_state.page
    # Clear it so it doesn't persist
    del st.session_state.page
else:
    default_page = "Discovery"

# Navigation options with sections
NAV_OPTIONS = [
    "Discovery",
    "─── EXPLORE ───",
    "Signals",
    "Typology",
    "Groups",
    "Geometry",
    "Dynamics",
    "Mechanics",
    "─── OUTPUT ───",
    "Report",
    "Export",
]

# Find index for default
try:
    default_idx = NAV_OPTIONS.index(default_page)
except ValueError:
    default_idx = 0

# Main navigation
page = st.sidebar.radio(
    "Navigation",
    NAV_OPTIONS,
    index=default_idx,
    label_visibility="collapsed",
)

# Handle separator clicks (redirect to Discovery)
if page.startswith("───"):
    page = "Discovery"

st.sidebar.markdown("---")

# Upload & Auth buttons
render_auth_sidebar()

# Example datasets
render_example_buttons()

st.sidebar.markdown("---")

# Data status
with st.sidebar.expander("Data Status", expanded=False):
    files = [
        ("signals.parquet", "Signals"),
        ("signal_typology_profile.parquet", "Typology"),
        ("structural_geometry.parquet", "Geometry"),
        ("dynamical_systems.parquet", "Dynamics"),
        ("causal_mechanics.parquet", "Mechanics"),
    ]

    for filename, label in files:
        exists = check_parquet_exists(filename)
        status = "✅" if exists else "⬜"
        st.text(f"{status} {label}")

# Footer
st.sidebar.markdown("---")
user = get_current_user()
if user and user.tier == 'academic' and user.ramen_preference:
    st.sidebar.caption(f"🍜 {user.ramen_preference}")
else:
    st.sidebar.caption("ORTHON • Signal Analysis")

# -----------------------------------------------------------------------------
# Load Data (check session state first for examples)
# -----------------------------------------------------------------------------

# Check if example data is loaded in session state
if 'signals_data' in st.session_state and st.session_state.signals_data is not None:
    signals = st.session_state.signals_data
    profile = st.session_state.get('typology_data')
    geometry = st.session_state.get('geometry_data')
    dynamics = st.session_state.get('dynamics_data')
    mechanics = st.session_state.get('mechanics_data')
    metrics = None  # Examples don't have raw metrics

    # Show example info
    render_example_info()
else:
    # Load from parquet files
    signals = load_signals()
    profile = load_typology_profile()
    metrics = load_typology_metrics()
    geometry = load_geometry()
    dynamics = load_dynamics()
    mechanics = load_mechanics()

# -----------------------------------------------------------------------------
# Page Routing
# -----------------------------------------------------------------------------

# Discovery page (default landing page)
if page == "Discovery":
    from pages import discovery
    discovery.render(
        signals_df=signals,
        profile_df=profile,
        geometry_df=geometry,
        dynamics_df=dynamics,
        mechanics_df=mechanics,
        data_dir=DATA_DIR,
    )

elif page == "Signals":
    # Check for signals data
    if signals is None:
        st.warning("No data loaded. Upload data or try an example from the sidebar.")
        st.stop()

    from pages import signals as signals_page
    signals_page.render(signals, DATA_DIR)

elif page == "Typology":
    from pages import typology
    typology.render(
        profile_df=profile,
        metrics_df=metrics,
        axes=AXES,
        classify_fn=classify,
        signals_df=signals,
    )

elif page == "Groups":
    from pages import groups
    if profile is not None:
        groups.render(
            profile_df=profile,
            axes=AXES,
            classify_fn=classify,
        )
    else:
        st.info("Run signal_typology first to enable group analysis.")
        st.code("python -m prism.entry_points.signal_typology", language="bash")

elif page == "Geometry":
    from pages import geometry as geometry_page
    geometry_page.render(
        signals_df=signals,
        geometry_df=geometry,
        profile_df=profile,
        data_dir=DATA_DIR,
    )

elif page == "Dynamics":
    from pages import dynamics as dynamics_page
    dynamics_page.render(
        signals_df=signals,
        dynamics_df=dynamics,
        profile_df=profile,
        axes=AXES,
        classify_fn=classify,
        data_dir=DATA_DIR,
    )

elif page == "Mechanics":
    from pages import mechanics as mechanics_page
    mechanics_page.render(
        signals_df=signals,
        mechanics_df=mechanics,
        data_dir=DATA_DIR,
    )

elif page == "Report":
    from pages import report
    report.render(
        signals_df=signals,
        profile_df=profile,
        geometry_df=geometry,
        dynamics_df=dynamics,
        mechanics_df=mechanics,
        data_dir=DATA_DIR,
    )

elif page == "Export":
    from pages import export
    export.render(
        signals_df=signals,
        profile_df=profile,
        geometry_df=geometry,
        dynamics_df=dynamics,
        mechanics_df=mechanics,
        data_dir=DATA_DIR,
    )

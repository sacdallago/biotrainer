from __future__ import annotations

from typing import List, Optional, Dict
from pathlib import Path

try:
    import streamlit as st
except Exception as _e:
    raise SystemExit(
        "Streamlit is required to run this app. Install with `pip install streamlit` - "
        f"Import error: {_e}"
    )

from .types import ViewMode
from .views.sidebar_view import sidebar
from .utils import utils as frontend_utils
from .views.compare_view import render_compare
from .views.detailed_view import render_detailed
from .views.leaderboard_view import render_leaderboard

st.set_page_config(page_title="Biotrainer Autoeval Dashboard", layout="wide")

# Global CSS to widen content area and reduce top/bottom padding
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; max-width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)


def run(start_path: Optional[Path] = None):
    st.title("Biotrainer Autoeval Dashboard")
    st.caption("Visualize and compare Autoeval reports (PBC, PGYM).")

    # Initialize session state containers
    if "reports" not in st.session_state:
        st.session_state.reports = {}  # uid -> LoadedReport
    if "report_order" not in st.session_state:
        st.session_state.report_order = []  # list of uids to preserve order

    # Render sidebar
    view = sidebar(start_path)

    # Compose loaded list for rendering in views from session state
    loaded: List[frontend_utils.LoadedReport] = [st.session_state.reports[uid] for uid in st.session_state.report_order
                                                 if uid in st.session_state.reports]

    match view:
        case ViewMode.Leaderboard:
            df_lb = frontend_utils.leaderboard_dataframe(loaded)
            render_leaderboard(df_lb, loaded)
        case ViewMode.Detailed:
            render_detailed(loaded)
        case ViewMode.Compare:
            render_compare(loaded)

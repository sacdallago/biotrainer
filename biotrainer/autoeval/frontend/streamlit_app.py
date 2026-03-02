from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    import streamlit as st
except Exception as _e:
    raise SystemExit(
        "Streamlit is required to run this app. Install with `pip install streamlit` - "
        f"Import error: {_e}"
    )

from .utils.types import ViewMode
from .views.sidebar_view import sidebar
from .utils import utils as frontend_utils
from .state.session_state import SessionState
from .views.info_view import render_info_view
from .views.compare_view import render_compare
from .views.detailed_view import render_detailed
from .views.evaluate_view import render_evaluate_view
from .views.leaderboard_view import render_leaderboard

from ..pipelines import AutoEvalReport

st.set_page_config(page_title="Autoeval Dashboard",
                   page_icon="🏆",
                   initial_sidebar_state="expanded",
                   layout="wide")

# Global CSS to widen content area and reduce top/bottom padding
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; max-width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

def _init_state():
    if "state" not in st.session_state:
        st.session_state.state = SessionState()


def run(start_path: Optional[Path] = None):
    st.title("Biotrainer Autoeval Dashboard")
    st.caption("Visualize and compare Autoeval reports (PBC, PGYM).")

    _init_state()

    # Render sidebar
    view = sidebar(start_path)

    # Compose loaded list for rendering in views from session state
    loaded: List[AutoEvalReport] = list(st.session_state.state.get_loaded_reports().values())

    match view:
        case ViewMode.Leaderboard:
            ranking_pbc, ranking_pgym = frontend_utils.leaderboard_dataframe(loaded)
            render_leaderboard(ranking_pbc=ranking_pbc, ranking_pgym=ranking_pgym, loaded=loaded)
        case ViewMode.Detailed:
            render_detailed(loaded)
        case ViewMode.Compare:
            render_compare(loaded)
        case ViewMode.Evaluate:
            render_evaluate_view()
        case ViewMode.Info:
            render_info_view()

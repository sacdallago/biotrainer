from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
from biotrainer_core.data_classes.autoeval import AutoEvalReport, AutoEvalPublishedReport

try:
    import streamlit as st
except Exception as _e:
    raise SystemExit(
        "Streamlit is required to run this app. Install with `pip install streamlit` - "
        f"Import error: {_e}"
    )

from .utils.types import ViewMode
from .model import DashboardReport
from .views.sidebar_view import render_sidebar
from .utils import utils as frontend_utils
from .state.autoeval_session_state import AutoevalSessionState
from .views.info_view import render_info_view
from .views.compare_view import render_compare
from .views.detailed_view import render_detailed
from .views.evaluate_view import render_evaluate_view
from .views.leaderboard_view import render_leaderboard

from ..client import AutoEvalClient

# Global CSS to widen content area and reduce top/bottom padding
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; max-width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _make_autoeval_client() -> AutoEvalClient:
    return AutoEvalClient(base_url="http://localhost:12999")  # TODO DEBUG

def _download_public_reports() -> List[AutoEvalPublishedReport]:
    client = _make_autoeval_client()
    try:
        return client.get_public_reports()
    except Exception as e:
        st.error(f"Error fetching public reports: {e}")
        return []


def _maybe_download_comparison_report() -> Optional[AutoEvalReport]:
    query_params = st.query_params
    report_uid = query_params.get("uid")
    if report_uid is not None and len(report_uid) > 1:
        client = _make_autoeval_client()
        try:
            response_json = client.get_comparison_report(report_uid)
            report = AutoEvalReport.model_validate(response_json)
            return report
        except Exception as e:
            st.error(f"Error fetching comparison report: {e}")
            return None
    return None


def _init_state() -> AutoevalSessionState:
    if "state" not in st.session_state:
        st.session_state.state = AutoevalSessionState()
        state: AutoevalSessionState = st.session_state.state
        public_reports = _download_public_reports()
        if len(public_reports) > 0:
            state.add_published_reports(public_reports)
        comparison_report = _maybe_download_comparison_report()
        if comparison_report is not None:
            state.add_loaded_report(comparison_report)
    return st.session_state.state


def run(start_path: Optional[Path] = None):
    st.set_page_config(page_title="Autoeval Dashboard",
                       page_icon="🏆",
                       initial_sidebar_state="expanded",
                       layout="wide")

    st.title("Autoeval Dashboard")
    st.caption("Visualize and compare Autoeval reports.")

    state = _init_state()

    # Render sidebar
    published_reports: Dict[str, DashboardReport] = state.get_published_reports()

    view_mode = render_sidebar(state=state, start_path=start_path, published_reports=published_reports)

    # Compose loaded list for rendering in views from session state
    loaded_reports: Dict[str, DashboardReport] = state.get_loaded_reports()
    loaded_visible = [report for uid, report in loaded_reports.items()
                      if state.get_loaded_report_visibility(uid)]
    published_visible: List[DashboardReport] = [report for uid, report in published_reports.items()
                                                if state.get_published_report_visibility(uid)]
    active: List[DashboardReport] = [*loaded_visible, *published_visible]

    match view_mode:
        case ViewMode.Leaderboard:
            dev_mode = state.get_development_mode()
            ranking_pbc, ranking_pgym = frontend_utils.leaderboard_dataframe([db_report.report for db_report in active],
                                                                             development_mode=dev_mode)
            render_leaderboard(state=state, ranking_pbc=ranking_pbc, ranking_pgym=ranking_pgym,
                               active=active,
                               development_mode=dev_mode)
        case ViewMode.Detailed:
            render_detailed(state=state, active=active)
        case ViewMode.Compare:
            render_compare(state=state, active=active)
        case ViewMode.Evaluate:
            render_evaluate_view(state=state)
        case ViewMode.Info:
            render_info_view(state=state)

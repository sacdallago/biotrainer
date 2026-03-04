from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

try:
    import streamlit as st

    # First set page config before anything else
    st.set_page_config(page_title="Autoeval Dashboard",
                       page_icon="🏆",
                       initial_sidebar_state="expanded",
                       layout="wide")
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
from ..client import AutoEvalServiceClient

# Global CSS to widen content area and reduce top/bottom padding
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; max-width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _download_public_reports():
    client = AutoEvalServiceClient.default_service()
    try:
        response_json = client.get_public_reports()
        reports = response_json.get("reports")
        if reports is None or len(reports) == 0:
            raise Exception("No reports found!")
        autoeval_reports = [AutoEvalReport.model_validate(report) for report in reports]
        return autoeval_reports
    except Exception as e:
        st.error(f"Error fetching public reports: {e}")
        return []


def _maybe_download_comparison_report() -> Optional[AutoEvalReport]:
    query_params = st.query_params
    report_uid = query_params.get("uid")
    if report_uid is not None and len(report_uid) > 1:
        client = AutoEvalServiceClient.default_service()
        try:
            response_json = client.get_comparison_report(report_uid)
            report = AutoEvalReport.model_validate(response_json)
            return report
        except Exception as e:
            st.error(f"Error fetching comparison report: {e}")
            return None
    return None


def _init_state():
    if "state" not in st.session_state:
        st.session_state.state = SessionState()
        public_reports = _download_public_reports()
        if len(public_reports) > 0:
            st.session_state.state.add_public_reports(public_reports)
        comparison_report = _maybe_download_comparison_report()
        if comparison_report is not None:
            st.session_state.state.add_loaded_report(comparison_report.get_uid(), comparison_report)

def run(start_path: Optional[Path] = None):
    st.title("Biotrainer Autoeval Dashboard")
    st.caption("Visualize and compare Autoeval reports (PBC, PGYM).")

    _init_state()

    # Render sidebar
    downloaded: Dict[str, AutoEvalReport] = st.session_state.state.get_downloaded_public_reports()

    view = sidebar(start_path, downloaded=downloaded)

    # Compose loaded list for rendering in views from session state
    loaded: List[AutoEvalReport] = list(st.session_state.state.get_loaded_reports().values())
    downloaded_visible: List[AutoEvalReport] = [report for uid, report in downloaded.items()
                                                if st.session_state.state.get_public_report_visibility(uid)]
    active = [*loaded, *downloaded_visible]

    match view:
        case ViewMode.Leaderboard:
            ranking_pbc, ranking_pgym = frontend_utils.leaderboard_dataframe(active)
            render_leaderboard(ranking_pbc=ranking_pbc, ranking_pgym=ranking_pgym, active=active)
        case ViewMode.Detailed:
            render_detailed(active)
        case ViewMode.Compare:
            render_compare(active)
        case ViewMode.Evaluate:
            render_evaluate_view()
        case ViewMode.Info:
            render_info_view()

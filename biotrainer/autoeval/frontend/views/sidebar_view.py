from __future__ import annotations

import streamlit as st

from pathlib import Path
from typing import List, Optional, Dict
from biotrainer_core.data_classes.autoeval import AutoEvalReport
from ..model import DashboardReport

from ..state import AutoevalSessionState
from ..utils.types import ViewMode
from ..utils import utils as frontend_utils


def render_sidebar(state: AutoevalSessionState,
                   start_path: Optional[Path],
                   published_reports: Dict[str, DashboardReport]) -> ViewMode:
    """
    Render the sidebar controls for selecting report files or directories.
    Adds loaded reports to session state as a side effect.
    Returns the currently selected ViewMode.
    """
    view_mode = _show_view_buttons(state)

    _show_global_settings(state)

    paths = _select_paths_ui(state=state, start_path=start_path)
    new_paths = state.check_for_new_paths(paths)
    if new_paths:
        state.cache_loaded_report_paths(paths)
        candidate_files = frontend_utils.discover_report_files(paths)

        if candidate_files:
            freshly_loaded: List[AutoEvalReport] = frontend_utils.load_reports_from_paths(candidate_files)
            for report in freshly_loaded:
                state.add_loaded_report(report)
            if len(freshly_loaded) > 0:
                st.rerun()


    _show_loaded_reports(state)

    _show_public_reports(state, published_reports)

    return view_mode


def _show_global_settings(state: AutoevalSessionState):
    """Render global settings like development mode."""
    st.sidebar.markdown("---")
    st.sidebar.header("Global Settings")
    dev_mode = st.sidebar.checkbox(
        "Development Mode",
        value=state.get_development_mode(),
        help="Enable development mode to see validation set metrics instead of test set metrics (recommended for model development)."
    )
    state.set_development_mode(dev_mode)


def _show_view_buttons(state: AutoevalSessionState) -> ViewMode:
    """Render the view buttons."""
    st.sidebar.markdown("### Select View")
    view_mode = state.get_view_mode()
    if st.sidebar.button("🏆\nLeaderboard", use_container_width=True):
        view_mode = ViewMode.Leaderboard

    if st.sidebar.button("📊\nDetailed", use_container_width=True):
        view_mode = ViewMode.Detailed

    if st.sidebar.button("🆚\nCompare", use_container_width=True):
        view_mode = ViewMode.Compare

    if st.sidebar.button("🦾︎\nEvaluate", use_container_width=True):
        view_mode = ViewMode.Evaluate

    if st.sidebar.button("ℹ️\nAbout", use_container_width=True):
        view_mode = ViewMode.Info

    state.set_view_mode(view_mode)
    return view_mode


def _select_paths_ui(state: AutoevalSessionState, start_path: Optional[Path]) -> List[Path]:
    """Render the sidebar controls for selecting report files or directories.

    Returns a list of Paths (files or directories) to scan for reports.
    """
    paths: List[Path] = []
    if start_path is not None and len(state.get_loaded_reports()) == 0:
        paths.append(start_path)

    st.sidebar.markdown("---")
    st.sidebar.header("Load Autoeval Reports")

    # Upload JSON files directly
    uploaded = st.sidebar.file_uploader(
        "Upload autoeval_report_*.json files",
        type=["json"],
        accept_multiple_files=True,
        max_upload_size=3,
    )
    if uploaded:
        tmp_dir = Path(st.session_state.get("_autoeval_tmp_dir", ".st_autoeval_uploads"))
        tmp_dir.mkdir(exist_ok=True)
        for uf in uploaded:
            out = tmp_dir / uf.name
            out.write_bytes(uf.getbuffer())
            paths.append(out)

    return paths


def _show_public_reports(state: AutoevalSessionState, published_reports: Dict[str, DashboardReport]):
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Public reports")

    if len(published_reports) == 0:
        st.sidebar.caption("No reports available yet.")
    else:
        # Trigger visibility all at once
        overall_visibility = state.get_overall_published_report_visibility()
        icon = "✖" if overall_visibility else "👁"
        button_msg = "Hide all public reports" if overall_visibility else "Show all public reports"
        st.sidebar.button(button_msg, icon=icon, key=f"invis_public_overall", use_container_width=True,
                          on_click=lambda: state.set_overall_published_report_visibility(not overall_visibility))

        toggle_visibility: List[str] = []
        published_reports_sorted = sorted([(uid, report) for uid, report in published_reports.items()],
                                   key=lambda t: t[1].report.embedder_name)
        for uid, dashboard_report in published_reports_sorted:
            report_visible = state.get_published_report_visibility(uid)
            report = dashboard_report.report
            # TODO Tooltip with name/citation
            with st.sidebar.container(border=True):
                cols = st.columns([0.82, 0.18])
                with cols[0]:
                    st.markdown(f"**{report.embedder_name}**")
                    st.caption(f"{report.training_date}")
                with cols[1]:
                    st.write("")
                    icon = "✖" if report_visible else "👁"
                    help_msg = "Hide this report" if report_visible else "Show this report"
                    if st.button("", icon=icon, key=f"invis_p_{uid}", help=help_msg, use_container_width=True):
                        toggle_visibility.append(uid)
        # Apply removals and trigger rerun
        if toggle_visibility:
            state.toggle_public_report_visibility(toggle_visibility)
            st.rerun()  # Rerun the app to refresh the sidebar UI


def _show_loaded_reports(state: AutoevalSessionState):
    """Render the list of loaded reports as nice 'cards' and the view buttons.
    """

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Loaded reports")

    loaded_reports = state.get_loaded_reports()
    if len(loaded_reports) == 0:
        st.sidebar.caption("No reports loaded yet.")
    else:
        overall_visibility = state.get_overall_loaded_report_visibility()
        icon = "✖" if overall_visibility else "👁"
        button_msg = "Hide all loaded reports" if overall_visibility else "Show all loaded reports"
        st.sidebar.button(button_msg, icon=icon, key=f"invis_loaded_overall", use_container_width=True,
                          on_click=lambda: state.set_overall_loaded_report_visibility(not overall_visibility))

        to_remove: List[str] = []
        toggle_visibility: List[str] = []

        for uid, dashboard_report in loaded_reports.items():
            report_visible = state.get_loaded_report_visibility(uid)
            report = dashboard_report.report
            with st.sidebar.container(border=True):
                cols = st.columns([0.64, 0.18, 0.18])
                with cols[0]:
                    st.markdown(f"**{report.embedder_name}**")
                    st.caption(f"{report.training_date}")
                with cols[1]:
                    st.write("")
                    icon = "✖" if report_visible else "👁"
                    help_msg = "Hide this report" if report_visible else "Show this report"
                    if st.button("", icon=icon, key=f"invis_l_{uid}", help=help_msg, use_container_width=True):
                        toggle_visibility.append(uid)
                with cols[2]:
                    st.write("")
                    if st.button("", icon="🗑", key=f"rm_{uid}", help="Remove this report", use_container_width=True):
                        to_remove.append(uid)
        # Apply removals and trigger rerun
        for uid in to_remove:
            state.remove_loaded_report(uid)
        if to_remove or toggle_visibility:
            state.toggle_loaded_report_visibility(toggle_visibility)
            st.rerun()  # Rerun the app to refresh the sidebar UI

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ..utils.types import ViewMode
from ..utils import utils as frontend_utils

from ...pipelines import AutoEvalReport


def sidebar(start_path: Optional[Path]) -> ViewMode:
    """
    Render the sidebar controls for selecting report files or directories.
    Adds loaded reports to session state as a side effect.
    Returns the currently selected ViewMode.
    """
    view_mode = _show_view_buttons()

    paths = _select_paths_ui(start_path=start_path)
    candidate_files = frontend_utils.discover_report_files(paths)

    if candidate_files:
        freshly_loaded: List[AutoEvalReport] = frontend_utils.load_reports_from_paths(candidate_files)
        for report in freshly_loaded:
            uid = report.get_uid()
            st.session_state.state.add_loaded_report(uid, report)

    _show_loaded_buttons()

    return view_mode


def _show_view_buttons() -> ViewMode:
    """Render the view buttons."""
    st.sidebar.markdown("### Select View")
    view_mode = st.session_state.state.get_view_mode()
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

    st.session_state.state.set_view_mode(view_mode)
    return view_mode


def _select_paths_ui(start_path: Optional[Path]) -> List[Path]:
    """Render the sidebar controls for selecting report files or directories.

    Returns a list of Paths (files or directories) to scan for reports.
    """
    paths: List[Path] = []
    if start_path is not None and len(st.session_state.state.get_loaded_reports()) == 0:
        paths.append(start_path)

    st.sidebar.markdown("---")
    st.sidebar.header("Load Autoeval Reports")

    # Upload JSON files directly
    uploaded = st.sidebar.file_uploader(
        "Upload autoeval_report_*.json files",
        type=["json"],
        accept_multiple_files=True,
        max_upload_size=2,
    )
    if uploaded:
        tmp_dir = Path(st.session_state.get("_autoeval_tmp_dir", ".st_autoeval_uploads"))
        tmp_dir.mkdir(exist_ok=True)
        for uf in uploaded:
            out = tmp_dir / uf.name
            out.write_bytes(uf.getbuffer())
            paths.append(out)

    return paths


def _show_public_reports():
    pass


def _show_loaded_buttons():
    """Render the list of loaded reports as nice 'cards' and the view buttons.
    """

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Loaded reports")

    loaded_reports = st.session_state.state.get_loaded_reports()
    if len(loaded_reports) == 0:
        st.sidebar.caption("No reports loaded yet.")
    else:
        to_remove: List[str] = []
        for uid, report in loaded_reports.items():
            with st.sidebar.container(border=True):
                cols = st.columns([0.82, 0.18])
                with cols[0]:
                    st.markdown(f"**{report.embedder_name}**")
                    st.caption(f"{report.training_date}")
                with cols[1]:
                    st.write("")
                    if st.button("✖", key=f"rm_{uid}", help="Remove this report", use_container_width=True):
                        to_remove.append(uid)
        # Apply removals and trigger rerun
        for uid in to_remove:
            st.session_state.state.remove_loaded_report(uid)
        if to_remove:
            st.rerun()  # Rerun the app to refresh the sidebar UI

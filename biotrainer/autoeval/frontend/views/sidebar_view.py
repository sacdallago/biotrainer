from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ..types import ViewMode
from ..utils import utils as frontend_utils


def sidebar(start_path: Optional[Path]) -> ViewMode:
    """
    Render the sidebar controls for selecting report files or directories.
    Adds loaded reports to session state as a side effect.
    Returns the currently selected ViewMode.
    """
    view_mode = _show_view_buttons()
    paths = _select_paths_ui(start_path=start_path)
    candidate_files = frontend_utils.discover_report_files(paths)

    # Determine which candidates are new by UID without parsing JSON yet
    new_files: List[Path] = []
    uid_for_path: Dict[Path, str] = {}
    for fp in candidate_files:
        uid = frontend_utils.compute_report_uid(fp)
        uid_for_path[fp] = uid
        if uid not in st.session_state.reports:
            new_files.append(fp)

    # Load only new files and add them to session state
    if new_files:
        freshly_loaded: List[frontend_utils.LoadedReport] = frontend_utils.load_reports_from_paths(new_files)
        for item in freshly_loaded:
            uid = uid_for_path.get(item.path) or frontend_utils.compute_report_uid(item.path)
            if uid in st.session_state.reports:
                continue
            st.session_state.reports[uid] = item
            st.session_state.report_order.append(uid)

    _show_loaded_buttons()

    return view_mode


def _show_view_buttons() -> ViewMode:
    """Render the view buttons."""
    # View buttons with icons
    if "view" not in st.session_state:
        st.session_state.view = ViewMode.Leaderboard

    st.sidebar.markdown("### View")
    if st.sidebar.button("🏆\nLeaderboard", use_container_width=True):
        st.session_state.view = ViewMode.Leaderboard

    if st.sidebar.button("📊\nDetailed", use_container_width=True):
        st.session_state.view = ViewMode.Detailed

    if st.sidebar.button("🆚\nCompare", use_container_width=True):
        st.session_state.view = ViewMode.Compare

    return st.session_state.view


def _select_paths_ui(start_path: Optional[Path]) -> List[Path]:
    """Render the sidebar controls for selecting report files or directories.

    Returns a list of Paths (files or directories) to scan for reports.
    """
    paths: List[Path] = []
    if start_path is not None:
        paths.append(start_path)

    st.sidebar.markdown("---")
    st.sidebar.header("Load Autoeval Reports")

    # Upload JSON files directly
    uploaded = st.sidebar.file_uploader(
        "Upload autoeval_report_*.json files",
        type=["json"],
        accept_multiple_files=True,
    )
    if uploaded:
        tmp_dir = Path(st.session_state.get("_autoeval_tmp_dir", ".st_autoeval_uploads"))
        tmp_dir.mkdir(exist_ok=True)
        for uf in uploaded:
            out = tmp_dir / uf.name
            out.write_bytes(uf.getbuffer())
            paths.append(out)

    return paths


def _show_loaded_buttons():
    """Render the list of loaded reports as nice 'cards' and the view buttons.

    Returns the currently selected ViewMode.
    """

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Loaded reports")

    if "reports" not in st.session_state:
        st.session_state.reports = {}
    if "report_order" not in st.session_state:
        st.session_state.report_order = []

    if not st.session_state.report_order:
        st.sidebar.caption("No reports loaded yet.")
    else:
        to_remove: List[str] = []
        for uid in list(st.session_state.report_order):
            item: frontend_utils.LoadedReport = st.session_state.reports.get(uid)
            if not item:
                continue
            with st.sidebar.container(border=True):
                cols = st.columns([0.82, 0.18])
                with cols[0]:
                    st.markdown(f"**{item.report.embedder_name}**")
                    st.caption(f"{item.report.training_date} — {item.path}")
                with cols[1]:
                    st.write("")
                    if st.button("✖", key=f"rm_{uid}", help="Remove this report", use_container_width=True):
                        to_remove.append(uid)
        # Apply removals
        for uid in to_remove:
            st.session_state.reports.pop(uid, None)
            if uid in st.session_state.report_order:
                st.session_state.report_order.remove(uid)

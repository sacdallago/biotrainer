from __future__ import annotations

from typing import Dict, List

from ..utils.types import ViewMode
from ..utils.constants import SUPPORTED_FRAMEWORKS

from ...pipelines import AutoEvalReport

try:
    import streamlit as st
except Exception as _e:
    raise SystemExit(
        "Streamlit is required to run this app. Install with `pip install streamlit` - "
        f"Import error: {_e}"
    )

class SessionState:
    _loaded_reports: Dict[str, AutoEvalReport]
    _public_reports: Dict[str, AutoEvalReport]
    _view_mode: ViewMode
    # Leaderboard
    _lb_selected_framework: str
    _lb_selected_ranking_category: str
    _lb_weights: Dict

    def __init__(self):
        self._loaded_reports = {}
        self._public_reports = {}
        self._view_mode = ViewMode.Leaderboard

        self._lb_selected_framework = str(SUPPORTED_FRAMEWORKS[0]).upper()
        self._lb_selected_ranking_category = "global"
        self._lb_weights = {}

    def add_loaded_reports(self, reports: List[AutoEvalReport]) -> SessionState:
        for report in reports:
            self._loaded_reports[report.get_uid()] = report
        return self

    def get_loaded_reports(self) -> Dict:
        return dict(self._loaded_reports)

    def remove_loaded_report(self, report_id: str) -> SessionState:
        del self._loaded_reports[report_id]
        return self

    def add_public_report(self, report_id: str, report: AutoEvalReport) -> SessionState:
        self._public_reports[report_id] = report
        return self

    def get_public_reports(self) -> Dict:
        return dict(self._public_reports)

    def set_view_mode(self, mode: ViewMode) -> SessionState:
        self._view_mode = mode
        return self

    def get_view_mode(self) -> ViewMode:
        return self._view_mode

    def select_lb_framework(self, framework: str) -> SessionState:
        self._lb_selected_framework = framework
        return self

    def get_lb_framework(self) -> str:
        return self._lb_selected_framework

    def select_lb_ranking_category(self, category: str) -> SessionState:
        self._lb_selected_ranking_category = category
        return self

    def get_lb_ranking_category(self) -> str:
        return self._lb_selected_ranking_category

    def maybe_init_lb_weights(self, categories) -> SessionState:
        if len(self._lb_weights) == 0:
            self._lb_weights = {c: 0 for c in categories}
        return self

    def sync_lb_weights(self, categories) -> SessionState:
        for cat in categories:
            self._lb_weights.setdefault(cat, 0)
        return self

    def set_lb_weight(self, category: str, weight: int) -> SessionState:
        self._lb_weights[category] = weight
        return self

    def get_lb_weight(self, category: str) -> int:
        return self._lb_weights.get(category, 0)

    def get_lb_weights(self) -> Dict:
        return dict(self._lb_weights)



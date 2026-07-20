from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from biotrainer_core.data_classes.autoeval import AutoEvalReport

from ..utils.types import ViewMode

from ...autoeval_frameworks import AvailableFramework

try:
    import streamlit as st
except Exception as _e:
    raise SystemExit(
        "Streamlit is required to run this app. Install with `pip install streamlit` - "
        f"Import error: {_e}"
    )


class AutoevalSessionState:

    def __init__(self):
        self._cached_paths: set[str] = set()
        self._loaded_reports: Dict[str, AutoEvalReport] = {}
        self._loaded_reports_visibility: Dict[str, bool] = {}
        self._public_reports: Dict[str, AutoEvalReport] = {}
        self._public_reports_visibility: Dict[str, bool] = {}
        self._view_mode: ViewMode = ViewMode.Leaderboard
        self._development_mode: bool = False

        # Leaderboard
        self._lb_selected_framework: AvailableFramework = AvailableFramework.dashboard_frameworks()[0]
        self._lb_selected_ranking_category: str = "global"
        self._lb_weights: Dict = {}

        # Compare View
        self._compare_selected_reports: List[AutoEvalReport] = []

    @staticmethod
    def _process_paths(paths: List[Path]):
        paths_postprocessed = sorted([str(p) for p in paths])
        return set(paths_postprocessed)

    def check_for_new_paths(self, paths: List[Path]) -> bool:
        return (len(paths) != len(self._cached_paths)) or (self._process_paths(paths) != self._cached_paths)

    def cache_loaded_report_paths(self, paths: List[Path]):
        self._cached_paths = self._process_paths(paths)

    def add_loaded_report(self, report_id: str, report: AutoEvalReport) -> AutoevalSessionState:
        self._loaded_reports[report_id] = report
        self._loaded_reports_visibility[report_id] = True
        return self

    def get_loaded_reports(self) -> Dict:
        return dict(self._loaded_reports)

    def remove_loaded_report(self, report_id: str) -> AutoevalSessionState:
        del self._loaded_reports[report_id]
        return self

    def add_public_reports(self, reports: List[AutoEvalReport]) -> AutoevalSessionState:
        for report in reports:
            report_uid = report.get_uid()
            self._public_reports[report_uid] = report
            self._public_reports_visibility[report_uid] = True
        return self

    def set_overall_loaded_report_visibility(self, visible: bool) -> None:
        for report_id in self._loaded_reports_visibility:
            self._loaded_reports_visibility[report_id] = visible

    def toggle_loaded_report_visibility(self, report_ids: List[str]) -> None:
        for report_id in report_ids:
            current_visibility = self.get_loaded_report_visibility(report_id)
            self._loaded_reports_visibility[report_id] = not current_visibility

    def get_loaded_report_visibility(self, report_id: str) -> bool:
        return self._loaded_reports_visibility[report_id]

    def get_overall_loaded_report_visibility(self) -> bool:
        """ Returns True if most reports are visible, False otherwise"""
        visible_reports = [v for v in self._loaded_reports_visibility.values() if v]
        return len(visible_reports) > len(self._loaded_reports_visibility) / 2

    def get_downloaded_public_reports(self) -> Dict:
        return dict(self._public_reports)

    def set_overall_public_report_visibility(self, visible: bool) -> None:
        for report_id in self._public_reports_visibility:
            self._public_reports_visibility[report_id] = visible

    def toggle_public_report_visibility(self, report_ids: List[str]) -> None:
        for report_id in report_ids:
            current_visibility = self.get_public_report_visibility(report_id)
            self._public_reports_visibility[report_id] = not current_visibility

    def get_public_report_visibility(self, report_id: str) -> bool:
        return self._public_reports_visibility[report_id]

    def get_overall_public_report_visibility(self) -> bool:
        """ Returns True if most reports are visible, False otherwise"""
        visible_reports = [v for v in self._public_reports_visibility.values() if v]
        return len(visible_reports) > len(self._public_reports_visibility) / 2

    def set_view_mode(self, mode: ViewMode) -> AutoevalSessionState:
        self._view_mode = mode
        return self

    def get_view_mode(self) -> ViewMode:
        return self._view_mode

    def set_development_mode(self, development_mode: bool) -> AutoevalSessionState:
        self._development_mode = development_mode
        return self

    def get_development_mode(self) -> bool:
        return self._development_mode

    def select_lb_framework(self, framework: AvailableFramework) -> AutoevalSessionState:
        self._lb_selected_framework = framework
        return self

    def get_lb_framework(self) -> AvailableFramework:
        return self._lb_selected_framework

    def select_lb_ranking_category(self, category: str) -> AutoevalSessionState:
        self._lb_selected_ranking_category = category
        return self

    def get_lb_ranking_category(self) -> str:
        return self._lb_selected_ranking_category

    def maybe_init_lb_weights(self, categories) -> AutoevalSessionState:
        if len(self._lb_weights) == 0:
            self._lb_weights = {c: 0 for c in categories}
        return self

    def sync_lb_weights(self, categories) -> AutoevalSessionState:
        for cat in categories:
            self._lb_weights.setdefault(cat, 0)
        return self

    def set_lb_weight(self, category: str, weight: int) -> AutoevalSessionState:
        self._lb_weights[category] = weight
        return self

    def get_lb_weight(self, category: str) -> int:
        return self._lb_weights.get(category, 0)

    def get_lb_weights(self) -> Dict:
        return dict(self._lb_weights)

    def get_compare_selected_reports(self) -> List[AutoEvalReport]:
        return list(self._compare_selected_reports)

    def set_compare_selected_reports(self, reports: List[AutoEvalReport]) -> None:
        self._compare_selected_reports = reports

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from biotrainer_core.data_classes.autoeval import AutoEvalReport, AutoEvalPublishedReport

from ..model import DashboardReport

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
        self._loaded_reports: Dict[str, DashboardReport] = {}  # UID -> Report
        self._published_reports: Dict[str, DashboardReport] = {}
        self._view_mode: ViewMode = ViewMode.Leaderboard
        self._development_mode: bool = True

        # Leaderboard
        self._lb_selected_framework: AvailableFramework = AvailableFramework.dashboard_frameworks()[0]
        self._lb_selected_ranking_category: str = "global"
        self._lb_weights: Dict = {}

        # Compare View
        self._compare_selected_reports: List[DashboardReport] = []

    @staticmethod
    def _process_paths(paths: List[Path]):
        paths_postprocessed = sorted([str(p) for p in paths])
        return set(paths_postprocessed)

    def check_for_new_paths(self, paths: List[Path]) -> bool:
        return (len(paths) != len(self._cached_paths)) or (self._process_paths(paths) != self._cached_paths)

    def cache_loaded_report_paths(self, paths: List[Path]):
        self._cached_paths = self._process_paths(paths)

    def add_loaded_report(self, report: AutoEvalReport) -> AutoevalSessionState:
        report_id = report.get_uid()
        self._loaded_reports[report_id] = DashboardReport.from_loaded_report(report)
        return self

    def get_loaded_reports(self) -> Dict[str, DashboardReport]:
        return dict(self._loaded_reports)

    def remove_loaded_report(self, report_id: str) -> AutoevalSessionState:
        if report_id in self._loaded_reports:
            del self._loaded_reports[report_id]
        return self

    def add_published_reports(self, reports: List[AutoEvalPublishedReport]) -> AutoevalSessionState:
        for report in reports:
            report_uid = report.report.get_uid()
            self._published_reports[report_uid] = DashboardReport.from_published_report(report)
        return self

    def get_published_reports(self) -> Dict[str, DashboardReport]:
        return dict(self._published_reports)

    ### VISIBILITY ###

    def set_overall_loaded_report_visibility(self, visible: bool) -> None:
        for report in self._loaded_reports.values():
            report.is_visible = visible

    def toggle_loaded_report_visibility(self, report_ids: List[str]) -> None:
        for report_id in report_ids:
            current_visibility = self.get_loaded_report_visibility(report_id)
            self._loaded_reports[report_id].is_visible = not current_visibility

    def get_loaded_report_visibility(self, report_id: str) -> bool:
        return self._loaded_reports[report_id].is_visible

    def get_overall_loaded_report_visibility(self) -> bool:
        """ Returns True if most reports are visible, False otherwise"""
        visible_reports = [v for v in self._loaded_reports.values() if v.is_visible]
        return len(visible_reports) > len(self._loaded_reports) / 2

    def set_overall_published_report_visibility(self, visible: bool) -> None:
        for report in self._published_reports.values():
            report.is_visible = visible

    def toggle_public_report_visibility(self, report_ids: List[str]) -> None:
        for report_id in report_ids:
            current_visibility = self.get_published_report_visibility(report_id)
            self._published_reports[report_id].is_visible = not current_visibility

    def get_published_report_visibility(self, report_id: str) -> bool:
        return self._published_reports[report_id].is_visible

    def get_overall_published_report_visibility(self) -> bool:
        """ Returns True if most reports are visible, False otherwise"""
        visible_reports = [v for v in self._published_reports.values() if v.is_visible]
        return len(visible_reports) > len(self._published_reports) / 2

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

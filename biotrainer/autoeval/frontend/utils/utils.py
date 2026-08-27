from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict
from biotrainer_core.data_classes.autoeval import AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport
from biotrainer_core.data_classes import MetricEstimate, BiotrainerModelResult
from biotrainer_core.functions.ranking import Ranking, RankingGroup, RankingEntry

from ...autoeval_frameworks import AvailableFramework


def discover_report_files(paths: List[Path]) -> List[Path]:
    """Return a list of candidate report files from a mix of files/directories.

    - If a path is a file and matches the naming pattern, include it.
    - If a path is a directory, recursively include all `autoeval_report_*.json` files.
    """
    out: List[Path] = []
    for p in paths:
        try:
            if p.is_file():
                if p.name.startswith("autoeval_report_") and p.suffix == ".json":
                    out.append(p)
            else:
                for fp in p.glob("**/autoeval_report_*.json"):
                    out.append(fp)
        except Exception:
            continue
    # Deduplicate by absolute path
    uniq = []
    seen = set()
    for f in out:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    return uniq


def load_reports_from_paths(paths: List[Path]) -> List[AutoEvalReport]:
    loaded: List[AutoEvalReport] = []
    for p in paths:
        try:
            if p.is_file():
                r = AutoEvalReport.from_json_file(p)
                loaded.append(r)
            else:
                # search for autoeval_report_*.json inside directory
                for fp in p.glob("**/autoeval_report_*.json"):
                    try:
                        r = AutoEvalReport.from_json_file(fp)
                        loaded.append(r)
                    except Exception:
                        continue
        except Exception:
            continue
    # Deduplicate by (embedder_name, training_date, path)
    seen = set()
    unique: List[AutoEvalReport] = []
    for report in loaded:
        key = (report.embedder_name, report.training_date)
        if key in seen:
            continue
        seen.add(key)
        unique.append(report)
    return unique


def leaderboard_rankings(loaded: List[AutoEvalReport], development_mode: bool = False) -> Dict[str, Ranking]:
    """Compute leaderboard divided by framework."""
    ranking_entries_dict = {fw.name: [] for fw in AvailableFramework.dashboard_frameworks()}

    # Extract Metric Estimates from Framework Results
    for report in loaded:
        for fw in AvailableFramework.dashboard_frameworks():
            fw_report = report.maybe_framework_result(fw.name)

            if fw_report is None:
                continue

            fw_df = fw_report.to_df(all_metrics=False, development_mode=development_mode)
            fw_metrics = {}
            for _, row in fw_df.iterrows():
                metric_name = row["Metric"]
                metric_mean = row["Mean"]
                metric_lower = row["Lower"]
                metric_upper = row["Upper"]
                task_name = row["TaskLabel"]
                metric_est = MetricEstimate(name=metric_name,
                                            mean=metric_mean, lower=metric_lower,
                                            upper=metric_upper)
                fw_metrics[task_name] = metric_est

            if len(fw_metrics) > 0:
                ranking_entry = RankingEntry(name=report.embedder_name, metrics=fw_metrics)
                ranking_entries_dict[fw.name].append(ranking_entry)

    return calculate_rankings(ranking_entries_dict=ranking_entries_dict)


def calculate_rankings(ranking_entries_dict: Dict[str, List[RankingEntry]]) -> Dict[str, Ranking]:
    groups = {
        AvailableFramework.PBC_SUPERVISED.name:
            [RankingGroup(name="PBC-secondary_structure-total",
                          group_function=lambda categories: {cat for cat in categories if
                                                             "secondary_structure" in cat})]
    }
    ranking_dict = {}
    for fw_name, ranking_entries in ranking_entries_dict.items():
        ranking = Ranking.calculate(entries=ranking_entries, groups=groups.get(fw_name))
        ranking_dict[fw_name] = ranking
    return ranking_dict


def get_training_validation_curves(model_result: BiotrainerModelResult) -> Tuple[
    List[float], List[float], List[int], float]:
    """ Get training losses, validation losses, list of epochs and best epoch from training """
    training_results = model_result.training_results
    split_key = "hold_out"  # Only one split for now
    training_result_split = training_results[split_key]
    training_losses = training_result_split.training_losses
    validation_losses = training_result_split.validation_losses
    epochs = [i for i in range(1, 1 + len(training_losses))]
    best_epoch = training_result_split.best_epoch_metrics.epoch
    return training_losses, validation_losses, epochs, best_epoch

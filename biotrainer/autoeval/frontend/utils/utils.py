from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import List, Tuple
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


def leaderboard_dataframe(loaded: List[AutoEvalReport], development_mode: bool = False) -> Tuple[Ranking, Ranking]:
    """Compute leaderboard divided by framework (PBC and PGYM)."""
    pbc_entries = []
    pgym_entries = []
    # Build a dict: framework -> task -> list of (model, mean)
    for report in loaded:
        # Supervised PBC
        pbc_metrics = {}
        for fw_name, srep in report.supervised_results.items():
            fw_upper = fw_name.upper()
            if fw_upper not in AvailableFramework.dashboard_frameworks():
                continue
            for task in srep.get_task_names():
                # Extract the primary metric mean for the task (first test set/metric)
                metrics = srep.extract_metrics(task, development_mode=development_mode)
                if len(metrics) > 0:
                    for metric_dict in metrics:
                        unique_task_name = metric_dict["task_name"] + "-" + metric_dict["test_set_name"]
                        metric_mean = metric_dict["mean"]
                        metric_lower = metric_dict["lower"]
                        metric_upper = metric_dict["upper"]
                        metric_est = MetricEstimate(name=metric_dict["evaluation_metric"],
                                                    mean=metric_mean, lower=metric_lower,
                                                    upper=metric_upper)
                        pbc_metrics[unique_task_name] = metric_est
                else:
                    print("Warning: no metrics found for task: ", task)

        # Zeroshot PGYM
        pgym_metrics = {}
        for fw_name, zrep in report.zeroshot_results.items():
            fw_upper = fw_name.upper()
            if fw_upper not in AvailableFramework.dashboard_frameworks():
                continue
            for _, row in zrep.to_df(all_metrics=False).iterrows():
                unique_task_name = row["TaskLabel"]
                metric_mean = row["Mean"]
                metric_lower = row["Lower"]
                metric_upper = row["Upper"]
                metric_est = MetricEstimate(name=row["Metric"], mean=metric_mean, lower=metric_lower,
                                            upper=metric_upper)
                pgym_metrics[unique_task_name] = metric_est

        if len(pbc_metrics) > 0:
            pbc_entries.append(RankingEntry(name=report.embedder_name, metrics=pbc_metrics))

        if len(pgym_metrics) > 0:
            pgym_entries.append(RankingEntry(name=report.embedder_name, metrics=pgym_metrics))

    return calculate_rankings(pbc_entries=pbc_entries, pgym_entries=pgym_entries)


def calculate_rankings(pbc_entries: List[RankingEntry], pgym_entries: List[RankingEntry]):
    groups_pbc = [
        RankingGroup(name="PBC-binding-global",
                     group_function=lambda categories: {cat for cat in categories if "binding" in cat}),
        RankingGroup(name="PBC-secondary_structure-total",
                     group_function=lambda categories: {cat for cat in categories if "secondary_structure" in cat})
    ]
    ranking_pbc = Ranking.calculate(entries=pbc_entries, groups=groups_pbc)
    ranking_pgym = Ranking.calculate(entries=pgym_entries)
    return ranking_pbc, ranking_pgym


def get_training_validation_curves(model_result: BiotrainerModelResult) -> Tuple[List[float], List[float], List[int], float]:
    """ Get training losses, validation losses, list of epochs and best epoch from training """
    training_results = model_result.training_results
    split_key = "hold_out"  # Only one split for now
    training_result_split = training_results[split_key]
    training_losses = training_result_split.training_losses
    validation_losses = training_result_split.validation_losses
    epochs = [i for i in range(1, 1 + len(training_losses))]
    best_epoch = training_result_split.best_epoch_metrics.epoch
    return training_losses, validation_losses, epochs, best_epoch

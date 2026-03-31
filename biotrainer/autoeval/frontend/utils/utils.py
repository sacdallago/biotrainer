from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import SUPPORTED_FRAMEWORKS

from ...pipelines.autoeval_report import AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport
from ....utilities import MetricEstimate

from ....utilities.ranking import Ranking, RankingGroup, RankingEntry


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


def leaderboard_dataframe(loaded: List[AutoEvalReport]) -> Tuple[Ranking, Ranking]:
    """Compute leaderboard divided by framework (PBC and PGYM)."""
    pbc_entries = []
    pgym_entries = []
    # Build a dict: framework -> task -> list of (model, mean)
    for report in loaded:
        # Supervised PBC
        pbc_metrics = {}
        for fw_name, srep in report.supervised_results.items():
            fw_upper = fw_name.upper()
            if fw_upper not in SUPPORTED_FRAMEWORKS:
                continue
            for task in srep.get_task_names():
                # Extract the primary metric mean for the task (first test set/metric)
                metrics = srep.extract_metrics(task)
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
            if fw_upper not in SUPPORTED_FRAMEWORKS:
                continue
            for _, row in zrep.to_df().iterrows():
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


def get_training_validation_curves(result_dict: Dict) -> Tuple[
    Optional[List[float]], Optional[List[float]], Optional[List[int]], Optional[float]]:
    """Try to extract training and validation loss curves from a supervised result dict.

    The schema may vary; we try common keys.
    Returns (train_losses, val_losses, epochs, best_epoch)
    """
    if result_dict is None:
        return None, None, None, None

    # Common patterns to try
    train_keys = [
        "training_losses", "train_losses", "training_loss_curve", "train_loss_curve",
        ("training", "losses"), ("train", "losses"),
    ]
    val_keys = [
        "validation_losses", "val_losses", "validation_loss_curve", "val_loss_curve",
        ("validation", "losses"), ("val", "losses"),
    ]

    def _dig(d: Dict, key):
        if isinstance(key, tuple):
            cur = d
            for k in key:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur
        return d.get(key)

    train = None
    for k in train_keys:
        v = _dig(result_dict, k)
        if isinstance(v, list):
            train = v
            break

    val = None
    for k in val_keys:
        v = _dig(result_dict, k)
        if isinstance(v, list):
            val = v
            break

    # Fallback: look into nested training_results.{split}.training_loss / validation_loss (dict of epoch->loss)
    best_epoch = None
    if train is None or val is None:
        tr_res = result_dict.get("training_results") if isinstance(result_dict, dict) else None
        if isinstance(tr_res, dict):
            # Prefer hold_out, else first available split
            split_key = "hold_out" if "hold_out" in tr_res else (next(iter(tr_res.keys())) if tr_res else None)
            split_obj = tr_res.get(split_key) if split_key else None
            if isinstance(split_obj, dict):
                tr_loss = split_obj.get("training_loss")
                va_loss = split_obj.get("validation_loss")
                best_epoch = split_obj.get("best_training_epoch_metrics", {}).get("epoch")
                if isinstance(tr_loss, dict):
                    # keys are epochs as strings, values floats
                    try:
                        items = sorted(((int(k), float(v)) for k, v in tr_loss.items()), key=lambda x: x[0])
                        train = [v for _, v in items]
                    except Exception:
                        pass
                if isinstance(va_loss, dict):
                    try:
                        items = sorted(((int(k), float(v)) for k, v in va_loss.items()), key=lambda x: x[0])
                        val = [v for _, v in items]
                    except Exception:
                        pass

    epochs = list(range(1, 1 + max(len(train or []), len(val or [])))) if (train or val) else None
    return train, val, epochs, best_epoch


def supervised_task_metrics_dataframe(sreport: SupervisedFrameworkReport, task_name: str) -> pd.DataFrame:
    metrics = sreport.extract_metrics(task_name)
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics)


def zeroshot_task_metrics_dataframe(zreport: ZeroShotFrameworkReport, task_name: str) -> pd.DataFrame:
    rr = zreport.aggregated_results.get(task_name)
    if rr is None:
        return pd.DataFrame()
    data = []
    # SCC should be shown as absolute value with domain [0,1]
    try:
        data.append({
            "Metric": rr.scc.name,
            "Mean": round(abs(rr.scc.mean), 3),
            "Lower": round(abs(rr.scc.lower), 3),
            "Upper": round(abs(rr.scc.upper), 3),
        })
    except Exception:
        pass
    try:
        data.append({
            "Metric": rr.ndcg.name,
            "Mean": round(rr.ndcg.mean, 3),
            "Lower": round(rr.ndcg.lower, 3),
            "Upper": round(rr.ndcg.upper, 3),
        })
    except Exception:
        pass
    return pd.DataFrame(data)

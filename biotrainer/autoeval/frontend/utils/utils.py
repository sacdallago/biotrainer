from __future__ import annotations

import hashlib
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import SUPPORTED_FRAMEWORKS
from ...pipelines.autoeval_report import AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport


@dataclass
class LoadedReport:
    path: Path
    report: AutoEvalReport


def compute_report_uid(path: Path) -> str:
    """Compute a stable unique identifier for a report file.

    Uses SHA1 over the file bytes plus filename to reduce collision likelihood.
    """
    try:
        h = hashlib.sha1()
        # Include name to slightly diversify across same-content different files
        h.update(path.name.encode("utf-8", errors="ignore"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        # Fallback to path-based UID if reading fails
        return f"path:{str(path.resolve())}"


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


def load_reports_from_paths(paths: List[Path]) -> List[LoadedReport]:
    loaded: List[LoadedReport] = []
    for p in paths:
        try:
            if p.is_file():
                r = AutoEvalReport.from_json_file(p)
                loaded.append(LoadedReport(path=p, report=r))
            else:
                # search for autoeval_report_*.json inside directory
                for fp in p.glob("**/autoeval_report_*.json"):
                    try:
                        r = AutoEvalReport.from_json_file(fp)
                        loaded.append(LoadedReport(path=fp, report=r))
                    except Exception:
                        continue
        except Exception:
            continue
    # Deduplicate by (embedder_name, training_date, path)
    seen = set()
    uniq: List[LoadedReport] = []
    for item in loaded:
        key = (item.report.embedder_name, item.report.training_date, str(item.path))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def leaderboard_dataframe(loaded: List[LoadedReport]) -> pd.DataFrame:
    """Compute a simple leaderboard divided by framework (PBC and PGYM).

    For each framework, we compute per-task ranks across reports using the main metric
    mean, then compute the average rank across tasks per report.
    """
    rows: List[Dict] = []
    # Build a dict: framework -> task -> list of (model, mean)
    for item in loaded:
        r = item.report
        # Supervised frameworks contribute (e.g., PBC)
        for fw_name, srep in r.supervised_results.items():
            fw_upper = fw_name.upper()
            if fw_upper not in SUPPORTED_FRAMEWORKS:
                continue
            for task in srep.get_task_names():
                # Extract the primary metric mean for the task (first test set/metric)
                try:
                    metrics = srep.extract_metrics(task)
                    # choose first metric as representative for ranking
                    mean_val = metrics[0]["mean"] if metrics else None
                except Exception:
                    mean_val = None
                rows.append({
                    "Framework": fw_upper,
                    "Task": task,
                    "Model": r.embedder_name,
                    "Mean": mean_val,
                    "Report Path": str(item.path),
                })
        # Zero-shot frameworks contribute as well (e.g., PGYM)
        for fw_name, zrep in r.zeroshot_results.items():
            fw_upper = fw_name.upper()
            if fw_upper not in SUPPORTED_FRAMEWORKS:
                continue
            for task in zrep.get_task_names():
                try:
                    rr = zrep.aggregated_results[task]
                    # prefer scc (Spearman) then ndcg; use mean for ranking
                    mean_val = round(rr.scc.mean, 3) if rr and rr.scc else None
                except Exception:
                    mean_val = None
                rows.append({
                    "Framework": fw_upper,
                    "Task": task,
                    "Model": r.embedder_name,
                    "Mean": mean_val,
                    "Report Path": str(item.path),
                })

    if not rows:
        return pd.DataFrame(columns=["Framework", "Model", "Avg Rank", "Num Tasks"])

    df = pd.DataFrame(rows).dropna(subset=["Mean"])  # drop where mean missing
    if df.empty:
        return pd.DataFrame(columns=["Framework", "Model", "Avg Rank", "Num Tasks"])

    # Rank per (Framework, Task) — higher is better (rank 1 is best)
    df["Rank"] = df.groupby(["Framework", "Task"])['Mean'].rank(ascending=False, method='min')

    # Average rank per (Framework, Model)
    agg = (
        df.groupby(["Framework", "Model"], as_index=False)
        .agg(Avg_Rank=("Rank", "mean"), Num_Tasks=("Task", "nunique"))
    )
    agg = agg.sort_values(["Framework", "Avg_Rank"]).reset_index(drop=True)
    agg.rename(columns={"Avg_Rank": "Avg Rank", "Num_Tasks": "Num Tasks"}, inplace=True)
    return agg


def get_training_validation_curves(result_dict: Dict) -> Tuple[
    Optional[List[float]], Optional[List[float]], Optional[List[int]]]:
    """Try to extract training and validation loss curves from a supervised result dict.

    The schema may vary; we try common keys.
    Returns (train_losses, val_losses, epochs)
    """
    if result_dict is None:
        return None, None, None

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
    if train is None or val is None:
        tr_res = result_dict.get("training_results") if isinstance(result_dict, dict) else None
        if isinstance(tr_res, dict):
            # Prefer hold_out, else first available split
            split_key = "hold_out" if "hold_out" in tr_res else (next(iter(tr_res.keys())) if tr_res else None)
            split_obj = tr_res.get(split_key) if split_key else None
            if isinstance(split_obj, dict):
                tr_loss = split_obj.get("training_loss")
                va_loss = split_obj.get("validation_loss")
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
    return train, val, epochs


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

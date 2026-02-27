from __future__ import annotations

from io import BytesIO
from typing import List, Optional

import pandas as pd

from .utils import LoadedReport


def _abs_if_spearman(metric_name: str, value: float) -> float:
    m = (metric_name or "").lower()
    if "spearman" in m or m == "scc" or "spearmans-corr-coeff" in m:
        try:
            return abs(float(value))
        except Exception:
            return value
    return value


def build_supervised_compare_df(reports: List[LoadedReport], framework: Optional[str] = None) -> pd.DataFrame:
    """Collect supervised metrics across all provided reports into a plotting DataFrame.

    Columns: [TaskLabel, Task, Test Set, Metric, Model, Mean, Lower, Upper]
    TaskLabel mimics the AutoEvalReport plot labeling style.
    """
    rows = []
    for it in reports:
        s_dict = it.report.supervised_results or {}
        for fw_name, srep in s_dict.items():
            if framework and fw_name != framework:
                continue
            for task in srep.get_task_names():
                for m in srep.extract_metrics(task):
                    # Label like: Task\n(TestSet - Metric) if test set != 'test' else Task\n(Metric)
                    test_set = m["test_set_name"]
                    metric = m["evaluation_metric"]
                    if test_set != "test":
                        label = f"{task}\n({test_set} - {metric})"
                    else:
                        label = f"{task}\n({metric})"
                    rows.append({
                        "TaskLabel": label,
                        "Task": task,
                        "Test Set": test_set,
                        "Metric": metric,
                        "Model": it.report.embedder_name,
                        "Mean": _abs_if_spearman(metric, float(m["mean"])),
                        "Lower": _abs_if_spearman(metric, float(m["lower"])),
                        "Upper": _abs_if_spearman(metric, float(m["upper"]))
                    })
    df = pd.DataFrame(rows)
    return df


def build_zeroshot_compare_df(reports: List[LoadedReport], framework: Optional[str] = None) -> pd.DataFrame:
    """Collect zero-shot aggregated metrics across all provided reports into a plotting DataFrame.

    Columns: [TaskLabel, Task, Metric, Model, Mean, Lower, Upper]
    TaskLabel: Task\n(Metric)
    """
    rows = []
    for it in reports:
        z_dict = it.report.zeroshot_results or {}
        for fw_name, zrep in z_dict.items():
            if framework and fw_name != framework:
                continue
            for task in zrep.get_task_names():
                rr = zrep.aggregated_results.get(task)
                if rr is None:
                    continue
                for metric in [rr.scc, rr.ndcg]:
                    name = getattr(metric, "name", "metric")
                    mean = float(getattr(metric, "mean", float("nan")))
                    lower = float(getattr(metric, "lower", float("nan")))
                    upper = float(getattr(metric, "upper", float("nan")))
                    mean = _abs_if_spearman(name, mean)
                    lower = _abs_if_spearman(name, lower)
                    upper = _abs_if_spearman(name, upper)
                    rows.append({
                        "TaskLabel": f"{task}\n({name})",
                        "Task": task,
                        "Metric": name,
                        "Model": it.report.embedder_name,
                        "Mean": round(mean, 3),
                        "Lower": round(lower, 3),
                        "Upper": round(upper, 3),
                    })
    df = pd.DataFrame(rows)
    # Put virus blocks last, if present
    if not df.empty and "virus" in "".join(df["TaskLabel"].tolist()).lower():
        df = df.sort_values(by=[df["TaskLabel"].str.contains("virus", case=False)], ascending=True)
    return df


def plot_comparison(df: pd.DataFrame, title: Optional[str] = None):
    """Create a matplotlib/seaborn grouped bar plot with CIs and bottom labels.

    Returns: (fig, ax)
    """
    if df is None or df.empty:
        return None, None

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        return None, None

    # Order categories to make plot stable
    x_labels = list(pd.unique(df["TaskLabel"]))
    models = list(pd.unique(df["Model"]))

    fig_height = max(6, len(x_labels) * 0.6)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind", n_colors=len(models))

    # Ensure order
    plot_df = df.copy()
    plot_df["TaskLabel"] = pd.Categorical(plot_df["TaskLabel"], categories=x_labels, ordered=True)
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=models, ordered=True)

    sns.barplot(
        data=plot_df,
        x="TaskLabel",
        y="Mean",
        hue="Model",
        capsize=0.2,
        err_kws={"linewidth": 1},
        alpha=0.85,
        palette=palette,
        ax=ax,
    )

    # Rotate x-ticks by 45 degrees
    plt.xticks(rotation=45, ha='right')

    # Add manual CI whiskers to match autoeval style and numbers at bottom
    n_models = len(models)
    for i, model in enumerate(models):
        m_df = plot_df[plot_df["Model"] == model]
        for j, label in enumerate(x_labels):
            row = m_df[m_df["TaskLabel"] == label]
            if row.empty:
                continue
            y_mean = float(row["Mean"].iloc[0])
            low = float(row["Lower"].iloc[0])
            up = float(row["Upper"].iloc[0])
            # Compute bar position:
            x = j + (i - n_models / 2 + 0.5) * (0.8 / max(n_models, 1))
            color = palette[i]
            ax.vlines(x=x, ymin=low, ymax=up, color=color, linewidth=2, alpha=0.7)
            # value at bottom
            try:
                ax.text(x, 0.01, f"{y_mean:.3f}", ha="center", va="bottom", fontsize=8, rotation=90,
                        color="black", fontweight="bold")
            except Exception:
                pass

    ax.set_xlabel("Task")
    ax.set_ylabel("Score")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()


def fig_to_pdf_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

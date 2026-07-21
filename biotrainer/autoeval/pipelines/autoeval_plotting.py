from __future__ import annotations

from io import BytesIO
from typing import List, Optional

import pandas as pd




def _get_palette(n_models: int):
    try:
        import seaborn as sns
        colorblind_palette = sns.color_palette("colorblind", n_colors=min(n_models, 10))
        if n_models > 10:
            husl_palette = sns.color_palette("husl", n_colors=max(1, n_models - 10))
            palette = colorblind_palette + husl_palette
        else:
            palette = colorblind_palette
        return palette
    except Exception:
        return None



def plot_comparison(df: pd.DataFrame):
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

    fig_width = 16
    fig_height = max(6, len(x_labels) * 0.6)

    # Create figure with 2 subplots: one for legend (bottom) and one for the plot (top)
    fig = plt.figure(figsize=(fig_width, fig_height + 1), dpi=600)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.1], hspace=2.0)

    # Main plot subplot (top)
    ax = fig.add_subplot(gs[0])

    # Legend subplot (bottom)
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    sns.set_style("whitegrid")
    palette = _get_palette(len(models))

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

    # Customize plot
    ax.set_title('Performance Comparison Across Tasks')
    ax.set_xlabel('Task', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')

    # Set yticks fontsize
    ax.tick_params(axis='y', labelsize=16)

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
                x_text = x + 0.02
                ax.text(x_text, 0.01, f"{y_mean:.3f}", ha="center", va="bottom", fontsize=8, rotation=90,
                        color="black", fontweight="bold")
            except Exception:
                pass

    # Move legend to the top subplot
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_ax.legend(legend_handles, legend_labels, loc='center', fancybox=True, shadow=True, ncol=4, fontsize=12)

    # Remove legend from main plot
    if ax.get_legend():
        ax.get_legend().remove()

    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    return fig, ax


def compute_paired_delta_stats(reports_by_model: dict, baseline_model: str, z: float = 1.96) -> dict:
    """Confidence interval of the *paired* per-dataset score difference (model - baseline).

    For zero-shot frameworks (e.g. PGYM) every model is scored on the same datasets, so the
    honest uncertainty of a model-vs-baseline comparison is the CI of the per-dataset deltas,
    not the spread of either model's scores across datasets. This compares the two models on
    the *same* dataset, one at a time, which cancels per-dataset difficulty.

    ``reports_by_model`` maps model name -> a report exposing ``aggregated_members``
    (combined_task_name -> [dataset_name]) and ``individual_results`` (dataset_name ->
    RankingResult with ``.scc.mean`` / ``.ndcg.mean``).

    Returns ``{(model, task, metric): {"mean_pp": signed_mean_delta_pp,
    "ci_pp": half_width_pp, "n": n_shared}}`` (95% CI, normal approximation z=1.96).
    Entries are only produced where >= 2 datasets are shared with the baseline; everything
    else is omitted and the plot falls back to its default whisker.
    """
    try:
        import numpy as np
    except Exception:
        return {}

    stats: dict = {}
    baseline = reports_by_model.get(baseline_model)
    if baseline is None:
        return stats
    base_members = getattr(baseline, "aggregated_members", {}) or {}
    base_individual = getattr(baseline, "individual_results", {}) or {}

    for model_name, report in reports_by_model.items():
        if model_name == baseline_model:
            continue
        members_by_task = getattr(report, "aggregated_members", {}) or {}
        individual = getattr(report, "individual_results", {}) or {}
        for task, members in members_by_task.items():
            base_task_members = set(base_members.get(task, []))
            shared = [d for d in members
                      if d in base_task_members and d in individual and d in base_individual]
            if len(shared) < 2:
                continue
            for metric in ("scc", "ndcg"):
                try:
                    m_scores = np.array([getattr(individual[d], metric).mean for d in shared], dtype=float)
                    b_scores = np.array([getattr(base_individual[d], metric).mean for d in shared], dtype=float)
                except Exception:
                    continue
                deltas = m_scores - b_scores
                n = len(deltas)
                se = float(np.std(deltas, ddof=1)) / (n ** 0.5)
                stats[(model_name, task, metric)] = {
                    "mean_pp": float(deltas.mean()) * 100.0,
                    "ci_pp": z * se * 100.0,
                    "n": n,
                }
    return stats


def plot_delta_comparison(df: pd.DataFrame, baseline_model: str, paired_stats: dict):
    """Create a matplotlib/seaborn grouped bar plot showing deltas relative to a baseline model.

    Whiskers are the 95% CI of the paired per-dataset difference (model - baseline) from
    ``paired_stats`` (see :func:`compute_paired_delta_stats`). A '*' marks comparisons whose
    CI excludes zero; non-significant bars are greyed out.

    Returns: (fig, ax)
    """
    if df is None or df.empty or baseline_model not in df["Model"].unique():
        return None, None

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None, None

    # Calculate deltas
    # We need to match tasks across models.
    # The df has columns: Task, Metric, TaskLabel, Model, Mean, Lower, Upper
    # Supervised also has: Test Set, Protocol
    # Some rows might be missing for some models.

    # Check if this is supervised (has Test Set column) or zero-shot
    has_test_set = "Test Set" in df.columns

    # Identify the baseline rows
    baseline_df = df[df["Model"] == baseline_model].copy()
    # Create a key for matching: Task | Test Set (if exists) | Metric
    df_key = baseline_df["Task"].astype(str)
    if has_test_set:
        df_key += "|" + baseline_df["Test Set"].fillna("").astype(str)
    df_key += "|" + baseline_df["Metric"].astype(str)
    baseline_df["key"] = df_key
    baseline_map = baseline_df.set_index("key")

    # Models to plot (excluding baseline)
    other_models = [m for m in df["Model"].unique() if m != baseline_model]
    if not other_models:
        return None, None

    delta_rows = []

    for _, row in df.iterrows():
        if row["Model"] == baseline_model:
            continue

        # Build the same key structure
        key = str(row["Task"])
        if has_test_set:
            key += "|" + str(row.get("Test Set", ""))
        key += "|" + str(row["Metric"])

        if key not in baseline_map.index:
            continue

        base_row = baseline_map.loc[key]
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]

        # Use the paired CI for the whisker and bar centre.
        # The raw signed paired mean is used so the bar stays consistent with the whisker
        # (the aggregated row["Mean"] is abs()'d for scc/Spearman, which would mis-place
        # the whisker if a mean is < 0).
        pstat = paired_stats.get((row["Model"], str(row["Task"]), str(row["Metric"])))
        if pstat is None:
            continue
        mean_delta = float(pstat["mean_pp"])
        ci = float(pstat["ci_pp"])
        lower_delta = mean_delta - ci
        upper_delta = mean_delta + ci
        significant = bool(abs(mean_delta) > ci)

        delta_rows.append({
            "Model": row["Model"],
            "TaskLabel": row["TaskLabel"],
            "Mean": mean_delta,
            "Lower": lower_delta,
            "Upper": upper_delta,
            "Significant": significant,
            "key": key
        })

    if not delta_rows:
        return None, None

    delta_df = pd.DataFrame(delta_rows)

    x_labels = list(pd.unique(delta_df["TaskLabel"]))
    models = list(pd.unique(delta_df["Model"]))

    fig_width = 16
    fig_height = max(6, len(x_labels) * 0.6)

    # Create figure with 2 subplots: one for legend (bottom) and one for the plot (top)
    fig = plt.figure(figsize=(fig_width, fig_height + 1), dpi=600)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.1], hspace=2.0)

    # Main plot subplot (top)
    ax = fig.add_subplot(gs[0])

    # Legend subplot (bottom)
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')

    sns.set_style("whitegrid")
    palette = _get_palette(len(models))

    # Ensure order
    delta_df["TaskLabel"] = pd.Categorical(delta_df["TaskLabel"], categories=x_labels, ordered=True)
    delta_df["Model"] = pd.Categorical(delta_df["Model"], categories=models, ordered=True)

    sns.barplot(
        data=delta_df,
        x="TaskLabel",
        y="Mean",
        hue="Model",
        palette=palette,
        alpha=0.85,
        ax=ax
    )

    ax.axhline(0, color='black', linewidth=1.5)
    title = f'Performance Delta relative to {baseline_model}\n(whiskers: 95% CI of paired per-dataset difference;  * = significant)'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Task', fontsize=16)
    ax.set_ylabel('Delta (pp)', fontsize=16)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')

    # Set yticks fontsize
    ax.tick_params(axis='y', labelsize=16)

    n_models = len(models)
    _span = max(1.0, float(delta_df[["Lower", "Upper"]].abs().to_numpy().max()))
    off_up, off_dn = _span * 0.05, _span * 0.10
    for i, model in enumerate(models):
        m_df = delta_df[delta_df["Model"] == model]
        for j, label in enumerate(x_labels):
            row = m_df[m_df["TaskLabel"] == label]
            if row.empty:
                continue
            y_mean = float(row["Mean"].iloc[0])
            low = float(row["Lower"].iloc[0])
            up = float(row["Upper"].iloc[0])
            sig = bool(row["Significant"].iloc[0])

            x = j + (i - n_models / 2 + 0.5) * (0.8 / max(n_models, 1))
            color = palette[i]
            # Grey out the whisker + label when the paired CI overlaps zero (not significant).
            not_significant = not sig
            ebar_color = "#9e9e9e" if not_significant else color

            # Error bars
            ax.vlines(x=x, ymin=low, ymax=up, color=ebar_color, linewidth=2)
            # Cap lines
            ax.hlines(y=[low, up], xmin=x - 0.05, xmax=x + 0.05, color=ebar_color, linewidth=2)

            # Value text; '*' marks a statistically significant difference (CI excludes 0).
            try:
                star = " *" if sig else ""
                text_color = "#9e9e9e" if not_significant else "black"
                text_y = up + off_up if y_mean >= 0 else low - off_dn
                ax.text(x, text_y, f"{y_mean:+.1f}{star}", ha="center", va="bottom", fontsize=9,
                        color=text_color, fontweight="bold")
            except Exception:
                pass

    # Autoscale tightly so the bars/whiskers are visible.
    span = max(1.0, float(delta_df[["Lower", "Upper", "Mean"]].abs().to_numpy().max())) * 1.35
    ax.set_ylim(-span, span)

    # Move legend to the bottom subplot
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_ax.legend(legend_handles, legend_labels, loc='center', fancybox=True, shadow=True, ncol=4, fontsize=12)

    # Remove legend from main plot
    if ax.get_legend():
        ax.get_legend().remove()

    # Adjust layout to prevent label cutoff
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

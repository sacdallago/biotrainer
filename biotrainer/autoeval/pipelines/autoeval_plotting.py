from __future__ import annotations

from io import BytesIO
from typing import List, Optional

import pandas as pd


def aggregate_dfs(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        return pd.concat(dfs, ignore_index=True)
    except ValueError:
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
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=600)
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

    # Customize plot
    plt.title('Performance Comparison Across Tasks')
    plt.xlabel('Task', fontsize=16)
    plt.ylabel('Score', fontsize=16)

    # Rotate x-axis labels for better readability 
    plt.xticks(rotation=45, ha='right', fontsize=14)

    # Set yticks fontsize
    plt.yticks(fontsize=16)

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

    # Adjust legend position
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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

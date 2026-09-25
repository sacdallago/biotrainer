from __future__ import annotations

import pandas as pd
import altair as alt
import streamlit as st

from typing import List, Tuple, Optional, Dict
from biotrainer_core.data_classes import BiotrainerModelResult
from biotrainer_core.data_classes.autoeval import (
    AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport,
    ContactFrameworkReport, UnsupervisedFrameworkReport, FrameworkReport,
    DEV_MODE_INDICATOR, DEV_MODE_ABLATED_INDICATOR
)
from ..model import DashboardReport

from ..state import AutoevalSessionState
from ..utils import utils as frontend_utils

_BIN_METRICS_01 = {"accuracy", "acc", "f1", "f1_score", "auc", "auroc", "mcc"}


def _postprocess_task_df(df_task: pd.DataFrame) -> pd.DataFrame:
    if df_task is None or df_task.empty:
        return df_task
    df = df_task.copy()
    # Normalize Spearman (absolute values, 0..1)
    if "Metric" in df.columns:
        mask_scc = df["Metric"].str.lower().str.contains("spearman") | (
                df["Metric"].str.lower() == "scc"
        ) | df["Metric"].str.lower().str.contains("spearmans-corr-coeff")
        for col in ["Mean", "Lower", "Upper"]:
            if col in df.columns:
                df.loc[mask_scc, col] = df.loc[mask_scc, col].abs()
    # Drop "Task" column and rename "TaskLabel" to "Task"
    if "Task" in df.columns and "TaskLabel" in df.columns:
        df = df.drop(columns=["Task"])
    if "Protocol" in df.columns:
        df = df.drop(columns=["Protocol"])
    if "TaskLabel" in df.columns:
        df = df.rename(columns={"TaskLabel": "Task"})
    return df


def _add_mode_column(df_comp: pd.DataFrame) -> pd.DataFrame:
    if df_comp is None or df_comp.empty:
        return df_comp
    df = df_comp.copy()

    def determine_mode(row) -> str:
        if "Test Set" in row and pd.notna(row["Test Set"]):
            test_set = str(row["Test Set"]).lower()
            if test_set == "validation":
                return "Dev"
            return "Eval"

        task_str = str(row.get("Task", "")) + " " + str(row.get("TaskLabel", ""))
        if DEV_MODE_ABLATED_INDICATOR in task_str:
            return "Ablation"
        elif DEV_MODE_INDICATOR in task_str:
            return "Dev"
        return "Eval"

    df["Mode"] = df.apply(determine_mode, axis=1)
    return df


def _metric_domain(metric_name: str) -> Tuple[float, float] | None:
    m = (metric_name or "").lower()
    if any(k in m for k in ["spearman", "scc", "accuracy", "acc", "f1", "auc", "auroc", "mcc"]):
        return 0.0, 1.0
    return None


def _build_metrics_chart(df_task: pd.DataFrame):
    if not df_task.empty:
        dfp = df_task.copy()
        dfp["CI"] = dfp.apply(lambda r: f"[{r['Lower']}, {r['Upper']}]", axis=1)
        bars = alt.Chart(dfp).mark_bar().encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y("Mean:Q", title="Score"),
            tooltip=[
                alt.Tooltip("Metric", title="Metric"),
                alt.Tooltip("Mean", title="Mean"),
                alt.Tooltip("CI", title="95% CI"),
            ],
        )
        error_bars = alt.Chart(dfp).mark_errorbar().encode(
            x=alt.X("Metric:N"),
            y=alt.Y("Lower:Q", title=""),
            y2="Upper:Q",
        )
        st.altair_chart((bars + error_bars).properties(height=320), use_container_width=True)


def _build_comparison_chart(df_comp: pd.DataFrame):
    if df_comp is None or df_comp.empty:
        return
    dfp = _add_mode_column(df_comp)
    dfp = _postprocess_task_df(dfp)
    if "Mode" not in dfp.columns or len(dfp["Mode"].unique()) <= 1:
        return

    dfp["CI"] = dfp.apply(lambda r: f"[{r['Lower']}, {r['Upper']}]", axis=1)

    modes_order = ["Dev", "Eval", "Ablation"]
    present_modes = [m for m in modes_order if m in dfp["Mode"].unique()]
    for m in dfp["Mode"].unique():
        if m not in present_modes:
            present_modes.append(m)

    mode_colors = {
        "Dev": "#1f77b4",       # Blue
        "Eval": "#2ca02c",      # Green
        "Ablation": "#d62728",  # Red
    }
    color_range = [mode_colors.get(m, "#7f7f7f") for m in present_modes]

    bars = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("Metric:N", title="Metric"),
        xOffset=alt.XOffset("Mode:N", sort=present_modes),
        y=alt.Y("Mean:Q", title="Score"),
        color=alt.Color(
            "Mode:N",
            scale=alt.Scale(domain=present_modes, range=color_range),
            legend=alt.Legend(title="Mode"),
        ),
        tooltip=[
            alt.Tooltip("Mode:N", title="Mode"),
            alt.Tooltip("Metric:N", title="Metric"),
            alt.Tooltip("Mean:Q", title="Mean"),
            alt.Tooltip("CI:N", title="95% CI"),
        ],
    )
    error_bars = alt.Chart(dfp).mark_errorbar().encode(
        x=alt.X("Metric:N"),
        xOffset=alt.XOffset("Mode:N", sort=present_modes),
        y=alt.Y("Lower:Q", title=""),
        y2="Upper:Q",
    )
    chart = (bars + error_bars).properties(height=320)
    st.markdown("#### Mode Comparison (Dev vs. Eval vs. Ablation)")
    st.altair_chart(chart, use_container_width=True)


def _select_framework(framework_dict: Dict[str, FrameworkReport], kind: str) -> Optional[Tuple[str, FrameworkReport]]:
    fw_names = list(framework_dict.keys())
    if not fw_names:
        st.info("No frameworks available.")
        return None
    fw_sel = st.selectbox("Framework", options=fw_names, key=f"fw_selector_{kind}")
    return fw_sel, framework_dict[fw_sel]


def _select_task(report: FrameworkReport, dev_mode: bool, kind: str) -> Optional[str]:
    tasks = report.filter_tasks_by_mode(dev_mode)
    if not tasks:
        st.info("No tasks available.")
        return None
    task = st.selectbox("Task", options=tasks, key=f"task_selector_{kind}")
    return task


def _render_task_metrics_and_charts(report: FrameworkReport, task: str, dev_mode: bool):
    df_task = report.to_df(all_metrics=True, development_mode=dev_mode, task_name_filter=lambda x: x != task)
    df_task = _postprocess_task_df(df_task)
    st.dataframe(df_task, use_container_width=True, hide_index=True)

    _build_metrics_chart(df_task)

    try:
        df_comp = report.to_task_comparison_df(task)
        _build_comparison_chart(df_comp)
    except Exception:
        pass


def _render_framework(framework_dict: Dict[str, FrameworkReport], dev_mode: bool, kind: str):
    fw_data = _select_framework(framework_dict, kind)
    if not fw_data:
        return
    _, report = fw_data
    task = _select_task(report, dev_mode, kind)
    if not task:
        return
    _render_task_metrics_and_charts(report, task, dev_mode)


def _render_supervised(autoeval_report: AutoEvalReport, dev_mode: bool):
    fw_data = _select_framework(autoeval_report.supervised_results, "supervised")
    if not fw_data:
        return
    _, rep = fw_data
    srep: SupervisedFrameworkReport = rep  # type: ignore
    embedding_stats = srep.accumulated_embedding_stats()
    if embedding_stats:
        st.markdown("#### Embedding Statistics")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Dimensions", embedding_stats.dims)
        with stats_cols[1]:
            st.metric("Residues Tracked", f"{embedding_stats.n_tracked:,}")
        with stats_cols[2]:
            st.metric("Min Value", f"{embedding_stats.min:.2f}")
        with stats_cols[3]:
            st.metric("Max Value", f"{embedding_stats.max:.2f}")

        # Range plot visualization
        range_df = pd.DataFrame({
            'dummy': [1],
            'min': [embedding_stats.min],
            'max': [embedding_stats.max]
        })

        range_chart = alt.Chart(range_df).mark_rule(size=8).encode(
            x=alt.X('min:Q',
                    scale=alt.Scale(domain=[embedding_stats.min - abs(embedding_stats.min) * 0.1,
                                            embedding_stats.max + abs(embedding_stats.max) * 0.1]),
                    title='Embedding Value Range'),
            x2='max:Q',
            tooltip=[
                alt.Tooltip('min:Q', title='Min', format='.4f'),
                alt.Tooltip('max:Q', title='Max', format='.4f')
            ]
        ).properties(height=80)

        # Add tick marks at min and max
        ticks = alt.Chart(range_df).transform_fold(
            ['min', 'max'],
            as_=['position_type', 'value']
        ).mark_tick(size=20, thickness=3).encode(
            x=alt.X('value:Q', title='Embedding Value Range'),
            color=alt.Color('position_type:N',
                            scale=alt.Scale(domain=['min', 'max'], range=['blue', 'red']),
                            legend=alt.Legend(title='Position')),
            tooltip=[
                alt.Tooltip('position_type:N', title='Position'),
                alt.Tooltip('value:Q', title='Value', format='.4f')
            ]
        )

        # Combine range line and ticks
        combined_chart = range_chart + ticks
        st.altair_chart(combined_chart, use_container_width=True)

        st.divider()

    task = _select_task(srep, dev_mode, "supervised")
    if not task:
        return

    _render_task_metrics_and_charts(srep, task, dev_mode)

    # Loss curves if present
    model_result: Optional[BiotrainerModelResult] = srep.task_results.get(task)
    if not model_result:
        st.warning("No model result available for this task!")
        return
    tr, va, epochs, best_epoch = frontend_utils.get_training_validation_curves(model_result)
    if tr or va:
        st.markdown("#### Training / Validation Loss")
        plot_df = pd.DataFrame({"epoch": epochs})
        if tr:
            plot_df["train_loss"] = tr
        if va:
            plot_df["val_loss"] = va
        try:
            plot_df["epoch"] = plot_df["epoch"] - 1
            plot_dfm = plot_df.melt("epoch", var_name="series", value_name="loss")
            line = (
                alt.Chart(plot_dfm)
                .mark_line()
                .encode(
                    x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1, format='d')),
                    y="loss:Q",
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["train_loss", "val_loss"],
                            range=["blue", "orange"],
                        ),
                        legend=alt.Legend(title="Loss Type"),
                    ),
                )
                .properties(height=320)
            )

            # Add vertical line for best epoch
            rule = (
                alt.Chart(pd.DataFrame({"best_epoch": [best_epoch]}))
                .mark_rule(color="black", strokeDash=[5, 5], size=2)
                .encode(
                    x="best_epoch:Q",
                    tooltip=[alt.Tooltip("best_epoch:Q", title="Best Epoch")]
                )
            )

            chart = (line + rule)
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.line_chart(plot_df.set_index("epoch"))
    else:
        st.caption("No training/validation loss curves found in this result.")


def render_detailed(state: AutoevalSessionState, active: list[DashboardReport]):
    st.subheader("Detailed Report View")

    if not active:
        st.info("Load reports to inspect details.")
        return

    labels = [f"{db_report.report.embedder_name} ({db_report.report.training_date})" for db_report in active]
    idx = st.selectbox("Select report", options=list(range(len(active))), format_func=lambda i: labels[i])
    db_report: DashboardReport = active[idx]
    autoeval_report: AutoEvalReport = db_report.report

    report_is_development = autoeval_report.is_development()
    dev_mode = report_is_development or state.get_development_mode()  # Force development mode if report is in dev mode

    # Summary
    st.metric("Model", autoeval_report.embedder_name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training date", autoeval_report.training_date)
    with col2:
        fwks = autoeval_report.all_framework_names()
        st.metric("Frameworks", len(fwks), help=", ".join(fwks))
    with (col3):
        development_help = ("At least one of the frameworks "
                            "has development mode enabled.") if report_is_development else ("All frameworks "
                                                                                            "used evaluation mode on "
                                                                                            "the full test sets.")
        st.metric("Development mode", report_is_development, help=development_help)

    if db_report.official:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Official Report", help="This report is officially approved "
                                                "and part of the official leaderboard.")
        with col2:
            st.markdown(f"[Citation]({db_report.citation})")

    st.divider()
    framework_tab_names: List[Tuple[str, str]] = []  # (label, kind)
    if autoeval_report.supervised_results:
        framework_tab_names.append(("Supervised", "supervised"))
    if autoeval_report.unsupervised_results:
        framework_tab_names.append(("Unsupervised", "unsupervised"))
    if autoeval_report.zeroshot_results:
        framework_tab_names.append(("Zero-Shot", "zeroshot"))
    if autoeval_report.zeroshot_contact_results:
        framework_tab_names.append(("Zero-Shot Contact", "zeroshot_contact"))
    if autoeval_report.supervised_contact_results:
        framework_tab_names.append(("Supervised Contact", "supervised_contact"))

    if not framework_tab_names:
        st.info("This report has no results.")
        return

    tabs = st.tabs([name for name, _ in framework_tab_names])
    for tab, (_, kind) in zip(tabs, framework_tab_names):
        with tab:
            if kind == "supervised":
                _render_supervised(autoeval_report, dev_mode)
            elif kind == "unsupervised":
                _render_framework(autoeval_report.unsupervised_results, dev_mode, kind)
            elif kind == "zeroshot":
                _render_framework(autoeval_report.zeroshot_results, dev_mode, kind)
            elif kind == "zeroshot_contact":
                _render_framework(autoeval_report.zeroshot_contact_results, dev_mode, kind)
            elif kind == "supervised_contact":
                _render_framework(autoeval_report.supervised_contact_results, dev_mode, kind)

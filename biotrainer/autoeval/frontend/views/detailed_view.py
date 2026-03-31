from __future__ import annotations

from typing import List, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ...pipelines.autoeval_report import AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport
from ..utils import utils as frontend_utils


_BIN_METRICS_01 = {"accuracy", "acc", "f1", "f1_score", "auc", "auroc", "mcc"}


def _scale_supervised_metric_df(df_task: pd.DataFrame) -> pd.DataFrame:
    if df_task is None or df_task.empty:
        return df_task
    df = df_task.copy()
    # Normalize Spearman (absolute values, 0..1)
    mask_scc = df["evaluation_metric"].str.lower().str.contains("spearman") | (
        df["evaluation_metric"].str.lower() == "scc"
    ) | df["evaluation_metric"].str.lower().str.contains("spearmans-corr-coeff")
    for col in ["mean", "lower", "upper"]:
        df.loc[mask_scc, col] = df.loc[mask_scc, col].abs()
    return df


def _metric_domain(metric_name: str) -> Tuple[float, float] | None:
    m = (metric_name or "").lower()
    if any(k in m for k in ["spearman", "scc", "accuracy", "acc", "f1", "auc", "auroc", "mcc"]):
        return 0.0, 1.0
    return None


def render_detailed(active: list[AutoEvalReport]):
    st.subheader("Detailed Report View")
    if not active:
        st.info("Load reports to inspect details.")
        return

    labels = [f"{report.embedder_name} ({report.training_date})" for report in active]
    idx = st.selectbox("Select report", options=list(range(len(active))), format_func=lambda i: labels[i])
    report = active[idx]

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", report.embedder_name)
    with col2:
        st.metric("Training date", report.training_date)
    with col3:
        fwks = list(report.supervised_results.keys()) + list(report.zeroshot_results.keys())
        st.metric("Frameworks", ", ".join(sorted(set([f.upper() for f in fwks]))) or "-")

    st.divider()
    framework_tab_names: List[Tuple[str, str]] = []  # (label, kind)
    if report.supervised_results:
        framework_tab_names.append(("Supervised", "supervised"))
    if report.zeroshot_results:
        framework_tab_names.append(("Zero-Shot", "zeroshot"))

    if not framework_tab_names:
        st.info("This report has no results.")
        return

    tabs = st.tabs([name for name, _ in framework_tab_names])
    for tab, (_, kind) in zip(tabs, framework_tab_names):
        with tab:
            if kind == "supervised":
                fw_names = list(report.supervised_results.keys())
                fw_sel = st.selectbox("Framework", options=fw_names)
                srep: SupervisedFrameworkReport = report.supervised_results[fw_sel]
                tasks = srep.get_task_names()
                if not tasks:
                    st.info("No tasks available.")
                    continue
                task = st.selectbox("Task", options=tasks)
                df_task = frontend_utils.supervised_task_metrics_dataframe(srep, task)
                df_task = _scale_supervised_metric_df(df_task)
                st.dataframe(df_task, use_container_width=True, hide_index=True)

                # Per-task plot (means with CIs) across test sets/metrics
                if not df_task.empty:
                    try:
                        import altair as alt
                        dfp = df_task.copy()
                        dfp["CI"] = dfp.apply(lambda r: f"[{r['lower']}, {r['upper']}]", axis=1)
                        # Determine if all metrics are in [0,1] family; if yes, enforce domain [0,1]
                        uniq_metrics = sorted(dfp["evaluation_metric"].astype(str).str.lower().unique())
                        if all(_metric_domain(m) == (0.0, 1.0) for m in uniq_metrics):
                            domain = (0.0, 1.0)
                        else:
                            domain = None
                        y_scale = alt.Scale(domain=domain) if domain else alt.Undefined
                        y_enc = alt.Y("mean:Q", title="Score", scale=y_scale)

                        # Base bar chart
                        bars = alt.Chart(dfp).mark_bar().encode(
                            x=alt.X("test_set_name:N", title="Test Set"),
                            y=y_enc,
                            color=alt.Color("evaluation_metric:N", title="Metric"),
                            tooltip=[
                                alt.Tooltip("evaluation_metric", title="Metric"),
                                alt.Tooltip("test_set_name", title="Test Set"),
                                alt.Tooltip("mean", title="Mean"),
                                alt.Tooltip("CI", title="95% CI"),
                            ],
                        )

                        # Adding error bars for confidence intervals
                        error_bars = alt.Chart(dfp).mark_errorbar().encode(
                            x=alt.X("test_set_name:N", title="Test Set"),
                            y=alt.Y("lower:Q", title=""),
                            y2="upper:Q",
                            color=alt.Color("evaluation_metric:N", title="Metric")
                        )

                        # Combine bar chart and error bars
                        chart = bars + error_bars
                        st.altair_chart(chart.properties(height=320), use_container_width=True)
                    except Exception:
                        pass

                # Loss curves if present
                result_dict = srep.results.get(task)
                tr, va, epochs, best_epoch = frontend_utils.get_training_validation_curves(result_dict)
                if tr or va:
                    st.markdown("#### Training / Validation Loss")
                    plot_df = pd.DataFrame({"epoch": epochs})
                    if tr:
                        plot_df["train_loss"] = tr
                    if va:
                        plot_df["val_loss"] = va
                    try:
                        import altair as alt
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

            else:  # zeroshot
                fw_names = list(report.zeroshot_results.keys())
                fw_sel = st.selectbox("Framework", options=fw_names)
                zrep: ZeroShotFrameworkReport = report.zeroshot_results[fw_sel]
                tasks = zrep.get_task_names()
                if not tasks:
                    st.info("No tasks available.")
                    continue
                task = st.selectbox("Task", options=tasks)
                df_task = frontend_utils.zeroshot_task_metrics_dataframe(zrep, task)
                st.dataframe(df_task, use_container_width=True, hide_index=True)

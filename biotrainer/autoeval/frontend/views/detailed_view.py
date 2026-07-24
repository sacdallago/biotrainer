from __future__ import annotations

import pandas as pd
import altair as alt
import streamlit as st

from typing import List, Tuple, Optional
from biotrainer_core.data_classes import BiotrainerModelResult
from biotrainer_core.data_classes.autoeval import AutoEvalReport, SupervisedFrameworkReport, ZeroShotFrameworkReport, \
    ContactFrameworkReport

from ..state import AutoevalSessionState
from ..utils import utils as frontend_utils

_BIN_METRICS_01 = {"accuracy", "acc", "f1", "f1_score", "auc", "auroc", "mcc"}


def _postprocess_task_df(df_task: pd.DataFrame) -> pd.DataFrame:
    if df_task is None or df_task.empty:
        return df_task
    df = df_task.copy()
    # Normalize Spearman (absolute values, 0..1)
    mask_scc = df["Metric"].str.lower().str.contains("spearman") | (
            df["Metric"].str.lower() == "scc"
    ) | df["Metric"].str.lower().str.contains("spearmans-corr-coeff")
    for col in ["Mean", "Lower", "Upper"]:
        df.loc[mask_scc, col] = df.loc[mask_scc, col].abs()
    # Drop "Task" column and rename "TaskLabel" to "Task"
    df = df.drop(columns=["Task"])
    df = df.drop(columns=["Protocol"]) if "Protocol" in df.columns else df
    df = df.rename(columns={"TaskLabel": "Task"})
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


def render_detailed(state: AutoevalSessionState, active: list[AutoEvalReport]):
    st.subheader("Detailed Report View")

    if not active:
        st.info("Load reports to inspect details.")
        return

    labels = [f"{report.embedder_name} ({report.training_date})" for report in active]
    idx = st.selectbox("Select report", options=list(range(len(active))), format_func=lambda i: labels[i])
    report: AutoEvalReport = active[idx]
    report_is_development = report.is_development()

    dev_mode = report_is_development or state.get_development_mode()  # Force development mode if report is in dev mode

    # Summary
    st.metric("Model", report.embedder_name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training date", report.training_date)
    with col2:
        fwks = report.all_framework_names()
        st.metric("Frameworks", len(fwks), help=", ".join(fwks))
    with (col3):
        development_help = ("At least one of the frameworks "
                            "has development mode enabled.") if report_is_development else ("All frameworks "
                                                                                            "used evaluation mode on "
                                                                                            "the full test sets.")
        st.metric("Development mode", report_is_development, help=development_help)

    st.divider()
    framework_tab_names: List[Tuple[str, str]] = []  # (label, kind)
    if report.supervised_results:
        framework_tab_names.append(("Supervised", "supervised"))
    if report.zeroshot_results:
        framework_tab_names.append(("Zero-Shot", "zeroshot"))
    if report.zeroshot_contact_results:
        framework_tab_names.append(("Zero-Shot Contact", "zeroshot_contact"))
    if report.supervised_contact_results:
        framework_tab_names.append(("Supervised Contact", "supervised_contact"))

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
                    # Create a dataframe with a single row representing the range
                    range_df = pd.DataFrame({
                        'dummy': [1],
                        'min': [embedding_stats.min],
                        'max': [embedding_stats.max]
                    })

                    # Create a range plot using a rule mark
                    import altair as alt
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

                task = st.selectbox("Task", options=tasks)
                df_sup = srep.to_df(all_metrics=True, development_mode=dev_mode)
                df_task = df_sup[df_sup["Task"] == task] if not df_sup.empty else df_sup
                df_task = _postprocess_task_df(df_task)
                st.dataframe(df_task, use_container_width=True, hide_index=True)

                _build_metrics_chart(df_task)

                # Loss curves if present
                model_result: Optional[BiotrainerModelResult] = srep.results.get(task)
                if not model_result:
                    st.warning("No model result available for this task!")
                    continue
                tr, va, epochs, best_epoch = frontend_utils.get_training_validation_curves(model_result)
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

            elif kind == "zeroshot":  # zeroshot
                fw_names = list(report.zeroshot_results.keys())
                fw_sel = st.selectbox("Framework", options=fw_names)
                zrep: ZeroShotFrameworkReport = report.zeroshot_results[fw_sel]
                tasks = zrep.get_task_names()
                if not tasks:
                    st.info("No tasks available.")
                    continue
                task = st.selectbox("Task", options=tasks)
                # Zero-shot doesn't support development mode yet, but we pass it anyway for consistency
                # though zeroshot_task_metrics_dataframe doesn't accept it yet.
                df_zrep = zrep.to_df(all_metrics=True, development_mode=dev_mode)
                df_task = df_zrep[df_zrep["Task"] == task] if not df_zrep.empty else df_zrep
                df_task = _postprocess_task_df(df_task)
                st.dataframe(df_task, use_container_width=True, hide_index=True)
                _build_metrics_chart(df_task)

            elif kind in ("zeroshot_contact", "supervised_contact"):  # contact
                if kind == "zeroshot_contact":
                    contact_results = report.zeroshot_contact_results
                else:
                    contact_results = report.supervised_contact_results
                fw_names = list(contact_results.keys())
                fw_sel = st.selectbox("Framework", options=fw_names, key=f"fw_selector_contact_{kind}")
                crep: ContactFrameworkReport = contact_results[fw_sel]
                tasks = crep.get_task_names()
                if not tasks:
                    st.info("No tasks available.")
                    continue
                task = st.selectbox("Task", options=tasks, key=f"task_selector_contact_{kind}")
                df_task = crep.to_df(all_metrics=True, development_mode=dev_mode)
                df_task = df_task[df_task["Task"] == task] if not df_task.empty else df_task
                st.dataframe(df_task[["Task", "Metric", "Mean", "Lower", "Upper"]], use_container_width=True,
                             hide_index=True)

                _build_metrics_chart(df_task)

from __future__ import annotations

import pandas as pd
import streamlit as st

from typing import List, Optional
from biotrainer_core.data_classes.autoeval import AutoEvalReport

from biotrainer_core.data_classes.autoeval.autoeval_report import _aggregate_dfs
from ..model import DashboardReport

from ..state import AutoevalSessionState

from ..plots import (
    plot_comparison,
    plot_delta_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
    compute_paired_delta_stats,
)

from ...autoeval_frameworks import AvailableFramework


def _render_framework_comparison(chosen_reports: List[AutoEvalReport],
                                 framework_name: str,
                                 df_fw: Optional[pd.DataFrame],
                                 choose_tasks_manually: bool = True,
                                 baseline_model: Optional[str] = None):
    st.markdown(f"#### {framework_name}")
    if df_fw is None or df_fw.empty:
        st.caption("No overlapping tasks to compare.")
        return

    # Task selection
    task_set = list(set(df_fw["TaskLabel"].unique()))
    task_set = sorted(task_set)
    if choose_tasks_manually:
        chosen_tasks = st.multiselect("Select tasks", key=f"multiselect_{framework_name}_{str(df_fw)}",
                                      default=task_set, options=task_set,
                                      # format_func=lambda
                                      #    task: f"{df_fw[df_fw['TaskLabel'] == task]['Test Set'].iloc[0]}-"
                                      #          f"{df_fw[df_fw['TaskLabel'] == task]['Task'].iloc[0].replace('PBC-', '')}"  #TODO
                                      )
    else:
        chosen_tasks = list(task_set)

    if len(chosen_tasks) == 0:
        st.info("Select at least two tasks to compare.")
        return

    df_fw = df_fw[df_fw["TaskLabel"].isin(chosen_tasks)]

    # Wide table
    index_cols = ["Task", "Test Set", "Metric"] if "Test Set" in df_fw.columns else ["Task", "Metric"]
    pivot = (
        df_fw.pivot_table(index=index_cols, columns="Model", values="Mean", aggfunc="first",
                          sort=False)
        .reset_index()
    )
    st.dataframe(pivot, use_container_width=True)

    st.markdown("**Absolute Comparison**")
    fig, ax = plot_comparison(df_fw)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig),
                           file_name=f"{framework_name}_absolute_comparison.png",
                           mime="image/png", key=f"abs_comp_png_{framework_name}_{str(df_fw)}")
        st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig),
                           file_name=f"{framework_name}_absolute_comparison.pdf",
                           mime="application/pdf", key=f"abs_comp_pdf_{framework_name}_{str(df_fw)}")
    else:
        st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

    if baseline_model is None or len(baseline_model) == 0:
        return

    # Delta Plots
    if framework_name != "PBC_SUPERVISED":
        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        paired_stats = compute_paired_delta_stats(
            {report.embedder_name: report.maybe_framework_result(framework_name)
             for report in chosen_reports if report.maybe_framework_result(framework_name) is not None},
            baseline_model,
        )
        fig_delta, ax_delta = plot_delta_comparison(df_fw, baseline_model, paired_stats=paired_stats)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name=f"{framework_name}_delta_comparison.png",
                               mime="image/png", key=f"delta_comp_png_{framework_name}_{str(df_fw)}")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name=f"{framework_name}_delta_comparison.pdf",
                               mime="application/pdf", key=f"delta_comp_pdf_{framework_name}_{str(df_fw)}")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")


def _aggregate_for_comparison(framework_name: str,
                              chosen_reports: List[AutoEvalReport],
                              dev_mode: bool):
    fw_results = [
        (report.maybe_framework_result(framework_name), report) for report in chosen_reports
    ]
    df = _aggregate_dfs([
        fw_result.to_df(all_metrics=False, development_mode=dev_mode).assign(
            Model=report.embedder_name)
        for fw_result, report in fw_results if fw_result is not None
    ])
    return df


def render_compare(state: AutoevalSessionState, active: List[DashboardReport]):
    active = [db_report.report for db_report in active]

    st.subheader("Compare Multiple Reports")
    if len(active) < 2:
        st.info("Load at least two reports to compare.")
        return

    # Report Selection
    default_chosen = state.get_compare_selected_reports()
    default_chosen = default_chosen if len(default_chosen) > 0 else active
    default_chosen = sorted(default_chosen, key=lambda report: report.embedder_name)
    chosen_reports = st.multiselect("Select reports", default=default_chosen, options=active,
                                    format_func=lambda report: report.embedder_name,
                                    on_change=lambda: state.set_compare_selected_reports(
                                        st.session_state.get("multiselect_compare_reports", [])),
                                    key="multiselect_compare_reports")

    if len(chosen_reports) == 1:
        st.info("Select at least two reports to compare.")
        st.stop()
    dev_mode = state.get_development_mode()

    # Baseline selection
    baseline_options = [report.embedder_name for report in chosen_reports]
    baseline_model = st.selectbox("Select baseline model for delta plots", options=baseline_options)

    # Define the framework configurations
    frameworks = [
        ("Supervised", AvailableFramework.PBC_SUPERVISED.name),
        ("Zero-Shot", AvailableFramework.PGYM.name),
        ("Zero-Shot Contact", AvailableFramework.PBC_ZEROSHOT_CONTACT.name),
        ("Supervised Contact", AvailableFramework.PBC_SUPERVISED_CONTACT.name),
    ]

    # Create tabs
    tabs = st.tabs([label for label, _ in frameworks])

    for tab, (label, framework_name) in zip(tabs, frameworks):
        with tab:
            df_fw = _aggregate_for_comparison(
                framework_name=framework_name,
                chosen_reports=chosen_reports,
                dev_mode=dev_mode
            )
            _render_framework_comparison(
                chosen_reports=chosen_reports,
                framework_name=framework_name,
                df_fw=df_fw,
                baseline_model=baseline_model
            )

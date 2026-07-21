from __future__ import annotations

from typing import List
from biotrainer_core.data_classes.autoeval import AutoEvalReport

from biotrainer_core.data_classes.autoeval.autoeval_report import _aggregate_dfs

try:
    import streamlit as st
except Exception:  # pragma: no cover
    raise

from ..state import AutoevalSessionState

from ...pipelines.autoeval_plotting import (
    plot_comparison,
    plot_delta_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
    compute_paired_delta_stats,
)

from ...autoeval_frameworks import AvailableFramework


def _render_supervised_comparison(chosen_reports: List[AutoEvalReport], baseline_model: str, dev_mode: bool):
    framework_name = AvailableFramework.PBC_SUPERVISED.name
    st.markdown(f"#### {framework_name}")
    df_sup = _aggregate_dfs([
        report.supervised_results[framework_name].to_df(framework=framework_name, development_mode=dev_mode).assign(
            Model=report.embedder_name)
        for report in chosen_reports if framework_name in report.supervised_results
    ])
    if df_sup is None or df_sup.empty:
        st.caption("No overlapping supervised tasks to compare.")
    else:
        # Task selection
        task_set = list(set(df_sup["TaskLabel"].unique()))
        task_set = sorted(task_set)
        chosen_tasks = st.multiselect("Select tasks", key=f"multiselect_sup_{len(task_set)}",
                                      default=task_set, options=task_set,
                                      format_func=lambda
                                          task: f"{df_sup[df_sup['TaskLabel'] == task]['Test Set'].iloc[0]}-"
                                                f"{df_sup[df_sup['TaskLabel'] == task]['Task'].iloc[0].replace('PBC-', '')}"
                                      )

        if len(chosen_tasks) == 0:
            st.info("Select at least two tasks to compare.")
            return
        df_sup = df_sup[df_sup["TaskLabel"].isin(chosen_tasks)]

        # Wide table
        pivot = (
            df_sup.pivot_table(index=["Task", "Test Set", "Metric"], columns="Model", values="Mean", aggfunc="first",
                               sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)

        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_sup)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="supervised_comparison.png",
                               mime="image/png", key="sup_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="supervised_comparison.pdf",
                               mime="application/pdf", key="sup_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        paired_stats = compute_paired_delta_stats(
            {report.embedder_name: report.zeroshot_results[framework_name]
             for report in chosen_reports if framework_name in report.zeroshot_results},
            baseline_model,
        )
        fig_delta, ax_delta = plot_delta_comparison(df_sup, baseline_model, paired_stats=paired_stats)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="supervised_delta_comparison.png",
                               mime="image/png", key="sup_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="supervised_delta_comparison.pdf",
                               mime="application/pdf", key="sup_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")


def _render_zero_shot_comparison(chosen_reports: List[AutoEvalReport], baseline_model: str, dev_mode: bool):
    framework_name = AvailableFramework.PGYM.name
    st.markdown(f"#### Zero-Shot {framework_name}")
    df_zero = _aggregate_dfs([
        report.zeroshot_results[framework_name].to_df(framework=framework_name, development_mode=dev_mode).assign(
            Model=report.embedder_name)
        for report in chosen_reports if framework_name in report.zeroshot_results
    ])
    if df_zero is None or df_zero.empty:
        st.caption("No overlapping zero-shot tasks to compare.")
    else:
        # Task selection
        task_set_zs = list(set(df_zero["Task"].unique()))
        chosen_tasks_zs = st.multiselect("Select tasks", key=f"multiselect_zs_{len(task_set_zs)}",
                                         default=task_set_zs, options=task_set_zs)

        if len(chosen_tasks_zs) == 0:
            st.info("Select at least two tasks to compare.")
            return
        df_zero = df_zero[df_zero["Task"].isin(chosen_tasks_zs)]

        pivot = (
            df_zero.pivot_table(index=["Task", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)

        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_zero)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="zeroshot_comparison.png",
                               mime="image/png", key="zero_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="zeroshot_comparison.pdf",
                               mime="application/pdf", key="zero_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        paired_stats = compute_paired_delta_stats(
            {report.embedder_name: report.zeroshot_results[framework_name]
             for report in chosen_reports if framework_name in report.zeroshot_results},
            baseline_model,
        )
        fig_delta, ax_delta = plot_delta_comparison(df_zero, baseline_model, paired_stats=paired_stats)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="zeroshot_delta_comparison.png",
                               mime="image/png", key="zero_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="zeroshot_delta_comparison.pdf",
                               mime="application/pdf", key="zero_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")


def _render_zeroshot_contact_comparison(chosen_reports: List[AutoEvalReport], baseline_model: str, dev_mode: bool):
    framework_name = AvailableFramework.PBC_ZEROSHOT_CONTACT.name
    st.markdown(f"#### Zero-Shot Contact {framework_name}")
    df_zsc = _aggregate_dfs([
        report.zeroshot_contact_results[framework_name].to_df(framework=framework_name, development_mode=dev_mode).assign(
            Model=report.embedder_name)
        for report in chosen_reports if framework_name in report.zeroshot_contact_results
    ])
    if df_zsc is None or df_zsc.empty:
        st.caption("No overlapping zero-shot contact tasks to compare.")
    else:
        # Task selection
        task_set_zsc = list(set(df_zsc["Task"].unique()))
        chosen_tasks_zsc = st.multiselect("Select tasks", key=f"multiselect_zsc_{len(task_set_zsc)}",
                                          default=task_set_zsc, options=task_set_zsc)

        if len(chosen_tasks_zsc) == 0:
            st.info("Select at least two tasks to compare.")
            return
        df_zsc = df_zsc[df_zsc["Task"].isin(chosen_tasks_zsc)]

        pivot = (
            df_zsc.pivot_table(index=["Task", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)

        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_zsc)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="zeroshot_contact_comparison.png",
                               mime="image/png", key="zsc_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="zeroshot_contact_comparison.pdf",
                               mime="application/pdf", key="zsc_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        paired_stats = compute_paired_delta_stats(
            {report.embedder_name: report.zeroshot_results[framework_name]
             for report in chosen_reports if framework_name in report.zeroshot_results},
            baseline_model,
        )
        fig_delta, ax_delta = plot_delta_comparison(df_zsc, baseline_model, paired_stats=paired_stats)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="zeroshot_contact_delta_comparison.png",
                               mime="image/png", key="zsc_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="zeroshot_contact_delta_comparison.pdf",
                               mime="application/pdf", key="zsc_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")


def _render_supervised_contact_comparison(chosen_reports: List[AutoEvalReport], baseline_model: str, dev_mode: bool):
    framework_name = AvailableFramework.PBC_SUPERVISED_CONTACT.name
    st.markdown(f"#### Supervised Contact {framework_name}")
    df_sc = _aggregate_dfs([
        report.supervised_contact_results[framework_name].to_df(framework=framework_name, development_mode=dev_mode).assign(
            Model=report.embedder_name)
        for report in chosen_reports if framework_name in report.supervised_contact_results
    ])
    if df_sc is None or df_sc.empty:
        st.caption("No overlapping supervised contact tasks to compare.")
    else:
        # Task selection
        task_set_sc = list(set(df_sc["Task"].unique()))
        chosen_tasks_sc = st.multiselect("Select tasks", key=f"multiselect_sc_{len(task_set_sc)}",
                                         default=task_set_sc, options=task_set_sc)

        if len(chosen_tasks_sc) == 0:
            st.info("Select at least two tasks to compare.")
            return
        df_sc = df_sc[df_sc["Task"].isin(chosen_tasks_sc)]

        pivot = (
            df_sc.pivot_table(index=["Task", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)

        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_sc)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="supervised_contact_comparison.png",
                               mime="image/png", key="sc_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="supervised_contact_comparison.pdf",
                               mime="application/pdf", key="sc_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        paired_stats = compute_paired_delta_stats(
            {report.embedder_name: report.zeroshot_results[framework_name]
             for report in chosen_reports if framework_name in report.zeroshot_results},
            baseline_model,
        )
        fig_delta, ax_delta = plot_delta_comparison(df_sc, baseline_model, paired_stats=paired_stats)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="supervised_contact_delta_comparison.png",
                               mime="image/png", key="sc_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="supervised_contact_delta_comparison.pdf",
                               mime="application/pdf", key="sc_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")


def render_compare(state: AutoevalSessionState, active: List[AutoEvalReport]):
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

    tab_sup, tab_zs, tab_zsc, tab_sc = st.tabs(["Supervised", "Zero-Shot", "Zero-Shot Contact", "Supervised Contact"])
    with tab_sup:
        _render_supervised_comparison(chosen_reports, baseline_model, dev_mode)
    with tab_zs:
        _render_zero_shot_comparison(chosen_reports, baseline_model, dev_mode)
    with tab_zsc:
        _render_zeroshot_contact_comparison(chosen_reports, baseline_model, dev_mode)
    with tab_sc:
        _render_supervised_contact_comparison(chosen_reports, baseline_model, dev_mode)

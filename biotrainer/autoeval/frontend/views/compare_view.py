from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    raise

from ..utils.plotting import (
    build_supervised_compare_df,
    build_zeroshot_compare_df,
    plot_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
)


def render_compare(loaded):
    st.subheader("Compare Multiple Reports")
    if len(loaded) < 2:
        st.info("Select at least two reports to compare.")
        return

    labels = [f"{it.report.embedder_name} ({it.report.training_date}) — {it.path.name}" for it in loaded]
    idxs = st.multiselect("Select reports", options=list(range(len(loaded))), format_func=lambda i: labels[i])
    if len(idxs) < 2:
        st.stop()

    chosen = [loaded[i] for i in idxs]

    # Supervised comparison
    st.markdown("#### Supervised (PBC)")
    df_sup = build_supervised_compare_df(chosen, framework="PBC")
    if df_sup is None or df_sup.empty:
        st.caption("No overlapping supervised tasks to compare.")
    else:
        # Wide table similar to previous pivot
        pivot = (
            df_sup.pivot_table(index=["Task", "Test Set", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)
        fig, ax = plot_comparison(df_sup, title="Performance Comparison Across Tasks")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="supervised_comparison.png", mime="image/png")
            with col2:
                st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="supervised_comparison.pdf", mime="application/pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

    # Zeroshot comparison
    st.markdown("#### Zero-Shot (PGYM)")
    df_zero = build_zeroshot_compare_df(chosen, framework="PGYM")
    if df_zero is None or df_zero.empty:
        st.caption("No overlapping zero-shot tasks to compare.")
    else:
        pivot = (
            df_zero.pivot_table(index=["Task", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)
        fig, ax = plot_comparison(df_zero, title="Performance Comparison Across Tasks")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="zeroshot_comparison.png", mime="image/png")
            with col2:
                st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="zeroshot_comparison.pdf", mime="application/pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

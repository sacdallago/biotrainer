from __future__ import annotations

from typing import Optional, List

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ..utils import constants as frontend_constants
from ..utils.plotting import (
    build_supervised_compare_df,
    build_zeroshot_compare_df,
    plot_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
)
from ..utils.utils import LoadedReport


def render_leaderboard(df_lb: pd.DataFrame, loaded: List[LoadedReport]):
    st.subheader("Leaderboard (average rank across tasks)")
    if df_lb is None or df_lb.empty:
        st.info("No data to display. Load reports to see the leaderboard.")
        return

    fw = st.selectbox("Framework", options=frontend_constants.SUPPORTED_FRAMEWORKS, index=0)
    sub = df_lb[df_lb["Framework"] == fw].copy()
    st.dataframe(sub, use_container_width=True, hide_index=True)

    # Bar chart: Avg Rank (lower is better) — we invert for visualization (score = -Avg Rank)
    try:
        import altair as alt  # optional

        sub_chart = sub.copy()
        sub_chart["Score"] = -sub_chart["Avg Rank"].astype(float)
        chart = (
            alt.Chart(sub_chart)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", sort="-y"),
                y=alt.Y("Score:Q", title="- Avg Rank (higher is better)"),
                tooltip=["Model", "Avg Rank", "Num Tasks"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.caption("Install 'altair' for charts, or rely on the table above.")

    st.markdown("#### Overall task comparison")
    # Render the AutoEval-style comparison plot for the chosen framework
    try:
        if fw.upper() == "PBC":
            df_plot = build_supervised_compare_df(loaded, framework="PBC")
        else:
            df_plot = build_zeroshot_compare_df(loaded, framework="PGYM")

        if df_plot is None or df_plot.empty:
            st.caption("No overlapping tasks available for a comparison plot.")
        else:
            fig, ax = plot_comparison(df_plot, title="Performance Comparison Across Tasks")
            if fig is None:
                st.info("Install 'matplotlib' and 'seaborn' to see the comparison plot.")
            else:
                st.pyplot(fig, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        "⬇️ Download PNG",
                        data=fig_to_png_bytes(fig),
                        file_name="comparison.png",
                        mime="image/png",
                    )
                with col_b:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=fig_to_pdf_bytes(fig),
                        file_name="comparison.pdf",
                        mime="application/pdf",
                    )
    except Exception:
        st.caption("Unable to render comparison plot.")

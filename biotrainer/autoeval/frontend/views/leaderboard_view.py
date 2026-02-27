from __future__ import annotations

from typing import Optional, List

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ..utils import constants as frontend_constants
from ..utils.utils import LoadedReport

from ...pipelines.autoeval_plotting import plot_comparison, fig_to_png_bytes, fig_to_pdf_bytes, aggregate_dfs


def render_leaderboard(df_lb: pd.DataFrame, loaded: List[LoadedReport]):
    st.subheader("Leaderboard (average rank across tasks)")
    if df_lb is None or df_lb.empty:
        st.info("No data to display. Load reports to see the leaderboard.")
        return

    fw = str(st.selectbox("Framework", options=frontend_constants.SUPPORTED_FRAMEWORKS, index=0)).upper()
    sub = df_lb[df_lb["Framework"] == fw].copy()
    st.dataframe(sub, use_container_width=True, hide_index=True)

    st.markdown("#### Overall task comparison")
    # Render the AutoEval-style comparison plot for the chosen framework
    try:
        if fw == "PBC":
            df_plot = aggregate_dfs([
                l.report.supervised_results[fw].to_df(framework=fw).assign(Model=l.report.embedder_name)
                for l in loaded if fw in l.report.supervised_results
            ])
        else:
            df_plot = aggregate_dfs([
                l.report.zeroshot_results[fw].to_df(framework=fw).assign(Model=l.report.embedder_name)
                for l in loaded if fw in l.report.zeroshot_results
            ])

        if df_plot is None or df_plot.empty:
            st.caption("No overlapping tasks available for a comparison plot.")
        else:
            fig, ax = plot_comparison(df_plot)
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
    except Exception as e:
        st.caption("Unable to render comparison plot.")
        print(e)
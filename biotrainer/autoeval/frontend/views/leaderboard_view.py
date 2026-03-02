from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - runtime import guard
    raise

from ..utils import constants as frontend_constants
from ..utils.utils import LoadedReport

from ...pipelines.autoeval_plotting import (
    plot_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
    aggregate_dfs,
)

from ....utilities.ranking import Ranking


# =========================
# Helper functions
# =========================

def _init_state(categories: List[str]):
    """Initialize Streamlit session state for leaderboard UI."""
    if "lb_selected_fw" not in st.session_state:
        st.session_state.lb_selected_fw = str(frontend_constants.SUPPORTED_FRAMEWORKS[0]).upper()
    if "lb_selected_ranking" not in st.session_state:
        st.session_state.lb_selected_ranking = "global"
    if "lb_weights" not in st.session_state:
        st.session_state.lb_weights = {c: 0 for c in categories}


def _build_title():
    st.subheader("Leaderboard")


def _get_rank_color(place: int) -> str:
    # Return hex colors to use in HTML badges
    zero_indexed = place - 1
    return ["#ffa000", "#9e9e9e", "#cd7f32"][zero_indexed] if zero_indexed < 3 else "#1976d2"


def _badge(place: int) -> str:
    color = _get_rank_color(place)
    return f"""
    <div style='width:32px;height:32px;border-radius:50%;background:{color};display:flex;align-items:center;justify-content:center;'>
      <span style='color:white;font-weight:700'>{place}</span>
    </div>
    """


def _build_framework_selector() -> str:
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown("**Framework**")
    with cols[1]:
        st.session_state.lb_selected_fw = str(
            st.selectbox(
                label="Framework",
                label_visibility="collapsed",
                options=frontend_constants.SUPPORTED_FRAMEWORKS,
                index=max(0, list(map(str.upper, frontend_constants.SUPPORTED_FRAMEWORKS)).index(
                    st.session_state.lb_selected_fw))
                if st.session_state.lb_selected_fw in list(map(str.upper, frontend_constants.SUPPORTED_FRAMEWORKS))
                else 0,
            )
        ).upper()
    return st.session_state.lb_selected_fw


def _build_information(ranking: Ranking):
    with st.expander("Information", expanded=False):
        st.write(
            f"Theoretical Highest Ranking Score: {ranking.maximum_ranking_value:.1f}"
        )
        st.write("Higher ranking is better")
        st.markdown("**Categories**")
        for cat in ranking.categories:
            st.caption(cat)


def _build_ranking_selection(ranking: Ranking):
    options = ["global"] + list(sorted(ranking.raw_categories))
    idx = options.index(st.session_state.lb_selected_ranking) if st.session_state.lb_selected_ranking in options else 0
    st.session_state.lb_selected_ranking = st.selectbox(
        "Select ranking",
        options=options,
        index=idx,
    )


def _build_weights_selection(ranking: Ranking):
    with st.expander("Change weights", expanded=False):
        cols = st.columns(2)
        for i, cat in enumerate(sorted(ranking.ranking_categories)):
            with cols[i % 2]:
                current = st.session_state.lb_weights.get(cat, 0)
                new_val = st.number_input(
                    f"{cat}", min_value=0, max_value=10, step=1, value=int(current), key=f"w_{cat}"
                )
                st.session_state.lb_weights[cat] = int(new_val)
                st.caption(f"{Ranking.get_score_multiplier(int(new_val)):.1f}x counted")


def _group_by_place(ranking_list: List[Tuple[int, object, float]]):
    grouped: Dict[int, List[Tuple[int, object, float]]] = {}
    for entry in ranking_list:
        place = entry[0]
        grouped.setdefault(place, []).append(entry)
    return grouped


def _tile_for_entry(ranking: Ranking, entry: Tuple[int, object, float]):
    place, ranking_entry, score = entry
    # Layout: badge | name | score
    cols = st.columns([0.2, 0.6, 0.2], gap="small")
    with cols[0]:
        st.markdown(_badge(place), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"**{ranking_entry.name}**")
    with cols[2]:
        verbose = ranking.verbose_ranking_by_entry(ranking_entry.name) or "No details available."
        score = f"**{score:.1f}**"
        with st.popover(score):
            st.text(verbose)


def _build_ranking_visualization(ranking: Ranking, ranking_list: List[Tuple[int, object, float]]):
    cols = st.columns([0.2, 0.6, 0.2], gap="small")
    with cols[0]:
        st.markdown("**Rank**")
    with cols[1]:
        st.markdown("**Model**")
    with cols[2]:
        st.markdown("**Score**")

    grouped = _group_by_place(ranking_list)
    for place, entries in grouped.items():
        if len(entries) > 1:
            st.markdown(f"— Tie for place {place} —", help="Multiple entries tied.")
        for e in entries:
            _tile_for_entry(ranking, e)
        # Use a thinner divider or conditional spacing for the next row
        if place < len(grouped):
            st.markdown("<hr style='margin:4px 0;'>", unsafe_allow_html=True)  # Minimal divider spacing


def _build_leaderboard_visualization(ranking: Ranking, leaderboard):
    _build_ranking_visualization(ranking, leaderboard)


def _build_category_visualization(category: str, ranking: Ranking):
    category_ranking = ranking.get_category_ranking(category=category)
    if category_ranking is None:
        st.warning(f"ERROR: No ranking found for category {category}!")
        return
    _build_ranking_visualization(ranking, category_ranking)


def _copy_ranking_controls(ranking: Ranking):
    # Clipboard access is restricted in browsers; provide a download + selectable text
    text = ranking.copied_ranking()
    st.download_button("⬇️ Download ranking.txt", data=text, file_name="ranking.txt", mime="text/plain")


# =========================
# Public entry point
# =========================

def render_leaderboard(ranking_pbc: Ranking, ranking_pgym: Ranking, loaded: List[LoadedReport]):
    # determine active ranking based on framework
    all_categories = sorted(list(ranking_pbc.ranking_categories.union(ranking_pgym.ranking_categories)))
    _init_state(all_categories)

    _build_title()
    fw = _build_framework_selector()
    ranking = ranking_pbc if fw == "PBC" else ranking_pgym

    # Sync weight keys with current ranking
    for cat in ranking.ranking_categories:
        st.session_state.lb_weights.setdefault(cat, 0)

    # Apply weights to current ranking object
    weighted_ranking = ranking.update_weights(st.session_state.lb_weights)

    _build_ranking_selection(weighted_ranking)

    leaderboard = weighted_ranking.get_leaderboard_ranking()

    if st.session_state.lb_selected_ranking == "global":
        _build_leaderboard_visualization(weighted_ranking, leaderboard)
    else:
        _build_category_visualization(st.session_state.lb_selected_ranking, weighted_ranking)

    if st.session_state.lb_selected_ranking == "global":
        cols = st.columns([1, 1])
        with cols[0]:
            _build_information(weighted_ranking)
        with cols[1]:
            _build_weights_selection(weighted_ranking)
    else:
        _build_information(weighted_ranking)

    _copy_ranking_controls(weighted_ranking)

    # Comparison plot section
    st.markdown("#### Overall task comparison")
    plot_maximum = st.slider("Select the maximum number of models to compare:", min_value=1, max_value=6, value=6)
    best_n_models = [entry[1].name.lower() for entry in leaderboard[:plot_maximum]]
    try:
        if fw == "PBC":
            dfs = [
                l.report.supervised_results[fw].to_df(framework=fw).assign(Model=l.report.embedder_name)
                for l in loaded
                if fw in l.report.supervised_results and l.report.embedder_name.lower() in best_n_models
            ]
            dfs = sorted(dfs, key=lambda df: best_n_models.index(df["Model"].str.lower().iloc[0]), reverse=True)
            df_plot = aggregate_dfs(dfs)
        else:
            dfs = [
                    l.report.zeroshot_results[fw].to_df(framework=fw).assign(Model=l.report.embedder_name)
                    for l in loaded
                    if fw in l.report.zeroshot_results and l.report.embedder_name.lower() in best_n_models
                ]
            dfs = sorted(dfs, key=lambda df: best_n_models.index(df["Model"].str.lower().iloc[0]), reverse=True)
            df_plot = aggregate_dfs(dfs)

        if df_plot is None or df_plot.empty:
            st.caption("No overlapping tasks available for a comparison plot.")
        else:
            fig, _ = plot_comparison(df_plot)
            if fig is None:
                st.info("Install 'matplotlib' and 'seaborn' to see the comparison plot.")
            else:
                st.pyplot(fig, use_container_width=True)
                st.download_button(
                    "⬇️ Download PNG",
                    data=fig_to_png_bytes(fig),
                    file_name="comparison.png",
                    mime="image/png",
                )
                st.download_button(
                    "⬇️ Download PDF",
                    data=fig_to_pdf_bytes(fig),
                    file_name="comparison.pdf",
                    mime="application/pdf",
                )
    except Exception as e:
        st.caption("Unable to render comparison plot.")
        print(e)

from __future__ import annotations

import streamlit as st

from typing import Dict, List, Tuple
from biotrainer_core.functions.ranking import Ranking
from biotrainer_core.data_classes.autoeval import AutoEvalReport
from biotrainer_core.data_classes.autoeval.autoeval_report import _aggregate_dfs
from ..model import DashboardReport

from ..state import AutoevalSessionState

from ...autoeval_frameworks import AvailableFramework

from ..plots import (
    plot_comparison,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
)

# =========================
# Helper functions
# =========================


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


def _build_framework_selector(state: AutoevalSessionState) -> AvailableFramework:
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown("**Framework**")
    with cols[1]:
        currently_selected = state.get_lb_framework()
        all_frameworks = list(map(lambda fw: fw.value.upper(), AvailableFramework.dashboard_frameworks()))
        selected_framework = st.selectbox(
                label="Framework",
                label_visibility="collapsed",
                options=all_frameworks,
                index=max(0, all_frameworks.index(currently_selected))
                if currently_selected in all_frameworks else 0,
            )
        selected_framework = AvailableFramework[selected_framework.upper()]
        state.select_lb_framework(selected_framework)
    return state.get_lb_framework()


def _build_information(ranking: Ranking):
    with st.expander("Information", expanded=False):
        st.write(
            f"Theoretical Highest Ranking Score: {ranking.maximum_ranking_value:.1f}"
        )
        st.write("Higher ranking is better")
        st.markdown("**Categories**")
        for cat in ranking.categories:
            st.caption(cat)


def _build_ranking_category_selection(state: AutoevalSessionState, ranking: Ranking) -> str:
    options = ["global"] + list(sorted(ranking.raw_categories | ranking.ranking_categories))
    currently_selected = state.get_lb_ranking_category()
    idx = options.index(currently_selected) if currently_selected in options else 0
    selected_ranking_category = st.selectbox(
        "Select ranking category",
        options=options,
        index=idx,
    )
    state.select_lb_ranking_category(selected_ranking_category)
    return selected_ranking_category


def _build_weights_selection(state: AutoevalSessionState, ranking: Ranking):
    with st.expander("Change weights", expanded=False):
        cols = st.columns(2)
        for i, cat in enumerate(sorted(ranking.ranking_categories)):
            with cols[i % 2]:
                current = state.get_lb_weight(cat)
                st.write(
                    f"Weight for {cat}: {current}"
                )
                new_val = st.number_input(
                    f"{cat}", min_value=0, max_value=10, step=1, value=int(current), key=f"w_{cat}"
                )
                new_val = int(str(new_val))
                state.set_lb_weight(cat, new_val)
                st.caption(f"{Ranking.get_score_multiplier(new_val):.1f}x counted")


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
        score = f"**{score:.2f}**"
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

def render_leaderboard(state: AutoevalSessionState,
                       ranking_pbc: Ranking,
                       ranking_pgym: Ranking,
                       active: List[DashboardReport],
                       development_mode: bool):
    # determine active ranking based on framework
    active = [db_report.report for db_report in active]  # TODO: Add tooltips

    all_categories = sorted(list(ranking_pbc.ranking_categories.union(ranking_pgym.ranking_categories)))
    state.maybe_init_lb_weights(all_categories)

    _build_title()
    fw = _build_framework_selector(state)
    ranking = ranking_pbc if fw == AvailableFramework.PBC_SUPERVISED else ranking_pgym  # TODO

    # Sync weight keys with current ranking
    state.sync_lb_weights(ranking.ranking_categories)

    # Apply weights to current ranking object
    weighted_ranking = ranking.update_weights(state.get_lb_weights())

    selected_ranking_category = _build_ranking_category_selection(state, weighted_ranking)

    leaderboard = weighted_ranking.get_leaderboard_ranking()

    if selected_ranking_category == "global":
        _build_leaderboard_visualization(weighted_ranking, leaderboard)
    else:
        _build_category_visualization(selected_ranking_category, weighted_ranking)

    if selected_ranking_category == "global":
        cols = st.columns([1, 1])
        with cols[0]:
            _build_information(weighted_ranking)
        with cols[1]:
            _build_weights_selection(state, weighted_ranking)
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
                report.supervised_results[fw].to_df(all_metrics=False, development_mode=development_mode).assign(
                    Model=report.embedder_name)
                for report in active
                if fw in report.supervised_results and report.embedder_name.lower() in best_n_models
            ]
            dfs = sorted(dfs, key=lambda df: best_n_models.index(df["Model"].str.lower().iloc[0]), reverse=True)
            df_plot = _aggregate_dfs(dfs)
        else:
            dfs = [
                report.zeroshot_results[fw].to_df(all_metrics=False, development_mode=development_mode).assign(
                    Model=report.embedder_name)
                for report in active
                if fw in report.zeroshot_results and report.embedder_name.lower() in best_n_models
            ]
            dfs = sorted(dfs, key=lambda df: best_n_models.index(df["Model"].str.lower().iloc[0]), reverse=True)
            df_plot = _aggregate_dfs(dfs)

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

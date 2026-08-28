from __future__ import annotations

import streamlit as st

from typing import Dict, List, Tuple, Optional
from .publish_dialog import publish_dialog
from biotrainer_core.functions.ranking import Ranking, RankingEntry

from .compare_view import _render_framework_comparison, _aggregate_for_comparison

from ..model import DashboardReport

from ..state import AutoevalSessionState

from ...autoeval_frameworks import AvailableFramework
from ...client import AutoEvalClient


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


def _group_by_place(ranking_list: List[Tuple[int, RankingEntry, float]]):
    grouped: Dict[int, List[Tuple[int, RankingEntry, float]]] = {}
    for entry in ranking_list:
        place = entry[0]
        grouped.setdefault(place, []).append(entry)
    return grouped


def _ranking_row_cols_spec(any_local_reports: bool):
    return [0.2, 0.5, 0.2, 0.1] if any_local_reports else [0.2, 0.6, 0.2]


def _ranking_entry_tile(ranking: Ranking, entry: Tuple[int, RankingEntry, float],
                        embedder_name_to_db_report: Dict[str, DashboardReport],
                        any_local_reports: bool,
                        state: Optional[AutoevalSessionState] = None,
                        client: Optional[AutoEvalClient] = None):
    place, ranking_entry, score = entry
    embedder_name = ranking_entry.name
    db_report = embedder_name_to_db_report.get(embedder_name, None)

    # Layout: badge | name | score
    cols = st.columns(_ranking_row_cols_spec(any_local_reports), gap="small")
    with cols[0]:
        st.markdown(_badge(place), unsafe_allow_html=True)
    with cols[1]:
        tooltip = None
        if db_report:
            tooltip = db_report.tooltip()
        st.markdown(f"**{embedder_name}**", help=tooltip)
    with cols[2]:
        verbose = ranking.verbose_ranking_by_entry(ranking_entry.name) or "No details available."
        score = f"**{score:.2f}**"
        with st.popover(score):
            st.markdown(verbose)

    if any_local_reports:
        with cols[3]:
            if db_report and db_report.is_loaded:
                if st.button("Publish", key=f"publish_btn_{place}_{embedder_name}_{db_report.report.get_uid()}",
                             use_container_width=True):
                    publish_dialog(db_report.report, state=state, client=client)
            else:
                st.markdown("")


def _build_ranking_visualization(ranking: Ranking, ranking_list: List[Tuple[int, RankingEntry, float]],
                                 embedder_name_to_db_report: Dict[str, DashboardReport],
                                 state: Optional[AutoevalSessionState] = None,
                                 client: Optional[AutoEvalClient] = None):
    any_local_reports = any([db_report.is_loaded for db_report in embedder_name_to_db_report.values()])

    cols = st.columns(_ranking_row_cols_spec(any_local_reports), gap="small")
    with cols[0]:
        st.markdown("**Rank**")
    with cols[1]:
        st.markdown("**Model**")
    with cols[2]:
        st.markdown("**Score**")

    if any_local_reports:
        with cols[3]:
            st.markdown("**Publish**")

    grouped = _group_by_place(ranking_list)
    for place, entries in grouped.items():
        if len(entries) > 1:
            st.markdown(f"— Tie for place {place} —", help="Multiple entries tied.")
        for e in entries:
            _ranking_entry_tile(ranking, e, embedder_name_to_db_report, any_local_reports, state=state, client=client)
        # Use a thinner divider or conditional spacing for the next row
        if place < len(grouped):
            st.markdown("<hr style='margin:4px 0;'>", unsafe_allow_html=True)  # Minimal divider spacing


def _build_leaderboard_visualization(ranking: Ranking, leaderboard,
                                     embedder_name_to_db_report: Dict[str, DashboardReport],
                                     state: Optional[AutoevalSessionState] = None,
                                     client: Optional[AutoEvalClient] = None):
    _build_ranking_visualization(ranking, leaderboard, embedder_name_to_db_report, state=state, client=client)


def _build_category_visualization(category: str, ranking: Ranking,
                                  embedder_name_to_db_report: Dict[str, DashboardReport],
                                  state: Optional[AutoevalSessionState] = None,
                                  client: Optional[AutoEvalClient] = None):
    category_ranking = ranking.get_category_ranking(category=category)
    if category_ranking is None:
        st.warning(f"ERROR: No ranking found for category {category}!")
        return
    _build_ranking_visualization(ranking, category_ranking, embedder_name_to_db_report, state=state, client=client)


def _copy_ranking_controls(ranking: Ranking):
    # Clipboard access is restricted in browsers; provide a download + selectable text
    text = ranking.copied_ranking()
    st.download_button("⬇️ Download ranking.txt", data=text, file_name="ranking.txt", mime="text/plain")


# =========================
# Public entry point
# =========================

def render_leaderboard(state: AutoevalSessionState,
                       ranking_dict: Dict[str, Ranking],
                       active: List[DashboardReport],
                       development_mode: bool,
                       client: Optional[AutoEvalClient] = None):
    # determine active ranking based on framework
    embedder_name_to_db_report = {db_report.report.embedder_name: db_report for db_report in active}
    active = [db_report.report for db_report in active]

    all_rankings = list(ranking_dict.values())
    all_categories = sorted(list(set().union(*[ranking.ranking_categories for ranking in all_rankings])))
    state.maybe_init_lb_weights(all_categories)

    _build_title()
    fw = _build_framework_selector(state)
    selected_ranking = ranking_dict.get(fw.name)

    if selected_ranking is None:
        st.markdown("**No ranking available for selected framework**")
        return

    # Sync weight keys with current ranking
    state.sync_lb_weights(selected_ranking.ranking_categories)

    # Apply weights to current ranking object
    weighted_ranking = selected_ranking.update_weights(state.get_lb_weights())

    selected_ranking_category = _build_ranking_category_selection(state, weighted_ranking)

    leaderboard = weighted_ranking.get_leaderboard_ranking()

    if selected_ranking_category == "global":
        _build_leaderboard_visualization(weighted_ranking, leaderboard, embedder_name_to_db_report, state=state,
                                         client=client)
    else:
        _build_category_visualization(selected_ranking_category, weighted_ranking, embedder_name_to_db_report,
                                      state=state, client=client)

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
    best_n_models = [entry[1].name for entry in leaderboard[:plot_maximum]]
    best_reports = [report for report in active if report.embedder_name in set(best_n_models)]
    framework_name = fw.name
    df_sup = _aggregate_for_comparison(
        framework_name=framework_name,
        chosen_reports=best_reports,
        dev_mode=development_mode)
    _render_framework_comparison(chosen_reports=best_reports,
                                 framework_name=framework_name,
                                 df_fw=df_sup,
                                 baseline_model=best_n_models[0] if len(best_n_models) > 1 else None)

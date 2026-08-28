import streamlit as st

from typing import Optional
from biotrainer_core.data_classes.autoeval import AutoEvalReport

from ..state import AutoevalSessionState

from ...client.autoeval_client import AutoEvalClient


@st.dialog("Publish Report")
def publish_dialog(report: AutoEvalReport,
                   state: Optional[AutoevalSessionState] = None,
                   client: Optional[AutoEvalClient] = None):
    if client is None:
        client = AutoEvalClient()
    if state is None and "state" in st.session_state:
        state = st.session_state.state

    st.markdown(f"#### Publish `{report.embedder_name}`")

    name = st.text_input("Name*", placeholder="Your full name", key=f"pub_name_{report.get_uid()}")
    email = st.text_input("Email*", placeholder="Your email address", key=f"pub_email_{report.get_uid()}")
    citation = st.text_input("Citation (optional DOI)", placeholder="https://doi.org/...",
                             key=f"pub_citation_{report.get_uid()}")

    st.info(
        "Publishing this report will make it publicly visible on the public leaderboard. "
        "Your name and the citation will be displayed on the leaderboard. Your e-mail is used in case of questions "
        "that might come up when we add your results to the official leaderboard. "
        "Your data will only be used for the purpose of the autoeval dashboard and not be shared with third parties. "
        "Published reports might be removed anytime from the leaderboard."
    )

    terms = st.checkbox("I understand these terms and conditions and agree "
                        "to the storing of my data for the purpose of the leaderboard.",
                        key=f"pub_terms_{report.get_uid()}")

    published_key = f"published_status_{report.get_uid()}"
    is_published = st.session_state.get(published_key, False)

    if is_published:
        st.success("Report published successfully!")
        if st.button("Close", key=f"close_btn_{report.get_uid()}", use_container_width=True):
            st.session_state[published_key] = False
            st.rerun()
        return

    cols = st.columns(2)
    with cols[0]:
        publish_clicked = st.button("Publish now", type="primary", use_container_width=True,
                                    key=f"pub_now_{report.get_uid()}")
    with cols[1]:
        cancel_clicked = st.button("Cancel", use_container_width=True, key=f"pub_cancel_{report.get_uid()}")

    if cancel_clicked:
        st.rerun()

    if publish_clicked:
        if not terms:
            st.error("Please accept the terms and conditions to publish.")
            return

        if not name or len(name.strip()) < 3:
            st.error("Please provide a valid publisher name (at least 3 characters).")
            return

        if not email or "@" not in email or len(email.strip()) < 5:
            st.error("Please provide a valid email address.")
            return

        citation_clean = citation.strip() if citation and citation.strip() else None
        if citation_clean and not citation_clean.lower().startswith("https://doi.org/"):
            st.error("Citation must be a valid DOI URL starting with 'https://doi.org/'")
            return

        maybe_error = client.publish_report(
            report=report,
            name=name.strip(),
            email=email.strip(),
            citation=citation_clean
        )

        if maybe_error:
            st.error(f"Error publishing report: {maybe_error}")
        else:
            st.session_state[published_key] = True
            if state is not None:
                state.remove_loaded_report(report.get_uid())
                try:
                    public_reports = client.get_public_reports()
                    if public_reports:
                        state.add_published_reports(public_reports)
                except Exception as e:
                    st.error(f"Error fetching updated public reports: {e}")
            st.success("Report published successfully!")
            if st.button("Close", key=f"close_after_pub_{report.get_uid()}", use_container_width=True):
                st.session_state[published_key] = False
                st.rerun()
